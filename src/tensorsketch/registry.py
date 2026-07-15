"""Name → object, so a config file can pick your model and your database.

Everything first-party lives in the single ``tensorsketch`` package — nothing here fragments
TensorSketch into
sub-packages you assemble by hand. Extras like ``tensorsketch-core[anthropic]`` only gate a heavy
*third-party
SDK*, exactly as before. What this module adds is the ability to construct a built-in by **name**::

    from tensorsketch import create_provider, create_backend

    model = create_provider("anthropic", model="claude-sonnet-4-6")
    store = create_backend("postgres", dsn="postgresql://...")

so the choice can come from a config value or a CLI flag instead of an import. Names resolve
**lazily**: nothing is imported until you actually build it, so ``import tensorsketch`` still pulls
in no
optional dependency, and listing the registered names imports nothing at all.

To keep this deliberately small, only the two seams where name selection genuinely pays off get a
registry — **providers** and **backends**. Add your own name in-process::

    register_provider("acme", AcmeProvider)

or, if you publish a package, declare an entry point (still one ``pip install`` for your users, no
TensorSketch-side plumbing):

.. code-block:: toml

    [project.entry-points."tensorsketch.providers"]
    acme = "loom_acme:AcmeProvider"

``create_provider("acme", ...)`` then works. That's the whole extension story — named factories
with lazy loading, no plugin objects, no separate core/community split.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .providers.base import ChatProvider
    from .runtime.checkpoint import Backend

T = TypeVar("T")

# A factory returns the class (or any callable) to construct — *not* an instance. Built-ins
# register a thunk that performs the heavy import only when the factory is called.
Factory = Callable[[], "type[T]"]


class Registry(Generic[T]):
    """A tiny name → factory table with lazy resolution and entry-point discovery.

    A registered *factory* yields the class to build; `create(name, ...)` calls it and then
    constructs the instance. Built-ins are registered as thunks that import lazily, and installed
    packages contribute names through the entry-point ``group`` — both resolved only on demand, so
    nothing optional is imported until something is actually built.
    """

    def __init__(self, group: str, *, label: str) -> None:
        self._group = group  # entry-point group, e.g. "tensorsketch.providers"
        self._label = label  # for error messages, e.g. "provider"
        self._factories: dict[str, Factory[T]] = {}
        self._scanned = False

    def register(self, name: str, factory: type[T] | Factory[T]) -> None:
        """Register a name. `factory` is the class itself, or a zero-arg callable returning it
        (pass a callable to defer a heavy import). An explicit registration wins over an
        entry point of the same name."""
        self._factories[name] = _as_factory(factory)

    def get(self, name: str) -> type[T]:
        """Resolve a name to its class/callable, importing lazily. Raises `KeyError` if unknown."""
        factory = self._factories.get(name)
        if factory is None:
            self._scan()
            factory = self._factories.get(name)
        if factory is None:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self._label} {name!r}; registered: {known}")
        return factory()

    def create(self, name: str, *args: object, **kwargs: object) -> T:
        """Construct the named object, forwarding positional/keyword args to its constructor."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Every registered name — built-ins plus installed entry points — imported: none."""
        self._scan()
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def _scan(self) -> None:
        # Discover entry-point names once. `entry_points` only reads metadata; `ep.load()` (which
        # imports the target) is deferred to `get`. `setdefault` lets an explicit `register` or a
        # built-in take precedence over an installed name.
        if self._scanned:
            return
        self._scanned = True
        for ep in entry_points(group=self._group):
            self._factories.setdefault(ep.name, _ep_factory(ep))


def _as_factory(factory: type[T] | Factory[T]) -> Factory[T]:
    if isinstance(factory, type):
        cls = factory
        return lambda: cls
    return factory


def _ep_factory(ep: object) -> Factory[T]:
    # `ep.load()` imports the entry point's module and returns the referenced attribute.
    return lambda: ep.load()  # type: ignore[attr-defined]


def _lazy(module: str, attr: str) -> Factory[T]:
    def load() -> type[T]:
        obj: type[T] = getattr(import_module(module), attr)
        return obj

    return load


# --- The two registries ---------------------------------------------------------------------
#
# Built-ins are pre-registered as lazy thunks: naming them here imports nothing, and the actual
# provider/backend module (and its optional SDK/driver) loads only when you `create` it.

providers: Registry[ChatProvider] = Registry("tensorsketch.providers", label="provider")
providers.register("fake", _lazy("tensorsketch.providers.fake", "FakeProvider"))
providers.register("anthropic", _lazy("tensorsketch.providers.anthropic", "AnthropicProvider"))
providers.register("openai", _lazy("tensorsketch.providers.openai", "OpenAIProvider"))
providers.register("google", _lazy("tensorsketch.providers.google", "GoogleProvider"))

backends: Registry[Backend] = Registry("tensorsketch.backends", label="backend")
backends.register("memory", _lazy("tensorsketch.runtime.checkpoint", "InMemoryBackend"))
backends.register("sqlite", _lazy("tensorsketch.runtime.backends.sql", "SqliteBackend"))
backends.register("postgres", _lazy("tensorsketch.runtime.backends.postgres", "PostgresBackend"))
backends.register("redis", _lazy("tensorsketch.runtime.backends.redis", "RedisBackend"))


# --- Convenience functions (the common surface) ---------------------------------------------


def create_provider(name: str, *args: object, **kwargs: object) -> ChatProvider:
    """Construct a provider by name, e.g. ``create_provider("anthropic", model=...)``."""
    return providers.create(name, *args, **kwargs)


def register_provider(name: str, factory: type[ChatProvider] | Factory[ChatProvider]) -> None:
    """Add a provider name (a class, or a callable returning one)."""
    providers.register(name, factory)


def create_backend(name: str, *args: object, **kwargs: object) -> Backend:
    """Construct a backend by name, e.g. ``create_backend("sqlite", "runs.db")``."""
    return backends.create(name, *args, **kwargs)


def register_backend(name: str, factory: type[Backend] | Factory[Backend]) -> None:
    """Add a backend name (a class, or a callable returning one)."""
    backends.register(name, factory)


__all__ = [
    "Registry",
    "backends",
    "create_backend",
    "create_provider",
    "providers",
    "register_backend",
    "register_provider",
]
