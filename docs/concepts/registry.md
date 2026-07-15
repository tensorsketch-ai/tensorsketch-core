# Choosing by name (registry)

Sometimes the model or the database shouldn't be hard-coded in an `import` — it should come from a
config file, an environment variable, or a `--flag`. TensorSketch's registry lets you build a built-in by
**name**:

```python
from tensorsketch import create_provider, create_backend

model = create_provider(cfg["provider"], model=cfg["model"])   # "anthropic" / "openai" / "google"
store = create_backend(cfg["backend"], dsn=cfg["dsn"])         # "sqlite" / "postgres" / "redis"
```

Swap the strings, swap the stack — nothing else in your graph changes.

## It's all one package

This is **not** a plugin ecosystem you assemble from pieces. Everything first-party ships in the
single `tensorsketch` package. The extras you already know (`tensorsketch-core[anthropic]`, `tensorsketch-core[postgres]`, …) only
gate a heavy *third-party SDK or driver*; they don't split TensorSketch into a core/community/provider
constellation you have to wire together. The registry just gives those built-ins a name.

Deliberately small: only the two seams where name-selection genuinely pays off have a registry —
**providers** and **backends**.

| Registry | Built-in names | Build with |
|---|---|---|
| providers | `fake`, `anthropic`, `openai`, `google` | `create_provider(name, **kwargs)` |
| backends | `memory`, `sqlite`, `postgres`, `redis` | `create_backend(name, **kwargs)` |

Constructor arguments pass straight through, so `create_provider("anthropic", model="…",
api_key="…")` is exactly `AnthropicProvider(model="…", api_key="…")`.

## Lazy by name — nothing imported until you build it

Listing or resolving a name imports **nothing** optional. `create_provider("anthropic", …)` is the
first moment the Anthropic SDK loads; until then it isn't touched. So `import tensorsketch` stays free of
every provider SDK and DB driver, and you can introspect what's available without paying for it:

```python
from tensorsketch.registry import providers, backends

providers.names()   # ['anthropic', 'fake', 'google', 'openai']  — no SDK imported
backends.names()    # ['memory', 'postgres', 'redis', 'sqlite']
```

## Add your own name

**In-process** — register a class (or a callable that returns one, to defer a heavy import):

```python
from tensorsketch import register_provider

register_provider("acme", AcmeProvider)
create_provider("acme", model="m1")
```

**From a package you publish** — declare an entry point. It's still one `pip install` for your
users and needs zero TensorSketch-side wiring; the name simply appears in the registry once installed:

```toml
# your package's pyproject.toml
[project.entry-points."tensorsketch.providers"]
acme = "loom_acme:AcmeProvider"

[project.entry-points."tensorsketch.backends"]
acme-db = "loom_acme:AcmeBackend"
```

```python
create_provider("acme", model="m1")     # resolves the installed entry point, lazily
```

An explicit `register_*` call takes precedence over an installed entry point of the same name, so
you can always override.

That's the whole extension story: named factories with lazy loading. No plugin objects, no
separate core/community packages. See [`examples/registry_by_name.py`](../../examples/registry_by_name.py).

## Relationship to the provider / backend guides

The registry is only the *selector*. Writing a provider is still the [`ChatProvider`
interface](providers.md); writing a backend is still the [`Backend` ABC](durability.md). The
registry just gives whatever you built a string name so config can choose it.
