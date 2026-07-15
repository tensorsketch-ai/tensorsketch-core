"""The name → object registry: build providers/backends by name, lazily, plus entry points."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tensorsketch import create_backend, create_provider, register_provider
from tensorsketch.providers.fake import FakeProvider
from tensorsketch.registry import Registry, backends, providers
from tensorsketch.runtime.backends.sql import SqliteBackend
from tensorsketch.runtime.checkpoint import InMemoryBackend


def test_create_builtin_provider() -> None:
    p = create_provider("fake", script=[])
    assert isinstance(p, FakeProvider)


def test_create_builtin_backend() -> None:
    assert isinstance(create_backend("memory"), InMemoryBackend)
    assert isinstance(create_backend("sqlite", ":memory:"), SqliteBackend)


def test_names_include_builtins() -> None:
    assert {"fake", "anthropic", "openai", "google"} <= set(providers.names())
    assert {"memory", "sqlite", "postgres", "redis"} <= set(backends.names())


def test_unknown_name_raises() -> None:
    with pytest.raises(KeyError) as excinfo:
        create_provider("nope")
    message = str(excinfo.value)
    assert "provider" in message and "nope" in message


def test_register_custom_class() -> None:
    register_provider("myfake", FakeProvider)
    assert isinstance(create_provider("myfake", script=[]), FakeProvider)


def test_register_accepts_a_deferred_thunk() -> None:
    # A callable returning the class defers the import until the name is built.
    providers.register("lazyfake", lambda: FakeProvider)
    assert isinstance(create_provider("lazyfake", script=[]), FakeProvider)


def test_entry_point_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    class Made:
        def __init__(self, x: int) -> None:
            self.x = x

    class FakeEP:
        name = "made"

        def load(self) -> type[Made]:
            return Made

    def fake_entry_points(group: str) -> list[FakeEP]:
        return [FakeEP()] if group == "tensorsketch.test.widgets" else []

    monkeypatch.setattr("tensorsketch.registry.entry_points", fake_entry_points)
    reg: Registry[Made] = Registry("tensorsketch.test.widgets", label="widget")

    # The installed name is discoverable without importing/constructing it.
    assert reg.names() == ["made"]
    built = reg.create("made", 7)
    assert built.x == 7


def test_explicit_registration_wins_over_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    class FromEntryPoint: ...

    class FromCode: ...

    class FakeEP:
        name = "dup"

        def load(self) -> type[FromEntryPoint]:
            return FromEntryPoint

    monkeypatch.setattr("tensorsketch.registry.entry_points", lambda group: [FakeEP()])
    reg: Registry[object] = Registry("tensorsketch.test.dup", label="thing")
    reg.register("dup", FromCode)
    assert reg.get("dup") is FromCode


def test_naming_imports_no_optional_sdk() -> None:
    # Run in a fresh interpreter so the assertion is independent of what other tests imported:
    # listing names and building an in-core backend must pull in no provider SDK or DB driver.
    code = textwrap.dedent(
        """
        import sys
        from tensorsketch import create_backend
        from tensorsketch.registry import backends, providers

        assert "anthropic" in providers.names()
        assert "postgres" in backends.names()
        assert type(create_backend("memory")).__name__ == "InMemoryBackend"

        leaked = [m for m in ("anthropic", "openai", "psycopg", "redis") if m in sys.modules]
        assert not leaked, leaked
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
