"""A recording stand-in for ``viser.ViserServer``: no sockets, every call and property write logged."""

from __future__ import annotations

import contextlib
from typing import Any


class FakeHandle:
    """Records constructor kwargs and later attribute writes; supports GUI callbacks and folders."""

    def __init__(self, scene: FakeScene | None, kind: str, name: str, kwargs: dict[str, Any]) -> None:
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "props", dict(kwargs))
        object.__setattr__(self, "writes", [])
        object.__setattr__(self, "callbacks", [])
        object.__setattr__(self, "removed", False)
        object.__setattr__(self, "visible", kwargs.get("visible", True))
        object.__setattr__(self, "value", kwargs.get("initial_value"))
        object.__setattr__(self, "label", name)

    def __getattr__(self, key: str) -> Any:
        # Constructor kwargs (e.g. a transform control's ``position``/``wxyz``) read back like viser handles.
        props = self.__dict__.get("props", {})
        if key in props:
            return props[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)
        self.writes.append(key)
        if self.scene is not None:
            self.scene.log.append((self.name, key))

    def remove(self) -> None:
        object.__setattr__(self, "removed", True)
        if self.scene is not None:
            self.scene.log.append((self.name, "remove"))

    def on_update(self, callback: Any) -> None:
        self.callbacks.append(callback)

    on_click = on_update
    on_drag_end = on_update

    def fire(self) -> None:
        for callback in self.callbacks:
            callback(None)

    def __enter__(self) -> FakeHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class FakeScene:
    """Every ``add_*`` returns a ``FakeHandle``; ``log`` lists (node, event) in order."""

    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}
        self.log: list[tuple[str, str]] = []
        self.pointer_callbacks: dict[str, Any] = {}

    def on_pointer_event(self, event_type: str) -> Any:
        return lambda callback: self.pointer_callbacks.__setitem__(event_type, callback)

    def remove_pointer_callback(self) -> None:
        self.pointer_callbacks.clear()

    def __getattr__(self, method: str) -> Any:
        if not method.startswith(("add_", "set_", "configure_")):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> FakeHandle | None:
            if not method.startswith("add_"):
                return None
            name = args[0]
            handle = FakeHandle(self, method[4:], name, {"args": args[1:], **kwargs})
            self.nodes[name] = handle
            self.log.append((name, "add"))
            return handle

        return call


class _FakeTabs:
    def add_tab(self, _label: str) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext()

    def remove(self) -> None:
        self.removed = True


class FakeGui:
    def __init__(self) -> None:
        self.controls: list[FakeHandle] = []

    def add_tab_group(self) -> _FakeTabs:
        return _FakeTabs()

    def __getattr__(self, method: str) -> Any:
        if not method.startswith(("add_", "configure_")):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> FakeHandle | None:
            if not method.startswith("add_"):
                return None
            label = args[0] if args and isinstance(args[0], str) else method[4:]
            handle = FakeHandle(None, method[4:], label, {"args": args, **kwargs})
            self.controls.append(handle)
            return handle

        return call

    def control(self, label: str) -> FakeHandle:
        return next(control for control in self.controls if control.name == label)


class FakeServer:
    def __init__(self, port: int = 8080) -> None:
        self.scene = FakeScene()
        self.gui = FakeGui()
        self.atomic_blocks = 0
        self.port = port

    def get_port(self) -> int:
        return self.port

    @contextlib.contextmanager
    def atomic(self):
        self.atomic_blocks += 1
        yield
