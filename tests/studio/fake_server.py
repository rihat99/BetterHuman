"""The studio tests' recording stand-in for ``viser.ViserServer``: no sockets, each call bound to viser's signature.

Every scene and GUI method the studio calls is bound with ``inspect.signature`` against the installed
``viser.SceneApi`` / ``viser.GuiApi`` method of the same name, so a method viser lacks, an unknown keyword or a
missing argument fails here as it would on viser. A handle reads back its call's arguments and records later writes.
Studio only, until ``better_robot.testing.FakeServer`` has the studio's GUI calls (``add_dropdown``, ``add_markdown``,
``add_text``, ``add_number``, ``add_tab_group``); then this file goes.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any

import viser


def _bound(api: type, method: str, args: tuple, kwargs: dict) -> dict[str, Any]:
    """The arguments of ``api.method(*args, **kwargs)`` by name, defaults applied; raises as viser would."""
    bound = inspect.signature(getattr(api, method)).bind(None, *args, **kwargs)
    bound.apply_defaults()
    return {name: value for name, value in bound.arguments.items() if name != "self"}


class FakeHandle:
    """Reads back its call's arguments, records later writes, fires its callbacks, and opens as a container."""

    def __init__(self, owner: Any, kind: str, name: str, props: dict[str, Any]) -> None:
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "props", props)
        object.__setattr__(self, "writes", [])
        object.__setattr__(self, "callbacks", [])
        object.__setattr__(self, "removed", False)
        object.__setattr__(self, "fired", False)
        object.__setattr__(self, "container", getattr(owner, "container", None))
        object.__setattr__(self, "visible", props.get("visible", True))
        # viser's dropdown starts at its first option when no initial value is given.
        initial = props.get("initial_value")
        object.__setattr__(self, "value", props["options"][0] if initial is None and "options" in props else initial)

    def __getattr__(self, key: str) -> Any:
        props = self.__dict__.get("props", {})
        if key in props:
            return props[key]
        raise AttributeError(f"{self.kind} handle has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        object.__setattr__(self, key, value)
        self.writes.append(key)
        if isinstance(self.owner, FakeScene):
            self.owner.log.append((self.name, key))

    @property
    def live(self) -> bool:
        """Not removed, nor inside a removed tab or folder."""
        return not self.removed and (self.container is None or self.container.live)

    def remove(self) -> None:
        object.__setattr__(self, "removed", True)
        if isinstance(self.owner, FakeScene):
            self.owner.log.append((self.name, "remove"))

    def on_update(self, callback: Any) -> Any:
        self.callbacks.append(callback)
        return callback

    on_click = on_update
    on_drag_end = on_update

    def fire(self) -> None:
        """Play the browser: call every registered callback."""
        object.__setattr__(self, "fired", True)
        for callback in self.callbacks:
            callback(None)

    def add_tab(self, label: str, icon: Any = None) -> FakeHandle:
        _bound(viser.GuiTabGroupHandle, "add_tab", (label,), {"icon": icon})
        return FakeHandle(self.owner, "tab", label, {})

    def __enter__(self) -> FakeHandle:
        self.owner.stack.append(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.owner.stack.pop()


class FakeScene:
    """Every ``add_*`` returns a ``FakeHandle``; ``log`` lists (node, event) in order."""

    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}
        self.log: list[tuple[str, str]] = []
        self.pointer_callbacks: dict[str, Any] = {}
        self.stack: list[FakeHandle] = []

    def __getattr__(self, method: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            props = _bound(viser.SceneApi, method, args, kwargs)
            event = {"on_click": "click", "on_rect_select": "rect-select"}.get(method)
            if event is not None:
                # viser's registration is a decorator factory: ``scene.on_click()(callback)``.
                def register(callback: Any) -> Any:
                    self.pointer_callbacks[event] = callback
                    return callback

                return register
            if method in ("remove_click_callback", "remove_rect_select_callback"):
                del self.pointer_callbacks["click" if method == "remove_click_callback" else "rect-select"]
                return None
            if not method.startswith("add_"):
                return None
            handle = FakeHandle(self, method[4:], props.pop("name"), props)
            self.nodes[handle.name] = handle
            self.log.append((handle.name, "add"))
            return handle

        if not callable(getattr(viser.SceneApi, method, None)):
            raise AttributeError(f"viser.SceneApi has no method {method!r}")
        return call


class FakeGui:
    """Every ``add_*`` returns a ``FakeHandle`` inside the open tab or folder; :meth:`control` finds one by label."""

    def __init__(self) -> None:
        self.controls: list[FakeHandle] = []
        self.stack: list[FakeHandle] = []

    @property
    def container(self) -> FakeHandle | None:
        return self.stack[-1] if self.stack else None

    def __getattr__(self, method: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> FakeHandle | None:
            props = _bound(viser.GuiApi, method, args, kwargs)
            if not method.startswith("add_"):
                return None
            handle = FakeHandle(self, method[4:], props.get("label", method[4:]), props)
            self.controls.append(handle)
            return handle

        if not callable(getattr(viser.GuiApi, method, None)):
            raise AttributeError(f"viser.GuiApi has no method {method!r}")
        return call

    def control(self, label: str) -> FakeHandle:
        """The one live control labelled ``label``."""
        found = [control for control in self.controls if control.name == label and control.live]
        if len(found) != 1:
            raise LookupError(f"{len(found)} live controls are labelled {label!r}")
        return found[0]


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
