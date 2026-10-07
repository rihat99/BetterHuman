"""H-API-2 / BR C-API-2: public callables take at most two positional parameters.

Every function in a public ``__all__`` and every public method of a public class is inspected; parameters after
the second (``self`` excluded) must be keyword-only. ``KEYWORD_ONLY_ALLOWLIST`` names the current violations with
their file:line; it only shrinks.
"""

from __future__ import annotations

import importlib
import inspect

PACKAGES = [
    "better_human",
    "better_human.core",
    "better_human.families",
    "better_human.io",
    "better_human.terms",
    "better_human.viz",
    "better_human.viz.studio",
]
MAX_POSITIONAL = 2

# Current violations (qualified name -> positional count). Shrink, never grow.
KEYWORD_ONLY_ALLOWLIST: dict[str, int] = {
    "better_human.core.rig.Rig.bake": 4,  # core/rig.py:47
    "better_human.core.rig.Rig.placements_from_rest": 3,  # core/rig.py:52
    "better_human.core.rig.Rig.corrective_features": 3,  # core/rig.py:61
    "better_human.core.rig.RegressedRig.bake": 4,  # core/rig.py:99
    "better_human.core.rig.RegressedRig.placements_from_rest": 3,  # core/rig.py:108
    "better_human.core.rig.RegressedRig.corrective_features": 3,  # core/rig.py:123
    "better_human.core.contact_frames.resolve_parents": 3,  # core/contact_frames.py:68
    "better_human.core.contact_frames.frame_local_placements": 5,  # core/contact_frames.py:128
    "better_human.io.load.resolve_model_path": 3,  # io/load.py:29
    "better_human.terms.contact_wrenches": 4,  # terms.py:182
    "better_human.viz.human_layers.forces_layer": 3,  # viz/human_layers.py:82
}


def _positional_count(fn: object, *, bound: bool) -> int:
    params = list(inspect.signature(fn).parameters.values())
    if bound and params:
        params = params[1:]
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(param.kind in kinds for param in params)


def _public_callables() -> dict[str, int]:
    counts: dict[str, int] = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                for attr, member in vars(obj).items():
                    if attr.startswith("_") or not inspect.isfunction(member):
                        continue
                    counts[f"{obj.__module__}.{obj.__qualname__}.{attr}"] = _positional_count(member, bound=True)
            elif inspect.isfunction(obj):
                counts[f"{obj.__module__}.{obj.__qualname__}"] = _positional_count(obj, bound=False)
    return counts


def test_public_callables_are_keyword_only_after_second_argument() -> None:
    violations = {name: count for name, count in _public_callables().items() if count > MAX_POSITIONAL}
    new = {name: count for name, count in violations.items() if KEYWORD_ONLY_ALLOWLIST.get(name) != count}
    assert not new, f"positional parameters beyond {MAX_POSITIONAL} (make them keyword-only): {new}"


def test_allowlist_has_no_stale_entries() -> None:
    counts = _public_callables()
    stale = [name for name, count in KEYWORD_ONLY_ALLOWLIST.items() if counts.get(name) != count]
    assert not stale, f"fixed or renamed; remove from KEYWORD_ONLY_ALLOWLIST: {stale}"
