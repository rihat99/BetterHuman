"""Every public type has a short repr (principle 4.9, decision record 0001): one line, under 120 characters, its name,
and its own (neither the default ``<module.Class object at 0x...>`` nor a dataclass repr that prints its tensors).

``BUILDERS`` builds each public class of the ``__all__`` tree (``better_robot.testing.public_objects``, each class
once, under the first path that lists it) from the random-weight SMPL-X archive (the studio from the SMPL one, on
BetterRobot's stand-in server) or from tensors large enough that a repr printing them spans lines. ``NO_INSTANCE`` holds the reasoned exceptions. The reprs of ``HumanModel``,
``Identity`` and ``HumanState`` name the family, and the model's also its sizes, dtype and device.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import shutil
import tempfile
from collections.abc import Callable
from functools import cache
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from better_robot.testing import FakeServer, public_objects

import better_human as bh
from better_human import core, families, viz
from better_human.studio import HumanStudio
from tests.support.random_archives import ARCHIVES, random_weight_smplx
from tests.support.synthetic import flame_spec

EXCEPTION = "an exception's repr is its message, which BetterRobot's error contract shapes"
ABSTRACT = "an abstract base: every concrete subclass is built"
NO_INSTANCE: dict[str, str] = {
    "better_human.io.AssetNotFoundError": EXCEPTION,
    "better_human.io.FormatError": EXCEPTION,
    "better_human.core.ClassicFormat": ABSTRACT,
    "better_human.core.Corrective": ABSTRACT,
    "better_human.core.Rig": ABSTRACT,
}


@cache
def _identity() -> bh.Identity:
    return random_weight_smplx().bake(betas=torch.zeros(2, random_weight_smplx().spec.identity_groups["betas"]))


def _state() -> bh.HumanState:
    return _identity().forward(random_weight_smplx().model.q_neutral.expand(3, 2, -1))


def _filled(cls: type, shape: tuple[int, ...], **given: object) -> object:
    """``cls`` with every field not in ``given`` a zero tensor of ``shape``."""
    zeros = {item.name: torch.zeros(shape) for item in dataclasses.fields(cls) if item.name not in given}
    return cls(**zeros, **given)


def _studio() -> HumanStudio:
    """The studio on the random SMPL archive, placed where ``families.smpl`` looks, on BetterRobot's stand-in server."""
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "smpl").mkdir()
        shutil.copy(ARCHIVES / "smpl_random.npz", Path(folder) / "smpl" / "SMPL_NEUTRAL.npz")
        with mock.patch.dict(os.environ, {"BETTERHUMAN_MODELS_DIR": folder}):
            return HumanStudio(server=FakeServer(), data_root=Path(folder) / "data", generator=np.random.default_rng(0))


BUILDERS: dict[str, Callable[[], object]] = {
    "better_human.core.ContactFrames": lambda: random_weight_smplx().contact_frames,
    "better_human.core.FamilySpec": lambda: random_weight_smplx().spec,
    "better_human.core.HumanModel": random_weight_smplx,
    "better_human.core.HumanState": _state,
    "better_human.core.Identity": _identity,
    "better_human.core.Landmarks": lambda: random_weight_smplx().spec.landmarks,
    "better_human.core.LinearCorrective": lambda: random_weight_smplx().spec.corrective,
    "better_human.core.MLPCorrective": lambda: _filled(core.MLPCorrective, (64, 300)),
    "better_human.core.Parts": lambda: random_weight_smplx().spec.parts,
    "better_human.core.RigBake": lambda: _filled(core.RigBake, (2, 300, 3)),
    "better_human.families.FLAMEClassic": lambda: _filled(families.FLAMEClassic, (3, 100)),
    "better_human.families.MANOClassic": lambda: _filled(families.MANOClassic, (3, 100)),
    "better_human.families.MHRClassic": lambda: _filled(families.MHRClassic, (3, 100)),
    "better_human.families.MHRFormat": families.MHRFormat,
    "better_human.families.MHRRig": lambda: _filled(families.MHRRig, (127, 3), model=random_weight_smplx().model),
    "better_human.families.RegressedRig": lambda: random_weight_smplx().spec.rig,
    "better_human.families.SMPLClassic": lambda: _filled(families.SMPLClassic, (3, 100)),
    "better_human.families.SMPLFamilyFormat": lambda: random_weight_smplx().spec.classic,
    "better_human.families.SMPLHClassic": lambda: _filled(families.SMPLHClassic, (3, 100)),
    "better_human.families.SMPLXClassic": lambda: bh.to_classic(_identity(), random_weight_smplx().model.q_neutral),
    "better_human.families.SOMAClassic": lambda: _filled(families.SOMAClassic, (3, 100)),
    "better_human.families.SOMAFormat": families.SOMAFormat,
    "better_human.families.SOMARig": lambda: _filled(families.SOMARig, (110, 78)),
    "better_human.mass.PartTopology": lambda: random_weight_smplx().part_topology,
    "better_human.studio.HumanStudio": _studio,
    "better_human.viz.HumanView": lambda: viz.HumanView(
        random_weight_smplx().bake().forward(random_weight_smplx().model.q_neutral)
    ),
}


def _public_classes() -> dict[str, type]:
    """Each public class once, under the first path that lists it."""
    found: dict[int, tuple[str, type]] = {}
    for path, value in public_objects("better_human").items():
        if inspect.isclass(value):
            found.setdefault(id(value), (path, value))
    return dict(found.values())


def test_every_public_class_is_built_or_has_a_reason() -> None:
    """A new public class needs a builder here (or a reason in ``NO_INSTANCE``)."""
    classes = _public_classes()
    assert not BUILDERS.keys() & NO_INSTANCE.keys()
    assert sorted(classes.keys() - BUILDERS.keys() - NO_INSTANCE.keys()) == []
    assert sorted((BUILDERS.keys() | NO_INSTANCE.keys()) - classes.keys()) == []


@pytest.mark.parametrize("path", sorted(BUILDERS))
def test_repr_is_one_short_line_naming_the_type(path: str) -> None:
    value = BUILDERS[path]()
    name = path.rsplit(".", 1)[-1]
    assert type(value).__name__ == name
    text = repr(value)
    assert "\n" not in text and len(text) < 120 and text.startswith(f"{name}("), f"{path}: {text[:200]!r}"
    assert "object at 0x" not in text and "tensor(" not in text, f"{path} prints its default repr: {text[:200]!r}"


@pytest.mark.slow
def test_record_0001_reprs_name_the_family(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    identity = human.bake(betas=torch.zeros(2, 100, dtype=torch.float64))
    state = identity.forward(identity.model.q_neutral.expand(3, 2, identity.model.nq))
    assert repr(human) == "HumanModel(flame, vertices=5023, joints=5, dtype=torch.float64, device=cpu)"
    assert repr(identity) == "Identity(flame, batch=(2,))"
    assert repr(state) == "HumanState(flame, batch=(3, 2))"
