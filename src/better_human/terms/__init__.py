"""Human cost terms for ``better_robot.optim``: plain functions over a cached :class:`~better_human.HumanState`.

Register the state once per problem (``problem.cached(human_state(identity), name="state")``); every term below reads
it, so one forward pass serves all terms. Pose variables are ``q (B, nq)`` or trajectories ``q (B, T, nq)``;
residuals keep the leading batch axes and flatten their event axes. Identity groups may be variables too
(``problem.cached(baked_identity(human), name="identity")``) for shape fitting.
"""

from .fitting import ground, joints, landmarks, points, pose_prior, reprojection, shape_prior
from .physics import contact_wrenches, root_wrench
from .state import baked_identity, human_state

__all__ = [
    "baked_identity",
    "contact_wrenches",
    "ground",
    "human_state",
    "joints",
    "landmarks",
    "points",
    "pose_prior",
    "reprojection",
    "root_wrench",
    "shape_prior",
]
