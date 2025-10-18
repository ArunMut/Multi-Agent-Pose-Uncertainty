"""Multi-agent information fusion utilities."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import se3, utils


def transport_information(information: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transport ``information`` into a common tangent space via the adjoint."""

    adj = se3.adjoint(transform)
    return adj.T @ information @ adj


def fuse_information(informations: Iterable[np.ndarray]) -> np.ndarray:
    """Sum a collection of information matrices."""

    informations = list(informations)
    if not informations:
        raise ValueError("At least one information matrix is required")
    result = np.zeros_like(informations[0])
    for info in informations:
        result = result + info
    return utils.symmetrize(result)


def fuse_agents(agent_infos: Sequence[np.ndarray], agent_poses: Sequence[np.ndarray], world_pose: np.ndarray) -> np.ndarray:
    """Fuse agent-specific informations into the tangent of ``world_pose``.

    Parameters
    ----------
    agent_infos:
        Iterable of 6x6 matrices expressed in each agent's tangent frame.
    agent_poses:
        Sequence of 4x4 transforms describing the pose of each agent with
        respect to the world frame.
    world_pose:
        4x4 pose describing the common tangent frame.
    """

    if len(agent_infos) != len(agent_poses):
        raise ValueError("agent_infos and agent_poses must have the same length")
    world_T = np.linalg.inv(world_pose)
    transported = []
    for info, pose in zip(agent_infos, agent_poses):
        relative = world_T @ pose
        transported.append(transport_information(info, relative))
    return fuse_information(transported)
