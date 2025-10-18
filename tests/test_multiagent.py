import numpy as np

from crb import fim, multiagent
from crb.rendering.toy_renderer import ToyRenderer


def make_pose(tx=0.0, ty=0.0, tz=0.0):
    pose = np.eye(4)
    pose[:3, 3] = np.array([tx, ty, tz])
    return pose


def test_information_transport_and_fusion():
    points = np.array([[0.0, 0.0, 1.0], [0.5, 0.2, 1.2]])
    renderer = ToyRenderer(points)
    base_pose = make_pose()
    info_agent = fim.compute_fim(renderer, base_pose, noise=0.05)

    agent_pose = make_pose(0.1, 0.0, 0.0)
    world_pose = np.eye(4)
    fused = multiagent.fuse_agents([info_agent], [agent_pose], world_pose)

    transported = multiagent.transport_information(info_agent, np.linalg.inv(world_pose) @ agent_pose)
    np.testing.assert_allclose(fused, transported)


def test_fuse_information_sum():
    A = np.eye(6)
    B = 2 * np.eye(6)
    fused = multiagent.fuse_information([A, B])
    np.testing.assert_allclose(fused, 3 * np.eye(6))
