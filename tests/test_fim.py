import numpy as np

from crb import fim, jacobians, se3
from crb.rendering.toy_renderer import ToyRenderer


def make_pose():
    return np.eye(4)


def test_fim_matches_analytic_jacobian():
    points = np.array([[0.0, 0.0, 1.0], [0.5, -0.2, 1.5]])
    renderer = ToyRenderer(points)
    pose = make_pose()
    basis = jacobians.canonical_se3_basis()
    columns = renderer.jacobian_columns(pose, basis=basis)
    analytic = renderer.analytic_jacobian(pose)
    np.testing.assert_allclose(columns, analytic, atol=1e-6)
    sigma = 0.01
    fim_numeric = fim.compute_fim(renderer, pose, noise=sigma)
    weighted = columns / sigma
    fim_expected = columns.T @ weighted
    np.testing.assert_allclose(fim_numeric, fim_expected)


def test_crb_inverse():
    points = np.array([[0.0, 0.0, 1.0], [0.2, 0.3, 1.2], [-0.4, 0.1, 0.9]])
    renderer = ToyRenderer(points)
    pose = make_pose()
    info = fim.compute_fim(renderer, pose, noise=0.1)
    cov = fim.crb_from_fim(info)
    recovered = cov @ info
    np.testing.assert_allclose(recovered, np.eye(6), atol=1e-6)
