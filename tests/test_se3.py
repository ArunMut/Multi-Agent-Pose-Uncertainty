import numpy as np

from crb import se3


def test_hat_vee_roundtrip():
    xi = np.array([0.1, -0.2, 0.3, 1.0, -0.5, 0.25])
    mat = se3.hat(xi)
    xi_back = se3.vee(mat)
    np.testing.assert_allclose(xi, xi_back)


def test_exp_log_roundtrip():
    xi = np.array([0.2, -0.1, 0.05, 0.5, -0.3, 0.7])
    T = se3.exp(xi)
    xi_back = se3.log(T)
    np.testing.assert_allclose(xi, xi_back, atol=1e-6)


def test_adjoint_properties():
    xi = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])
    T = se3.exp(xi)
    adj = se3.adjoint(T)
    perturb = np.array([-0.2, 0.3, 0.1, 0.5, 0.2, -0.4])
    lhs = se3.vee(np.linalg.inv(T) @ se3.hat(perturb) @ T)
    rhs = np.linalg.solve(adj, perturb)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)
