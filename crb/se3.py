"""Utilities for working with twists and transformations on SE(3).

The implementations in this module follow the micro-Lie algebra
conventions popularised by Sola ("Lie Theory for the Roboticist")
and Barfoot ("State Estimation for Robotics").  We adopt the
convention that pose perturbations are applied on the left, i.e.
``exp(xi) @ T`` where ``xi`` is a 6-vector twist and ``T`` is a
homogeneous 4x4 transform.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

_EPS = 1e-9


def skew(omega: Iterable[float]) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix for a rotation vector.

    Parameters
    ----------
    omega:
        Iterable of length 3 containing the components of the rotation
        vector.
    """

    wx, wy, wz = omega
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=float
    )


def hat(xi: Iterable[float]) -> np.ndarray:
    """Lift a 6-vector twist to the corresponding 4x4 matrix."""

    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,):
        raise ValueError("xi must be a 6-vector")
    omega = xi[:3]
    v = xi[3:]
    mat = np.zeros((4, 4), dtype=float)
    mat[:3, :3] = skew(omega)
    mat[:3, 3] = v
    return mat


def vee(mat: np.ndarray) -> np.ndarray:
    """Project a 4x4 matrix in se(3) to its 6-vector coordinates."""

    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError("mat must be 4x4")
    omega = np.array([mat[2, 1], mat[0, 2], mat[1, 0]], dtype=float)
    v = mat[:3, 3]
    return np.concatenate([omega, v])


def exp(xi: Iterable[float]) -> np.ndarray:
    """Exponential map from se(3) to SE(3).

    Uses the closed-form Rodrigues formula for SO(3) combined with the
    standard expression for the V matrix describing the translational
    component.
    """

    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,):
        raise ValueError("xi must be a 6-vector")
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    R = _exp_so3(omega)
    V = _left_jacobian_so3_matrix(omega)
    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log(T: np.ndarray) -> np.ndarray:
    """Logarithmic map from SE(3) to se(3)."""

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")
    R = T[:3, :3]
    t = T[:3, 3]
    omega = _log_so3(R)
    V = _left_jacobian_so3_matrix(omega)
    V_inv = np.linalg.inv(V)
    v = V_inv @ t
    return np.concatenate([omega, v])


def adjoint(T: np.ndarray) -> np.ndarray:
    """Compute the adjoint matrix of an SE(3) transformation."""

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")
    R = T[:3, :3]
    t = T[:3, 3]
    adj = np.zeros((6, 6), dtype=float)
    adj[:3, :3] = R
    adj[3:, 3:] = R
    adj[3:, :3] = skew(t) @ R
    return adj


def left_jacobian_SO3(omega: Iterable[float]) -> np.ndarray:
    """Left Jacobian of SO(3).

    This function returns the matrix ``J_l`` such that ``exp(omega)`` is
    locally approximated by ``I + J_l * omega``.  It is well-behaved even
    for small rotation magnitudes thanks to a series expansion.
    """

    omega = np.asarray(omega, dtype=float)
    return _left_jacobian_so3_matrix(omega)


def left_jacobian_SE3(xi: Iterable[float]) -> np.ndarray:
    """Left Jacobian of SE(3) for a 6-vector twist."""

    xi = np.asarray(xi, dtype=float)
    if xi.shape != (6,):
        raise ValueError("xi must be a 6-vector")
    omega = xi[:3]
    v = xi[3:]
    Jl_so3 = _left_jacobian_so3_matrix(omega)
    Jl = np.zeros((6, 6), dtype=float)
    Jl[:3, :3] = Jl_so3
    Jl[3:, 3:] = Jl_so3
    Q = _Q_matrix(omega, v)
    Jl[3:, :3] = Q
    return Jl


def perturb(pose: np.ndarray, xi: Iterable[float]) -> np.ndarray:
    """Apply a left perturbation ``exp(xi)`` to the pose ``pose``."""

    return exp(xi) @ pose


def _exp_so3(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    if theta < _EPS:
        K = skew(omega)
        return np.eye(3) + K + 0.5 * (K @ K)
    axis = omega / theta
    K = skew(axis)
    s = math.sin(theta)
    c = math.cos(theta)
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < _EPS:
        # Use first-order approximation of the logarithm.
        return np.array([R[2, 1], R[0, 2], R[1, 0]]) / 2.0
    omega_hat = (theta / (2.0 * math.sin(theta))) * (R - R.T)
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def _left_jacobian_so3_matrix(omega: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    K = skew(omega)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    s = math.sin(theta)
    c = math.cos(theta)
    theta2 = theta * theta
    return (
        np.eye(3)
        + (1 - c) / theta2 * K
        + (theta - s) / (theta2 * theta) * (K @ K)
    )


def _Q_matrix(omega: np.ndarray, v: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(omega)
    K = skew(omega)
    if theta < _EPS:
        return 0.5 * skew(v)
    s = math.sin(theta)
    c = math.cos(theta)
    theta2 = theta * theta
    theta3 = theta2 * theta
    return (
        0.5 * skew(v)
        + (theta - s) / (theta3) * (K @ skew(v) + skew(v) @ K - K @ skew(v) @ K)
        + (1 - 0.5 * theta2 - c) / (theta3 * theta) * (K @ K @ skew(v) + skew(v) @ K @ K)
    )
