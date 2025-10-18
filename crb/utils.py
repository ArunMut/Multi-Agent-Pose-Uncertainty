"""Assorted numerical helpers used across the CRB library."""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

ArrayLike = np.ndarray | Sequence[float] | float


def canonical_basis(dim: int) -> np.ndarray:
    """Return the canonical basis vectors for ``R^dim`` as columns."""

    return np.eye(dim)


def ensure_matrix(x: ArrayLike, *, shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Convert ``x`` to a 2-D floating point numpy array."""

    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = np.diag(arr)
    if arr.ndim != 2:
        raise ValueError("Expected an array broadcastable to 2-D")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Matrix has shape {arr.shape}, expected {shape}")
    return arr


def apply_covariance_inverse(matrix: np.ndarray, noise: ArrayLike | Callable[[np.ndarray], np.ndarray] | None) -> np.ndarray:
    """Apply the inverse covariance (``Sigma^{-1}``) to ``matrix``.

    Parameters
    ----------
    matrix:
        Array with shape (n, k) corresponding to k columns of a Jacobian.
    noise:
        Either ``None`` (identity covariance), a scalar variance, a vector
        of per-pixel variances, a covariance matrix, or a callable that
        applies ``Sigma^{-1}`` to a matrix.  Scalars/vectors describe the
        covariance ``Sigma``; the routine internally inverts them.
    """

    if noise is None:
        return matrix
    if callable(noise):
        return noise(matrix)
    arr = np.asarray(noise, dtype=float)
    if arr.ndim == 0:
        if arr <= 0:
            raise ValueError("Variance must be positive")
        return matrix / arr
    if arr.ndim == 1:
        if np.any(arr <= 0):
            raise ValueError("Variance entries must be positive")
        return matrix / arr[:, None]
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Covariance matrix must be square")
        return np.linalg.solve(arr, matrix)
    raise ValueError("Unsupported noise description")


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetrised version ``0.5 * (M + M^T)``."""

    return 0.5 * (matrix + matrix.T)


def is_positive_semidefinite(matrix: np.ndarray, *, atol: float = 1e-10) -> bool:
    """Check positive semi-definiteness using an eigenvalue test."""

    vals = np.linalg.eigvalsh(symmetrize(matrix))
    return np.all(vals >= -atol)


def add_ridge(matrix: np.ndarray, ridge: float) -> np.ndarray:
    """Return ``matrix + ridge * I`` without modifying the input."""

    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if ridge == 0:
        return matrix
    return matrix + ridge * np.eye(matrix.shape[0])
