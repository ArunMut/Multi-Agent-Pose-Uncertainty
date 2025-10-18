"""Helpers for manipulating renderer Jacobians and pixel subsets."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import utils


def canonical_se3_basis() -> np.ndarray:
    """Return the canonical basis of se(3) as column vectors."""

    return utils.canonical_basis(6)


def restrict_pixels(matrix: np.ndarray, pixels: Sequence[int] | None) -> np.ndarray:
    """Select a subset of rows (pixels) from ``matrix``.

    Parameters
    ----------
    matrix:
        The full matrix with shape (num_pixels, k).
    pixels:
        Optional ordered sequence of pixel indices.  When ``None`` the
        original matrix is returned.
    """

    if pixels is None:
        return matrix
    return matrix[np.asarray(list(pixels), dtype=int), :]


def assemble_information(columns: np.ndarray, noise: utils.ArrayLike | None = None) -> np.ndarray:
    """Assemble a Fisher information matrix from Jacobian columns.

    Parameters
    ----------
    columns:
        Matrix with shape (num_pixels, 6) whose columns correspond to the
        derivatives of image intensities with respect to the six pose
        coordinates.
    noise:
        Description of the observation covariance.  Delegates to
        :func:`crb.utils.apply_covariance_inverse`.
    """

    weighted = utils.apply_covariance_inverse(columns, noise)
    fim = columns.T @ weighted
    return utils.symmetrize(fim)
