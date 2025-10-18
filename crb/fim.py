"""Assembly of Fisher information matrices (FIM) and Cramér–Rao bounds."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import jacobians, utils


class InformationComputationError(RuntimeError):
    """Raised when a Fisher information matrix cannot be computed."""


def compute_fim(
    renderer,
    pose: np.ndarray,
    *,
    noise: utils.ArrayLike | None = None,
    basis: np.ndarray | None = None,
    pixels: Sequence[int] | None = None,
    ridge: float = 0.0,
) -> np.ndarray:
    """Compute the Fisher information matrix for ``pose``.

    Parameters
    ----------
    renderer:
        Object implementing ``jacobian_columns`` that returns the columns
        of the image Jacobian with respect to pose coordinates.
    pose:
        4x4 homogeneous transform describing the camera pose.
    noise:
        Optional description of the pixel-noise covariance ``Sigma``.
    basis:
        Columns describing the pose perturbation basis.  Defaults to the
        canonical se(3) basis.
    pixels:
        Optional sequence of pixel indices used when forming the FIM.
    ridge:
        Non-negative ridge value added to the resulting information
        matrix for numerical stability.
    """

    if basis is None:
        basis = jacobians.canonical_se3_basis()
    columns = renderer.jacobian_columns(pose, basis=basis, pixels=pixels)
    expected_cols = basis.shape[1] if basis.ndim == 2 else basis.size
    if columns.shape[1] != expected_cols:
        raise InformationComputationError("Renderer returned incompatible columns")
    fim = jacobians.assemble_information(columns, noise)
    if ridge:
        fim = utils.add_ridge(fim, ridge)
    return fim


def crb_from_fim(fim: np.ndarray, *, pseudo: bool = False) -> np.ndarray:
    """Return the Cramér–Rao bound (covariance) for ``fim``.

    Parameters
    ----------
    fim:
        Symmetric positive semi-definite Fisher information matrix.
    pseudo:
        When ``True`` the Moore–Penrose pseudoinverse is used.  By default
        a standard inverse is attempted and ``InformationComputationError``
        is raised if the matrix is singular.
    """

    fim = utils.symmetrize(fim)
    try:
        if pseudo:
            return np.linalg.pinv(fim)
        return np.linalg.inv(fim)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise InformationComputationError("Singular Fisher information matrix") from exc


def information_metrics(fim: np.ndarray) -> dict[str, float]:
    """Convenience metrics (trace/logdet/min-eig) for interpretation."""

    fim = utils.symmetrize(fim)
    eigs = np.linalg.eigvalsh(fim)
    slogdet = np.linalg.slogdet(fim)
    return {
        "trace": float(np.trace(fim)),
        "logdet": float(slogdet[0] * slogdet[1]),
        "min_eig": float(np.min(eigs)),
        "condition_number": float(np.max(eigs) / np.clip(np.min(eigs), 1e-12, None)),
    }
