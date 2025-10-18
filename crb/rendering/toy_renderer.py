"""A tiny synthetic renderer for exercising the API in unit tests."""
from __future__ import annotations

import numpy as np

from .renderer_api import Renderer
from .. import se3


class ToyRenderer(Renderer):
    """Render 3D landmark projections as a flattened vector.

    Each landmark is projected onto the camera frame using a simple
    orthographic model.  The rendered vector stores the x/y image
    coordinates followed by the depth.  While highly simplified, the
    renderer captures the essential dependency of measurements on pose and
    provides non-trivial Jacobians for testing.
    """

    def __init__(self, points: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be of shape (N, 3)")
        self._points = np.asarray(points, dtype=float)

    def num_measurements(self) -> int:  # pragma: no cover - trivial
        return self._points.shape[0] * 3

    def render(self, pose: np.ndarray) -> np.ndarray:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_points = (rotation @ self._points.T).T + translation
        image = camera_points[:, :2].reshape(-1)
        depth = camera_points[:, 2]
        return np.concatenate([image, depth])

    def jacobian_columns(self, pose, *, basis, pixels=None, epsilon=1e-6):
        # Override to supply a slightly larger epsilon for numerical stability.
        return super().jacobian_columns(pose, basis=basis, pixels=pixels, epsilon=epsilon)

    def analytic_jacobian(self, pose: np.ndarray) -> np.ndarray:
        """Return the dense Jacobian (for verification purposes)."""

        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        num_points = self._points.shape[0]
        jac = np.zeros((self.num_measurements(), 6), dtype=float)
        for i, point in enumerate(self._points):
            p_cam = rotation @ point + translation
            J_rot = -se3.skew(p_cam)
            J_trans = np.eye(3)
            block = np.hstack([J_rot, J_trans])
            jac[2 * i, :] = block[0]
            jac[2 * i + 1, :] = block[1]
            jac[2 * num_points + i, :] = block[2]
        return jac
