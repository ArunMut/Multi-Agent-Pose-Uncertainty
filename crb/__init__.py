"""Render-aware Cramér–Rao bound utilities."""
from . import fim, jacobians, multiagent, se3, selection, utils
from .fim import compute_fim, crb_from_fim, information_metrics
from .jacobians import canonical_se3_basis
from .multiagent import fuse_agents, fuse_information, transport_information
from .selection import greedy_tile_selection, SelectionResult

__all__ = [
    "compute_fim",
    "crb_from_fim",
    "information_metrics",
    "canonical_se3_basis",
    "fuse_agents",
    "fuse_information",
    "transport_information",
    "greedy_tile_selection",
    "SelectionResult",
    "fim",
    "jacobians",
    "multiagent",
    "se3",
    "selection",
    "utils",
]
