"""Bandwidth-aware tile selection heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from . import utils


Objective = str


@dataclass
class SelectionResult:
    chosen: Dict[Tuple[int, int], np.ndarray]
    information: np.ndarray


def greedy_tile_selection(
    tiles: Mapping[Tuple[int, int], np.ndarray],
    *,
    per_agent_budget: Mapping[int, int] | None = None,
    global_budget: int | None = None,
    objective: Objective = "logdet",
    base_information: np.ndarray | None = None,
    ridge: float = 1e-9,
) -> SelectionResult:
    """Greedy tile selection following the objectives described in the paper."""

    if not tiles:
        raise ValueError("No tiles provided")
    if base_information is None:
        dim = next(iter(tiles.values())).shape[0]
        base_information = np.zeros((dim, dim))
    base_information = utils.symmetrize(base_information)

    remaining = set(tiles.keys())
    chosen: Dict[Tuple[int, int], np.ndarray] = {}
    per_agent_budget = dict(per_agent_budget or {})
    agent_counts = {agent: 0 for agent, _ in tiles.keys()}
    selected_info = base_information.copy()

    def objective_value(info: np.ndarray) -> float:
        sym = utils.add_ridge(info, ridge)
        if objective == "logdet":
            sign, logdet = np.linalg.slogdet(sym)
            return float(logdet if sign > 0 else -np.inf)
        if objective == "trace":
            return float(np.trace(sym))
        if objective == "lambda_min":
            eigs = np.linalg.eigvalsh(sym)
            return float(np.min(eigs))
        raise ValueError(f"Unknown objective '{objective}'")

    def can_select(agent: int) -> bool:
        if per_agent_budget and agent in per_agent_budget:
            return agent_counts[agent] < per_agent_budget[agent]
        return True

    def budget_remaining() -> bool:
        if global_budget is None:
            return True
        return len(chosen) < global_budget

    current_value = objective_value(selected_info)
    while remaining and budget_remaining():
        best_tile = None
        best_info = None
        best_value = current_value
        for tile in list(remaining):
            agent, tile_id = tile
            if not can_select(agent):
                continue
            candidate = utils.symmetrize(selected_info + tiles[tile])
            value = objective_value(candidate)
            if value > best_value + 1e-12:
                best_value = value
                best_tile = tile
                best_info = candidate
        if best_tile is None:
            break
        remaining.remove(best_tile)
        agent_counts[best_tile[0]] += 1
        chosen[best_tile] = tiles[best_tile]
        selected_info = best_info  # type: ignore[assignment]
        current_value = best_value
    return SelectionResult(chosen=chosen, information=selected_info)
