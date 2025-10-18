import numpy as np

from crb import selection


def make_tile(scale):
    base = np.eye(6)
    return scale * base


def test_logdet_greedy_selection_respects_budget():
    tiles = {
        (0, 0): make_tile(1.0),
        (0, 1): make_tile(0.2),
        (1, 0): make_tile(0.5),
    }
    result = selection.greedy_tile_selection(tiles, global_budget=2, objective="logdet")
    assert len(result.chosen) == 2
    assert set(result.chosen.keys()).issubset(tiles.keys())


def test_lambda_min_improves_information():
    tiles = {
        (0, 0): make_tile(0.1),
        (0, 1): make_tile(0.3),
        (1, 0): make_tile(0.5),
    }
    result = selection.greedy_tile_selection(
        tiles,
        global_budget=2,
        per_agent_budget={0: 1, 1: 1},
        objective="lambda_min",
    )
    # Selected information should be positive semi-definite
    eigs = np.linalg.eigvalsh(result.information)
    assert np.all(eigs >= -1e-8)
