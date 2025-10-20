# Multi-Agent Pose Uncertainty: A Differentiable Rendering Cramér–Rao Bound

This repo is for the "Multi-Agent Pose Uncertainty: A Differentiable Rendering Cramér–Rao Bound" paper, presented at ICCV 2025 Workshop on Large Scale Cross Device Localization.

The goal is to provide a clear, well-tested reference implementation of
the core mathematical primitives:

- Lie-group utilities on SE(3) (exponential/logarithm, adjoint, Jacobians)
- Implicit assembly of the pose Fisher information matrix
- Multi-agent adjoint transport and information fusion
- Bandwidth-aware tile selection heuristics for communication-limited
  settings

## Scope

The project intentionally **does not** ship a dataset pipeline, renderer
training scripts, or large-scale experiment orchestration.  Instead, the
library focuses on the abstractions required to integrate differentiable
renderers with pose-information analysis.  Users can plug in their own
renderers by implementing the minimal `Renderer` interface located in
`crb/rendering/renderer_api.py`. If you have questions or need assistance, please contact [arunadarsh.m@gmail.com](mailto:arunadarsh.m@gmail.com).

## Installation

The package targets Python 3.9+ and depends only on NumPy.  You can
install the project in editable mode while developing:

```bash
pip install -e .
```

## Running the tests

The included unit tests exercise the SE(3) helpers, FIM assembly, and
multi-agent utilities.  They run quickly on CPU-only environments:

```bash
pytest
```

## Repository layout

```
crb/                  Core Python library
  rendering/          Minimal renderer interface and toy implementation
  se3.py              Lie-group utilities for SE(3)
  fim.py              Fisher information assembly and CRB helpers
  jacobians.py        Jacobian manipulation utilities
  multiagent.py       Adjoint transport and fusion helpers
  selection.py        Bandwidth-aware tile selection
  utils.py            Numerics helpers

tests/                CPU-fast unit tests
  test_*.py

docs/                 Lightweight API and methodology notes
  api.md
  methodology.md
```

## Citing

If you find this library useful in your research, please cite the paper
using the following BibTeX entry:

```
@inproceedings{
muthukkumar2025multiagent,
  title={Multi-Agent Pose Uncertainty: A Differentiable Rendering {Cramér--Rao Bound}},
  author={Arun Muthukkumar},
  booktitle={ICCV 2025 Workshop on Large Scale Cross Device Localization},
  year={2025},
  url={https://openreview.net/forum?id=mXJ8waZdSv}
}
```
