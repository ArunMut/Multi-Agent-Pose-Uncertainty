# API Reference

## `crb.compute_fim(renderer, pose, **kwargs)`

Assemble the Fisher information matrix for a camera `pose` using the
provided `renderer`.  The renderer must expose a
`jacobian_columns(pose, basis, pixels=None)` method returning the Jacobian
columns with respect to pose perturbations.  Optional keyword arguments:

- `noise`: scalar/diagonal/full covariance or callable applying
  `Sigma^{-1}`
- `basis`: 6xK matrix describing the perturbation basis (defaults to the
  canonical se(3) basis)
- `pixels`: optional iterable of pixel indices to subsample
- `ridge`: numerical ridge added to the information matrix

The result is a 6x6 symmetric matrix.

## `crb.crb_from_fim(fim, pseudo=False)`

Invert a Fisher information matrix to obtain a Cramér–Rao covariance.  Set
`pseudo=True` to use the Moore–Penrose pseudoinverse when the information
is singular.

## `crb.multiagent`

- `transport_information(info, transform)`: apply SE(3) adjoint transport
  to express an information matrix in a different tangent frame.
- `fuse_information(infos)`: sum multiple information matrices.
- `fuse_agents(agent_infos, agent_poses, world_pose)`: transport each
  agent's information into the world frame and fuse them.

## `crb.selection`

`greedy_tile_selection(tiles, per_agent_budget=None, global_budget=None,
objective="logdet", base_information=None, ridge=1e-9)` implements the
bandwidth-aware tile selection heuristic discussed in the paper.  Tiles
are passed as a mapping from `(agent_id, tile_id)` to 6x6 information
contributions.  The function returns a `SelectionResult` with the chosen
subset and the final fused information matrix.

## Renderer interface

Custom renderers should subclass `crb.rendering.renderer_api.Renderer` and
implement the `render(pose)` and `num_measurements()` methods.  The base
class provides a finite-difference implementation of
`jacobian_columns`, which can be overridden for performance.
