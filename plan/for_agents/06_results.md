# Order 06 results

Completed on `dev` on 2026-07-20 after Order 05.

## Delivered

- `better_human.viz.SkinnedMeshMode`, following BetterRobot's render-mode
  lifecycle. Attach uploads one shaped rest surface; update consumes existing
  BetterRobot `Data`, runs the body's pose-corrective/LBS path, and replaces
  the same backend node. Arbitrary leading body batches use the render
  context's flattened `batch_index`.
- Recording-backend tests cover the protocol, one attach upload, posed update,
  batch selection, visibility/detach, robot/body mismatch, and unchanged
  tensor dtype/device at the renderer boundary. The core contract test uses a
  synthetic triangle and therefore does not depend on licensed weights.
- Three executable demos with deterministic `--headless --frames N` guards:
  SMPL beta/joint sliders using `Model.integrate`, full versus body-only
  SMPL-X, and `solve_ik` driving the automatically generated
  `body_left_wrist` frame to a draggable target.
- A `viewer` project extra delegated to `better-robot[viewer]`, with its
  resolved dependencies in `uv.lock`.
- A Get Started README covering the alpha sibling-checkout requirement,
  official licensed-weight sources, canonical models layout and environment
  variable, legacy converter, minimal shape/pose flow, supported spawns,
  and demos/docs entry points.

## Viewer path and measurements

- The resolved and tested integration uses Viser `1.0.30`.
- Inspection showed that Viser's `MeshHandle` inherits assignable
  `MeshProps.vertices`; assigning a same-shaped NumPy array performs an
  in-place property update and queues only that property. BetterRobot's public
  `RendererBackend`, however, exposes `add_mesh` but no mesh-vertex update.
- BetterHuman therefore uses the documented backend-neutral fallback:
  `add_mesh` again with the same node name. It does not import Viser or reach
  into `ViserBackend._nodes`.
- On this host, 200 CPU SMPL updates through the recording backend took
  `2.066 s` (`96.8 FPS`). The actual Viser backend, with a local server but no
  browser client attached, took `3.351 s` (`59.7 FPS`). These are raw update
  loop rates, not browser rasterization measurements.
- Full/body-only SMPL-X rendered one headless comparison frame in `0.026 s`
  after loading; the spawns had 55 and 22 native joints respectively.
- The IK demo completed one target solve/render loop in `0.988 s`, with a
  measured wrist position error of `0.000000 m` for the scripted reachable
  target.

## Verification

- `uv lock --check`: 87 root-project packages, current.
- `uv run pytest tests -q -rs`: **102 passed, 6 skipped, 7 warnings in
  81.43 s**. Skips are four sandbox CUDA tests and two explicit absent
  on-demand SMPL-X golden cases. Warnings are the four required float32
  gradcheck notices, PyTorch's CSR-beta notice, and two previously recorded
  SciPy pickle deprecations.
- Focused skinned-mesh tests: **4 passed, 1 sandbox CUDA skip**; the same CUDA
  case passed separately on GPU 3 in **2.22 s**.
- `uv run ruff check src tests examples`: passed.
- Each example launched against local weights, rendered one headless frame,
  and exited successfully. The SMPL demo also launched a real local Viser
  server, rendered 200 updates, and stopped with exit code zero. Per the plan,
  browser-side visual judgment remains for the owner; no screenshot assertion
  was introduced.
- The combined `models/` and `references/` path/size/mtime manifest remained
  `1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`.
- BetterRobot remained clean on `dev`; no dependency-side files were changed.

## Deviations and findings

- **Efficient-update fallback:** Viser offers the requested efficient vertex
  setter, but BetterRobot does not expose it through `RendererBackend`.
  BetterHuman uses the plan's same-name `add_mesh` fallback rather than a
  private backend reach-through. A future general BetterRobot
  `update_mesh_vertices` protocol addition could remove the redundant face
  upload.
- **Mode ownership:** the mode landed in BetterHuman, not BetterRobot. Its
  vertex computation requires `ParametricBody.vertices_from_data`; a generic
  callback-driven deforming-mesh abstraction would have only one in-tree
  caller today and would violate the project's two-caller abstraction rule.
- **Visual verification boundary:** no browser client or owner visual review
  was available to the agent. Server startup, updates, shutdown, shapes, and
  IK error were verified mechanically; subjective appearance is deliberately
  not claimed.
