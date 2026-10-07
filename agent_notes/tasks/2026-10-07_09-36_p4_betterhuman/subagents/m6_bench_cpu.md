# M6 benchmark, CPU (8 threads, float32)

Produced by `uv run python benchmarks/run.py --device cpu --threads 8 --batches 1 64 1024 8192 --out benchmarks/results/p4_bodies_cpu.json`, rendered with `benchmarks/report.py`. Ratio = ours / baseline median (below 1 means ours is faster). Baselines: upstream vendored smplx forward (betas+expression+pose, return_verts) and forward+backward (grad of vertex sum w.r.t. pose); MHR TorchScript `mhr_model.pt` forward (identity+pose+expression, correctives on). MHR "default" vertices pass the expression stream; SMPL-X "default" passes no stream. Cells above an estimated 8 GB are skipped and listed.

torch 2.13.0+cu129 | cpu (AMD EPYC 9654 96-Core Processor) | threads 8 | commit b08b9732d5 | float32

| family | op | batch | variant | median ms | p90 ms | baseline ms | ours / baseline |
|---|---|---:|---|---:|---:|---:|---:|
| SMPL | bake | 1 | mass | 11.217 | 11.253 |  |  |
| SMPL | bake | 1 | no_mass | 0.409 | 0.418 |  |  |
| SMPL | vertices | 1 | upstream | 1.747 | 1.799 |  |  |
| SMPL | vertices_backward | 1 | upstream | 4.435 | 4.456 |  |  |
| SMPL | vertices | 1 | default | 1.212 | 1.230 | 1.747 | 0.69 |
| SMPL | vertices_backward | 1 | default | 3.251 | 3.265 | 4.435 | 0.73 |
| SMPL | fk | 1 | default | 0.261 | 0.270 |  |  |
| SMPL | rnea | 1 | default | 0.307 | 0.317 |  |  |
| SMPL | vertices | 64 | upstream | 9.879 | 10.003 |  |  |
| SMPL | vertices_backward | 64 | upstream | 40.430 | 46.170 |  |  |
| SMPL | vertices | 64 | default | 14.749 | 15.048 | 9.879 | 1.49 |
| SMPL | vertices_backward | 64 | default | 41.319 | 41.712 | 40.430 | 1.02 |
| SMPL | fk | 64 | default | 0.845 | 0.859 |  |  |
| SMPL | rnea | 64 | default | 1.401 | 1.414 |  |  |
| SMPL | vertices | 1024 | upstream | 310.521 | 330.482 |  |  |
| SMPL | vertices_backward | 1024 | upstream | 632.313 | 642.061 |  |  |
| SMPL | vertices | 1024 | default | 296.465 | 297.442 | 310.521 | 0.95 |
| SMPL | vertices_backward | 1024 | default | 657.843 | 661.858 | 632.313 | 1.04 |
| SMPL | fk | 1024 | default | 6.873 | 6.932 |  |  |
| SMPL | rnea | 1024 | default | 15.841 | 16.083 |  |  |
| SMPL | vertices | 8192 | upstream | skipped: estimated 10.8 GB > 8 GB |  |  |  |
| SMPL | vertices | 8192 | default | skipped: estimated 10.8 GB > 8 GB |  |  |  |
| SMPL | vertices_backward | 8192 | default | skipped: estimated 14.4 GB > 8 GB |  |  |  |
| SMPL | fk | 8192 | default | 50.568 | 51.012 |  |  |
| SMPL | rnea | 8192 | default | 81.943 | 86.395 |  |  |
| SMPL-X | bake | 1 | mass | 23.453 | 23.622 |  |  |
| SMPL-X | bake | 1 | no_mass | 1.221 | 1.230 |  |  |
| SMPL-X | vertices | 1 | upstream | 7.129 | 7.186 |  |  |
| SMPL-X | vertices_backward | 1 | upstream | 14.729 | 14.861 |  |  |
| SMPL-X | vertices | 1 | default | 3.024 | 3.046 | 7.129 | 0.42 |
| SMPL-X | vertices_backward | 1 | default | 7.180 | 7.204 | 14.729 | 0.49 |
| SMPL-X | vertices | 1 | expression_stream_joints | 3.543 | 3.555 | 7.129 | 0.50 |
| SMPL-X | vertices_backward | 1 | expression_stream_joints | 7.854 | 7.902 | 14.729 | 0.53 |
| SMPL-X | vertices | 1 | expression_frozen_joints | 3.198 | 3.220 | 7.129 | 0.45 |
| SMPL-X | vertices_backward | 1 | expression_frozen_joints | 7.367 | 7.527 | 14.729 | 0.50 |
| SMPL-X | fk | 1 | default | 0.350 | 0.362 |  |  |
| SMPL-X | rnea | 1 | default | 0.372 | 0.390 |  |  |
| SMPL-X | vertices | 64 | upstream | 29.727 | 30.235 |  |  |
| SMPL-X | vertices_backward | 64 | upstream | 69.727 | 71.223 |  |  |
| SMPL-X | vertices | 64 | default | 37.241 | 46.991 | 29.727 | 1.25 |
| SMPL-X | vertices_backward | 64 | default | 75.996 | 83.433 | 69.727 | 1.09 |
| SMPL-X | vertices | 64 | expression_stream_joints | 45.249 | 46.668 | 29.727 | 1.52 |
| SMPL-X | vertices_backward | 64 | expression_stream_joints | 87.547 | 92.495 | 69.727 | 1.26 |
| SMPL-X | vertices | 64 | expression_frozen_joints | 24.346 | 26.336 | 29.727 | 0.82 |
| SMPL-X | vertices_backward | 64 | expression_frozen_joints | 83.582 | 92.287 | 69.727 | 1.20 |
| SMPL-X | fk | 64 | default | 1.802 | 1.815 |  |  |
| SMPL-X | rnea | 64 | default | 3.980 | 4.189 |  |  |
| SMPL-X | vertices | 1024 | upstream | 627.354 | 636.228 |  |  |
| SMPL-X | vertices_backward | 1024 | upstream | 1163.498 | 1230.908 |  |  |
| SMPL-X | vertices | 1024 | default | 463.783 | 468.301 | 627.354 | 0.74 |
| SMPL-X | vertices_backward | 1024 | default | 932.695 | 1008.091 | 1163.498 | 0.80 |
| SMPL-X | vertices | 1024 | expression_stream_joints | 499.906 | 543.311 | 627.354 | 0.80 |
| SMPL-X | vertices_backward | 1024 | expression_stream_joints | 995.843 | 1132.946 | 1163.498 | 0.86 |
| SMPL-X | vertices | 1024 | expression_frozen_joints | 459.951 | 489.680 | 627.354 | 0.73 |
| SMPL-X | vertices_backward | 1024 | expression_frozen_joints | 1096.523 | 1175.151 | 1163.498 | 0.94 |
| SMPL-X | fk | 1024 | default | 11.985 | 12.417 |  |  |
| SMPL-X | rnea | 1024 | default | 25.555 | 26.519 |  |  |
| SMPL-X | vertices | 8192 | upstream | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | default | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | default | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | expression_stream_joints | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | expression_stream_joints | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | expression_frozen_joints | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | expression_frozen_joints | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | fk | 8192 | default | 119.747 | 126.097 |  |  |
| SMPL-X | rnea | 8192 | default | 168.978 | 171.729 |  |  |
| MHR | bake | 1 | mass | 20.250 | 20.689 |  |  |
| MHR | bake | 1 | no_mass | 6.195 | 6.641 |  |  |
| MHR | vertices | 1 | torchscript | 37.753 | 39.328 |  |  |
| MHR | vertices | 1 | default | 22.437 | 23.672 | 37.753 | 0.59 |
| MHR | vertices_backward | 1 | default | 48.307 | 52.101 |  |  |
| MHR | fk | 1 | default | 0.941 | 0.955 |  |  |
| MHR | rnea | 1 | default | 0.548 | 0.560 |  |  |
| MHR | vertices | 64 | torchscript | 278.090 | 283.614 |  |  |
| MHR | vertices | 64 | default | 119.390 | 133.155 | 278.090 | 0.43 |
| MHR | vertices_backward | 64 | default | 241.879 | 248.304 |  |  |
| MHR | fk | 64 | default | 4.996 | 5.460 |  |  |
| MHR | rnea | 64 | default | 10.241 | 10.988 |  |  |
| MHR | vertices | 1024 | torchscript | 5218.719 | 5336.000 |  |  |
| MHR | vertices | 1024 | default | 1310.056 | 1404.487 | 5218.719 | 0.25 |
| MHR | vertices_backward | 1024 | default | 2847.675 | 2862.508 |  |  |
| MHR | fk | 1024 | default | 41.152 | 41.404 |  |  |
| MHR | rnea | 1024 | default | 45.219 | 45.453 |  |  |
| MHR | vertices | 8192 | torchscript | skipped: estimated 29.0 GB > 8 GB |  |  |  |
| MHR | vertices | 8192 | default | skipped: estimated 29.0 GB > 8 GB |  |  |  |
| MHR | vertices_backward | 8192 | default | skipped: estimated 38.7 GB > 8 GB |  |  |  |
| MHR | fk | 8192 | default | 463.301 | 469.719 |  |  |
| MHR | rnea | 8192 | default | 389.351 | 412.786 |  |  |
