# M6 benchmark, CUDA (GPU 0, float32)

Produced by `CUDA_VISIBLE_DEVICES=0 uv run python benchmarks/run.py --device cuda --batches 1 64 1024 8192 --out benchmarks/results/p4_bodies_cuda.json` (GPU 0 reported FREE by gpu_free.py), rendered with `benchmarks/report.py`. Same baselines and conventions as m6_bench_cpu.md; the 8 GB skip estimate is the CPU-calibrated one.

torch 2.13.0+cu129 | cuda (NVIDIA RTX PRO 6000 Blackwell Server Edition) | threads 192 | commit b08b9732d5 | float32

| family | op | batch | variant | median ms | p90 ms | baseline ms | ours / baseline |
|---|---|---:|---|---:|---:|---:|---:|
| SMPL | bake | 1 | mass | 20.751 | 21.710 |  |  |
| SMPL | bake | 1 | no_mass | 0.423 | 0.438 |  |  |
| SMPL | vertices | 1 | upstream | 1.481 | 1.513 |  |  |
| SMPL | vertices_backward | 1 | upstream | 7.783 | 7.857 |  |  |
| SMPL | vertices | 1 | default | 1.188 | 1.206 | 1.481 | 0.80 |
| SMPL | vertices_backward | 1 | default | 6.320 | 6.451 | 7.783 | 0.81 |
| SMPL | fk | 1 | default | 0.604 | 0.621 |  |  |
| SMPL | rnea | 1 | default | 0.728 | 0.743 |  |  |
| SMPL | vertices | 64 | upstream | 1.747 | 1.767 |  |  |
| SMPL | vertices_backward | 64 | upstream | 5.571 | 5.619 |  |  |
| SMPL | vertices | 64 | default | 1.591 | 1.607 | 1.747 | 0.91 |
| SMPL | vertices_backward | 64 | default | 4.854 | 4.882 | 5.571 | 0.87 |
| SMPL | fk | 64 | default | 0.726 | 0.736 |  |  |
| SMPL | rnea | 64 | default | 0.798 | 0.807 |  |  |
| SMPL | vertices | 1024 | upstream | 8.109 | 8.126 |  |  |
| SMPL | vertices_backward | 1024 | upstream | 22.070 | 22.104 |  |  |
| SMPL | vertices | 1024 | default | 8.054 | 8.089 | 8.109 | 0.99 |
| SMPL | vertices_backward | 1024 | default | 22.382 | 22.460 | 22.070 | 1.01 |
| SMPL | fk | 1024 | default | 0.738 | 0.765 |  |  |
| SMPL | rnea | 1024 | default | 0.808 | 0.831 |  |  |
| SMPL | vertices | 8192 | upstream | skipped: estimated 10.8 GB > 8 GB |  |  |  |
| SMPL | vertices | 8192 | default | skipped: estimated 10.8 GB > 8 GB |  |  |  |
| SMPL | vertices_backward | 8192 | default | skipped: estimated 14.4 GB > 8 GB |  |  |  |
| SMPL | fk | 8192 | default | 0.762 | 0.778 |  |  |
| SMPL | rnea | 8192 | default | 1.629 | 1.637 |  |  |
| SMPL-X | bake | 1 | mass | 45.372 | 46.737 |  |  |
| SMPL-X | bake | 1 | no_mass | 0.470 | 0.481 |  |  |
| SMPL-X | vertices | 1 | upstream | 2.651 | 2.757 |  |  |
| SMPL-X | vertices_backward | 1 | upstream | 10.674 | 15.534 |  |  |
| SMPL-X | vertices | 1 | default | 1.246 | 1.318 | 2.651 | 0.47 |
| SMPL-X | vertices_backward | 1 | default | 6.434 | 7.000 | 10.674 | 0.60 |
| SMPL-X | vertices | 1 | expression_stream_joints | 2.055 | 2.155 | 2.651 | 0.78 |
| SMPL-X | vertices_backward | 1 | expression_stream_joints | 9.076 | 10.271 | 10.674 | 0.85 |
| SMPL-X | vertices | 1 | expression_frozen_joints | 1.291 | 1.332 | 2.651 | 0.49 |
| SMPL-X | vertices_backward | 1 | expression_frozen_joints | 8.635 | 9.639 | 10.674 | 0.81 |
| SMPL-X | fk | 1 | default | 0.789 | 1.624 |  |  |
| SMPL-X | rnea | 1 | default | 0.716 | 0.781 |  |  |
| SMPL-X | vertices | 64 | upstream | 3.032 | 3.373 |  |  |
| SMPL-X | vertices_backward | 64 | upstream | 10.440 | 15.131 |  |  |
| SMPL-X | vertices | 64 | default | 1.827 | 1.959 | 3.032 | 0.60 |
| SMPL-X | vertices_backward | 64 | default | 5.279 | 7.526 | 10.440 | 0.51 |
| SMPL-X | vertices | 64 | expression_stream_joints | 2.694 | 2.824 | 3.032 | 0.89 |
| SMPL-X | vertices_backward | 64 | expression_stream_joints | 8.569 | 9.805 | 10.440 | 0.82 |
| SMPL-X | vertices | 64 | expression_frozen_joints | 1.911 | 2.024 | 3.032 | 0.63 |
| SMPL-X | vertices_backward | 64 | expression_frozen_joints | 9.474 | 10.407 | 10.440 | 0.91 |
| SMPL-X | fk | 64 | default | 0.855 | 1.697 |  |  |
| SMPL-X | rnea | 64 | default | 0.771 | 0.801 |  |  |
| SMPL-X | vertices | 1024 | upstream | 14.903 | 15.036 |  |  |
| SMPL-X | vertices_backward | 1024 | upstream | 36.785 | 36.987 |  |  |
| SMPL-X | vertices | 1024 | default | 12.291 | 12.368 | 14.903 | 0.82 |
| SMPL-X | vertices_backward | 1024 | default | 34.160 | 34.253 | 36.785 | 0.93 |
| SMPL-X | vertices | 1024 | expression_stream_joints | 13.377 | 13.481 | 14.903 | 0.90 |
| SMPL-X | vertices_backward | 1024 | expression_stream_joints | 35.497 | 35.645 | 36.785 | 0.96 |
| SMPL-X | vertices | 1024 | expression_frozen_joints | 12.364 | 12.475 | 14.903 | 0.83 |
| SMPL-X | vertices_backward | 1024 | expression_frozen_joints | 34.252 | 34.354 | 36.785 | 0.93 |
| SMPL-X | fk | 1024 | default | 0.832 | 0.888 |  |  |
| SMPL-X | rnea | 1024 | default | 0.815 | 0.858 |  |  |
| SMPL-X | vertices | 8192 | upstream | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | default | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | default | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | expression_stream_joints | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | expression_stream_joints | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | vertices | 8192 | expression_frozen_joints | skipped: estimated 16.5 GB > 8 GB |  |  |  |
| SMPL-X | vertices_backward | 8192 | expression_frozen_joints | skipped: estimated 22.0 GB > 8 GB |  |  |  |
| SMPL-X | fk | 8192 | default | 1.057 | 1.128 |  |  |
| SMPL-X | rnea | 8192 | default | 3.657 | 3.674 |  |  |
| MHR | bake | 1 | mass | 21.790 | 21.977 |  |  |
| MHR | bake | 1 | no_mass | 5.309 | 5.921 |  |  |
| MHR | vertices | 1 | torchscript | 2.176 | 2.305 |  |  |
| MHR | vertices | 1 | default | 2.500 | 2.634 | 2.176 | 1.15 |
| MHR | vertices_backward | 1 | default | 13.467 | 13.884 |  |  |
| MHR | fk | 1 | default | 1.612 | 1.815 |  |  |
| MHR | rnea | 1 | default | 0.770 | 0.857 |  |  |
| MHR | vertices | 64 | torchscript | 3.366 | 3.542 |  |  |
| MHR | vertices | 64 | default | 3.952 | 4.107 | 3.366 | 1.17 |
| MHR | vertices_backward | 64 | default | 11.360 | 16.250 |  |  |
| MHR | fk | 64 | default | 2.033 | 2.097 |  |  |
| MHR | rnea | 64 | default | 0.833 | 0.854 |  |  |
| MHR | vertices | 1024 | torchscript | 26.287 | 26.415 |  |  |
| MHR | vertices | 1024 | default | 26.831 | 26.973 | 26.287 | 1.02 |
| MHR | vertices_backward | 1024 | default | 71.860 | 71.991 |  |  |
| MHR | fk | 1024 | default | 2.084 | 2.258 |  |  |
| MHR | rnea | 1024 | default | 1.546 | 1.587 |  |  |
| MHR | vertices | 8192 | torchscript | skipped: estimated 29.0 GB > 8 GB |  |  |  |
| MHR | vertices | 8192 | default | skipped: estimated 29.0 GB > 8 GB |  |  |  |
| MHR | vertices_backward | 8192 | default | skipped: estimated 38.7 GB > 8 GB |  |  |  |
| MHR | fk | 8192 | default | 3.483 | 3.547 |  |  |
| MHR | rnea | 8192 | default | 12.106 | 12.142 |  |  |
