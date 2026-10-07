# M1 test-port state (subagent ran out of budget)
Done: tests/parity/test_smpl{,x}_parity.py (new API, +static-joint case), tests/families/test_{smplh,mano,flame,densities}.py,
skin_affines/lbs_affines tests appended to tests/core/test_pipeline.py, tests/bodies/test_densities.py trashed.
Not done (still on the old API, collection/fixture errors): tests/bodies/test_smpl.py, tests/bodies/test_smplx.py (-> tests/families/),
tests/core/test_mass.py, tests/core/test_contact_frames.py, tests/dynamics/test_human_dynamics.py. tests/io/*, test_smoke.py,
test_classic.py, test_reduction.py already import cleanly (pass). test_reduction corrective-slice: float64 still differs by 7.1e-15
(1 ulp), so left float32 and failing as instructed.
Facts learned: classic zero pose gives q[:3] = identity.pelvis (q_neutral has 0 root translation); SMPL-H hand pose moves all
vertices slightly via posedirs (use a 1e-3 threshold); bh.load("mhr", path, lod=1, mass=False) works.
