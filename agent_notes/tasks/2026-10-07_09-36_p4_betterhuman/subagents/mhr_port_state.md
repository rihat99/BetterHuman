# MHR port state (subagent, ran out of token budget)
Done: families/mhr.py (loader, MHRClassic, mhr_from_classic, mhr_to_classic), families/mhr_rig.py (AuthoredRig),
families/mhr_archive.py / mhr_tree.py / mhr_config.py (verbatim copies of bodies/mhr/{archive,tree,config}.py, imports
retargeted), families/__init__.py exports mhr. tests/parity/test_mhr_parity.py ported: 18 passed (31 s, CPU), same ATOLS.
Not done: tests/parity/test_mhr_torchscript.py port, tests/families/test_mhr.py port, per-LOD worst-error table, timing.
Note: assets need BETTERHUMAN_MODELS_DIR=$PWD/models when no path is passed.
