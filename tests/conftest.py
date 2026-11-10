"""Session rules shared by every test: nothing skips or expects to fail, and a selected ``licensed`` test needs the
weights directory.

BetterRobot's plugin ``better_robot.testing.no_skips`` fails every skipped or xfailed outcome, at collection or at run
time and however it was produced (BetterRobot decision record 0033). A test that cannot run on every machine carries a
marker the tier commands deselect: ``licensed`` or ``cuda``.

A test marked ``licensed`` reads licensed weights through ``better_human.io.weight_path``. When any such test is
selected and ``$BETTERHUMAN_MODELS_DIR`` is unset or names no directory, collection stops with the fix, so a run
without weights fails once and loudly instead of reporting each test on its own. A file missing inside the
directory fails only the test that reads it.
"""

from __future__ import annotations

import pytest

from better_human.io import AssetNotFoundError, models_dir

pytest_plugins = ["better_robot.testing.no_skips"]


@pytest.hookimpl(trylast=True)  # after ``-m`` deselection, so only selected tests count
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any(item.get_closest_marker("licensed") for item in items):
        return
    try:
        models_dir()
    except AssetNotFoundError as error:
        raise pytest.UsageError(
            f"{error}\nThe selected tests include `licensed` ones; set the variable, or deselect them with "
            "-m 'not licensed'."
        ) from error
