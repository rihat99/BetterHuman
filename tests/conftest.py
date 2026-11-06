"""Session rules shared by every test: nothing skips or expects to fail, and a selected ``licensed`` test needs the
weights directory.

A skipped or xfailed outcome, at collection or at run time and however it was produced, fails instead (BetterRobot
decision record 0033). A test that cannot run on every machine carries a marker the tier commands deselect:
``licensed`` or ``cuda``.

A test marked ``licensed`` reads licensed weights through ``better_human.io.weight_path``. When any such test is
selected and ``$BETTERHUMAN_MODELS_DIR`` is unset or names no directory, collection stops with the fix, so a run
without weights fails once and loudly instead of reporting each test on its own. A file missing inside the
directory fails only the test that reads it.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from better_human.io import AssetNotFoundError, models_dir


def _fail_if_skipped(report: pytest.CollectReport | pytest.TestReport) -> None:
    if not report.skipped:
        return
    if hasattr(report, "wasxfail"):
        what, reason = "xfailed", report.wasxfail
    else:
        what, reason = "skipped", report.longrepr[2] if isinstance(report.longrepr, tuple) else report.longrepr
    report.outcome = "failed"
    report.longrepr = (
        f"{report.nodeid} {what}: {reason}\nNo test skips or expects to fail (BetterRobot decision record 0033): "
        "mark a test that needs the weights `licensed` or a GPU `cuda`, and let anything else fail."
    )


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    report = yield
    _fail_if_skipped(report)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    _fail_if_skipped(report)
    return report


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
