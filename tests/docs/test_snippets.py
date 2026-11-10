"""Run every Python fence of ``docs/**/*.md`` once and compare every ``output`` fence in full.

``better_robot.testing.snippets.check_page`` does the run and the comparison (its docstring has the rules: seeded,
CPU only, a fake viser server, floats to half a unit of their last printed digit, an opt-out only with a reason).
A page runs on the random-weight archives of ``tests/support/random_weights/`` (decision record 0016) with
``$BETTERHUMAN_MODELS_DIR`` unset, so a fence that reaches for licensed weights fails; a page that needs them is a
key of ``LICENSED``, with its reason, and runs in the release tier. A decision record's fences never run: an accepted
record keeps its text (principle 8.2), so its example shows the API as decided, not as it is now. This module also
checks that output fences dock under a Python fence and that every referenced image exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from better_robot.testing.snippets import check_page

REPO = Path(__file__).parents[2]
DOCS = REPO / "docs"
RECORDS = DOCS / "contracts" / "decisions"
OUTPUT_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```[ \t]*output[^\n]*\n(?P<body>.*?)^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL
)
IMAGE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)")
VIDEO_SRC = re.compile(r"""(?:src|poster)=["'](?P<path>[^"']+)["']""")
FIGURE = re.compile(r"(?<![\w/])(?P<path>(?:\.\./)*(?:figures|images)/[\w./-]+\.(?:svg|png|jpg|gif|webm|mp4|webp))")
FOLDER_REASONS = {
    "docs/contracts/": (
        "a rule's example illustrates the rule: its 'bad' half is wrong on purpose, and it names objects the page "
        "does not build"
    ),
}
LICENSED = {
    "docs/guides/load-a-family.md": "it teaches loading every family's licensed weights from $BETTERHUMAN_MODELS_DIR",
}
# Loading MHR, SOMA and SMPL-X from licensed files takes about a minute on a host whose transparent huge pages stall
# large reads (docs/guides/run-the-tests.md); a page that does it gets the time.
LICENSED_TIMEOUT_S = 600.0


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _name(page: Path) -> str:
    return page.relative_to(REPO).as_posix()


@pytest.mark.parametrize(
    "page",
    [
        pytest.param(p, id=_name(p), marks=[pytest.mark.licensed] if _name(p) in LICENSED else [])
        for p in _pages()
        if not p.is_relative_to(RECORDS)
    ],
)
def test_page_fences_run_and_print_their_output_fences(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every fence runs once; every output fence is compared in full; only a ``LICENSED`` page sees the weights."""
    if _name(page) not in LICENSED:
        monkeypatch.delenv("BETTERHUMAN_MODELS_DIR", raising=False)
    timeout_s = LICENSED_TIMEOUT_S if _name(page) in LICENSED else 120.0
    compared = check_page(page, root=REPO, folder_reasons=FOLDER_REASONS, timeout_s=timeout_s)
    assert compared == len(OUTPUT_FENCE.findall(page.read_text())), f"{page.name}: an output fence went unchecked"


def test_licensed_pages_exist() -> None:
    """A ``LICENSED`` key names a page, so a moved page cannot leave the release tier silently."""
    assert all((REPO / name).is_file() for name in LICENSED), sorted(LICENSED)


def _output_fences(page: Path) -> list[tuple[int, str, str]]:
    """``(line, indent, body)`` of every ```output fence on the page."""
    text = page.read_text()
    return [(text.count("\n", 0, m.start()), m["indent"], m["body"]) for m in OUTPUT_FENCE.finditer(text)]


@pytest.mark.parametrize("page", _pages(), ids=_name)
def test_output_fences_follow_python(page: Path) -> None:
    """Every ```output pane docks directly under a ```python fence that closed on the previous line."""
    lines = page.read_text().splitlines()
    for line, indent, _ in _output_fences(page):
        assert line > 0 and lines[line - 1] == indent + "```", (
            f"{page.relative_to(REPO)}:{line + 1} output fence is not docked"
        )
        start = next(i for i in range(line - 2, -1, -1) if lines[i].startswith(indent + "```"))
        assert lines[start].startswith(indent + "```python"), (
            f"{page.relative_to(REPO)}:{line + 1} output follows a non-python fence"
        )


def test_index_has_runnable_snippet() -> None:
    """The landing page example is part of the snippet run."""
    assert re.search(r"^```python\n", (DOCS / "index.md").read_text(), re.MULTILINE)


@pytest.mark.parametrize("page", [*_pages(), REPO / "README.md"], ids=lambda p: p.relative_to(REPO).as_posix())
def test_referenced_images_exist(page: Path) -> None:
    """Every ``![...](path)`` and ``figures/*.svg`` / ``images/*`` reference of the docs and README resolves."""
    text = page.read_text()
    paths = {m["path"] for m in IMAGE.finditer(text)} | {m["path"] for m in FIGURE.finditer(text)}
    # mkdocs rewrites Markdown links but leaves raw HTML alone, and with directory URLs a page lives one level
    # deeper in the built site than its source: a page becomes the index page of a folder named after it.
    built_dir = page.parent if page.name == "index.md" else page.parent / page.stem
    video_paths = {
        m["path"] for video in re.findall(r"<video.*?</video>", text, re.DOTALL) for m in VIDEO_SRC.finditer(video)
    }
    missing = [
        path
        for path in sorted(paths)
        if "://" not in path and not (page.parent / path.split("#")[0]).resolve().exists()
    ] + [path for path in sorted(video_paths) if not (built_dir / path).resolve().exists()]
    assert not missing, f"{page.relative_to(REPO)} references missing files: {missing}"
