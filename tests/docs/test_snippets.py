"""Execute the Python snippets in ``docs/**/*.md`` and ``README.md``; check that every referenced image exists."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
DOCS = REPO / "docs"
FAKE_SERVER = REPO / "tests" / "viz" / "fake_server.py"
MODELS_ENV = "BETTERHUMAN_MODELS_DIR"
PAGE_TIMEOUT_S = 120

# Fence info string: ```python [tags...]; a ``{ .no-run }`` attribute opts the block out of execution.
FENCE = re.compile(r"^```[ \t]*python(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
IMAGE = re.compile(r"!\[[^\]]*\]\((?P<path>[^)\s]+)")
OUTPUT_FENCE = re.compile(r"^(?P<indent>[ \t]*)```[ \t]*output[^\n]*\n(?P<body>.*?)^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL)
VIDEO_SRC = re.compile(r"""(?:src|poster)=["'](?P<path>[^"']+)["']""")
# ``BR_DOCS_CAPTURE=1`` writes each page's per-block stdout to ``$BR_DOCS_CAPTURE_DIR/<page>.outputs.json``.
CAPTURE = os.environ.get("BR_DOCS_CAPTURE") == "1"
CAPTURE_DIR = Path(os.environ.get("BR_DOCS_CAPTURE_DIR", "/tmp/br_docs_outputs"))
FIGURE = re.compile(r"(?P<path>(?:\.\./)*(?:figures|images)/[\w./-]+\.(?:svg|png|jpg|gif|webm|mp4|webp))")

# Runs in a fresh interpreter so the CPU-only environment is set before torch loads, and so viser never binds:
# ``viser.ViserServer`` is replaced by the recording fake from the viewer tests.
RUNNER = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("docs_fake_server", sys.argv[1])
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)
try:
    import viser
    viser.ViserServer = lambda *args, **kwargs: fake.FakeServer()
except ImportError:
    pass
namespace = {"__name__": "__docs__"}
outputs = []
for index, source in enumerate(sys.argv[2:]):
    if __import__("os").environ.get("BR_DOCS_CAPTURE") == "1":
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"<block {index}>", "exec"), namespace)
        outputs.append(buffer.getvalue())
    else:
        exec(compile(source, f"<block {index}>", "exec"), namespace)
if outputs:
    sys.stderr.write("\\n__BR_DOCS_OUTPUTS__" + __import__("json").dumps(outputs))
"""


def _pages() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


def _runnable_blocks(page: Path) -> list[str]:
    return [m["body"] for m in FENCE.finditer(page.read_text()) if "no-run" not in m["info"]]


def _pages_with_code() -> list[Path]:
    return [page for page in [*_pages(), REPO / "README.md"] if _runnable_blocks(page)]


@pytest.mark.parametrize("page", _pages_with_code(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_page_snippets_execute(page: Path) -> None:
    """Run a page's Python blocks in order in one namespace, CPU only, with a fake viser server."""
    if not os.environ.get(MODELS_ENV):
        pytest.skip(f"{MODELS_ENV} is unset: the snippets load licensed body weights")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER, str(FAKE_SERVER), *_runnable_blocks(page)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=PAGE_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, f"{page.relative_to(REPO)} failed:\n{completed.stderr[-4000:]}"
    if CAPTURE:
        outputs = json.loads(completed.stderr.rsplit("__BR_DOCS_OUTPUTS__", 1)[1])
        target = CAPTURE_DIR / page.relative_to(REPO).with_suffix(".outputs.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(outputs, indent=1))


def _output_fences(page: Path) -> list[tuple[int, str, str]]:
    """``(line, indent, body)`` of every ```output fence on the page."""
    text = page.read_text()
    return [(text.count("\n", 0, m.start()), m["indent"], m["body"]) for m in OUTPUT_FENCE.finditer(text)]


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_output_fences_follow_python(page: Path) -> None:
    """Every ```output pane docks directly under a ```python fence that closed on the previous line."""
    lines = page.read_text().splitlines()
    for line, indent, _ in _output_fences(page):
        assert line > 0 and lines[line - 1] == indent + "```", f"{page.relative_to(REPO)}:{line + 1} output fence is not docked"
        start = next(i for i in range(line - 2, -1, -1) if lines[i].startswith(indent + "```"))
        assert lines[start].startswith(indent + "```python"), f"{page.relative_to(REPO)}:{line + 1} output follows a non-python fence"


@pytest.mark.parametrize("page", [DOCS / "index.md", DOCS / "getting-started.md"], ids=lambda p: p.name)
def test_getting_started_outputs_fresh(page: Path) -> None:
    """The first line of each output pane matches what the preceding block really prints (cheap staleness guard)."""
    if not os.environ.get(MODELS_ENV):
        pytest.skip(f"{MODELS_ENV} is unset: the snippets load licensed body weights")
    text = page.read_text()
    fences = [(m.start(), m["body"]) for m in OUTPUT_FENCE.finditer(text)]
    if not fences:
        pytest.skip("no output fences")
    blocks = [(m.end(), m["body"], "no-run" in m["info"]) for m in FENCE.finditer(text)]
    runnable = [body for _, body, skip in blocks if not skip]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MPLBACKEND": "Agg", "BR_DOCS_CAPTURE": "1"}
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER, str(FAKE_SERVER), *runnable],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=PAGE_TIMEOUT_S, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    outputs = json.loads(completed.stderr.rsplit("__BR_DOCS_OUTPUTS__", 1)[1])
    for fence_start, body in fences:
        end, _, skip = max((b for b in blocks if b[0] <= fence_start), key=lambda b: b[0])
        if skip:
            continue
        actual = outputs[[b for b in blocks if not b[2]].index((end, _, skip))]
        first = body.splitlines()[0]
        assert actual.startswith(first), f"{page.name}: stale output pane, expected to start with {first!r}, got {actual[:120]!r}"


def test_index_has_runnable_snippet() -> None:
    """The landing page example is part of the snippet run."""
    assert _runnable_blocks(DOCS / "index.md")


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_referenced_images_exist(page: Path) -> None:
    """Every ``![...](path)`` and ``figures/*.svg`` / ``images/*`` reference resolves to a file."""
    text = page.read_text()
    paths = {m["path"] for m in IMAGE.finditer(text)} | {m["path"] for m in FIGURE.finditer(text)}
    paths |= {m["path"] for video in re.findall(r"<video.*?</video>", text, re.DOTALL) for m in VIDEO_SRC.finditer(video)}
    missing = [
        path
        for path in sorted(paths)
        if "://" not in path and not (page.parent / path.split("#")[0]).resolve().exists()
    ]
    assert not missing, f"{page.relative_to(REPO)} references missing files: {missing}"
