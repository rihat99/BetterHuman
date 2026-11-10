"""The README's first screen runs one example and shows its real output (principle 7.2).

Principle 7.2 is titled "Make the README's first screen say what the project is, prove it, install it, and run one
example with its real output", and its Check reads "A docs test asserts that the README's first `python` block,
followed by an `output` block, starts within the first 40 lines." The title puts the output on the first screen as
well, so the output fence must also close by line 40. ``tests/docs/test_snippets.py`` runs the fence and compares
the output.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).parents[2] / "README.md"
FIRST_SCREEN_LINES = 40


def test_the_first_python_fence_and_its_output_are_on_the_first_screen() -> None:
    """The first Python fence has an output fence docked under it, and that output fence closes by line 40."""
    lines = README.read_text().splitlines()
    python_open = next(i for i, line in enumerate(lines) if line.startswith("```python"))
    python_close = lines.index("```", python_open + 1)
    assert lines[python_close + 1].startswith("```output"), (
        "README.md's first Python fence has no output fence under it"
    )
    output_close = lines.index("```", python_close + 2)
    assert output_close + 1 <= FIRST_SCREEN_LINES, f"README.md's first example output closes on line {output_close + 1}"
