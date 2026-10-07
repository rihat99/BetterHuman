"""Print a markdown table from a ``benchmarks/run.py`` JSON file: ``python benchmarks/report.py <json>``."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    result = json.loads(Path(sys.argv[1]).read_text())
    print(
        f"torch {result['torch']} | {result['device']} ({result['device_name']}) | threads {result['threads']} | "
        f"commit {result['git_commit'][:10]} | {result['dtype']}\n"
    )
    print("| family | op | batch | variant | median ms | p90 ms | baseline ms | ours / baseline |")
    print("|---|---|---:|---|---:|---:|---:|---:|")
    for row in result["rows"]:
        if "skipped" in row:
            cells = ["skipped: " + row["skipped"], "", "", ""]
        else:
            cells = [f"{row['median_ms']:.3f}", f"{row['p90_ms']:.3f}"]
            if "ratio_vs_baseline" in row:
                cells += [f"{row['baseline_median_ms']:.3f}", f"{row['ratio_vs_baseline']:.2f}"]
            else:
                cells += ["", ""]
        print("| " + " | ".join([row["family"], row["op"], str(row["batch"]), row["variant"], *cells]) + " |")


if __name__ == "__main__":
    main()
