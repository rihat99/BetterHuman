# 0027 BetterHuman's coupling page lists all seven formats it versions

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
BetterRobot 0030 lists exactly six formats in the coupling standard, three of them BetterHuman's: `better_human.mhr`,
`better_human.soma` (record [0012](0012-bh-converted-archives.md)) and `better_human.frame_set` (record
[0013](0013-bh-package-data-and-load-speed.md)). BetterHuman versions four more (`io.formats.LATEST`), all from 0013:
`better_human.segmentation` and `better_human.density_tables` (package data) and `better_human.studio.segmentation` and
`better_human.studio.pose` (the studio's files). Work package X1 put all seven on BetterHuman's page
(`docs/contracts/coupling-standard.md`), which is a change to what 0030 lists and so needs a record (ruling R-3).

## Decision
We will list on BetterHuman's coupling page every format `io.formats.LATEST` names, each with the same promise: a
`format` and a `format_version` field, and every version written stays readable, with one exception: the studio
pose format refuses version 0, the format-less files the studio wrote before poses named a format, because they lack
the joint names a version-1 file needs (`docs/reference/formats/studio_pose.md`). BetterRobot's page keeps 0030's
three BetterHuman formats, the ones another repository reads; the four others are read and written by BetterHuman and
its studio.

## Example
```text
format                            direction    owner         promise
better_human.segmentation         read         BetterHuman   format and format_version fields; every version written stays readable
better_human.studio.pose          read, write  BetterHuman   format and format_version fields; every version written stays readable
```

## Consequences
A script that reads a studio pose or a segmentation file has the same promise as one that reads a frame set, and a
new versioned format joins the page in the commit that adds it to `io.formats.LATEST`.

## Alternatives
List the four in BetterRobot's standard: BetterRobot's page would carry formats no BetterRobot code and no other
repository reads. Leave them off both pages: their versions would carry no written promise.

## Compatibility
None: the formats and their readers exist; only the page changed.

## Enforcement
`tests/contract/test_coupling_standard.py` (BetterHuman's page against the library) and `tests/contract/test_formats.py`
(each format's promise, and a spec page under `docs/reference/formats/` for each).

## Open questions
1. BetterHuman cannot write BetterRobot 0030's Status line: does BetterRobot record this amendment of 0030's list
   itself, or is this record enough?
2. The coupling page's `better_human.studio.pose` row, a frozen line, states the promise without the version-0
   exception; correcting it changes a frozen line of the coupling standard, which needs the owner's decision.
