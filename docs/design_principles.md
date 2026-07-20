# Design principles

These rules encode the owner's taste for this repository. They mirror
BetterRobot's rules; when the two disagree, BetterRobot's numerical contracts
win. Read this before writing or refactoring any code.

## Over-engineering is the enemy

1. **Minimum code that solves the problem.** No speculative flexibility, no
   config knobs nobody asked for, no error handling for impossible states.
2. **An abstraction needs ≥ 2 in-tree callers.** The parametric-body base
   class earns its existence by SMPL and SMPL-X both using it from day one;
   nothing else gets a base class until a second subclass exists.
3. **No forwarding layers, no raise-only placeholders, no "for later" hooks.**
   Unimplemented ideas live in `plan/` or the roadmap, not in the API.
4. **Line count is a design signal.** If a diff grows a module past what the
   change plainly needs, structure was added that the problem doesn't have.

## Human-specific rules

5. **If it is not human-specific, it does not belong here.** Kinematics,
   dynamics, Lie math, solvers, viewers come from BetterRobot. The moment a
   would-be BetterHuman utility is model-agnostic robotics, it is a
   BetterRobot contribution instead.
6. **One model, one file; shared math in shared modules.** A body model
   subclass declares *what* it is (topology, parameter spaces, weight-file
   schema, classic-convention quirks); the *how* (LBS, blend shapes,
   correctives, regressors) lives once in common modules. Copy-pasted
   forward passes across model classes — the smplx-package disease — are
   forbidden.
7. **Partial models are parameters, never subclasses.** `SMPLX(...,
   use_hands=False)` style spawn options, implemented by one generic
   reduction mechanism in the base machinery. No `SMPLXmini`-style class
   forks, ever.
8. **Classic conventions are quarantined at the boundary.** Inside
   BetterHuman there is exactly one state convention: BetterRobot-style `q`.
   Each model's native format (axis-angle vectors, offset root translations,
   PCA hand spaces) exists only inside its `from_classic` / `to_classic`
   pair, which must be exact inverses and covered by round-trip tests.
9. **Parity with the original implementation is the safety net.** Every
   model's forward pass is verified numerically against the original
   author-released code (run in its own pinned venv) before the model is
   considered supported. Never weaken a parity test to land a change.
10. **Weights are the user's.** No downloading, no bundling, no silent
    caching of licensed model files. Explicit path argument, documented
    directory convention, clear error messages when files are missing or
    malformed.

## Numerical and performance honesty

11. **Never silently fall back** to a slow path; warn once with the reason.
12. **No host synchronization in hot loops**; preserve dtype/device; never
    silently cast, move, or squeeze. Batched by default, trailing event
    axes, arbitrary leading batch axes — identical to BetterRobot contracts.
13. **GPU is a first-class target.** Device-conditional behavior is
    validated on both CPU and CUDA on this host.

## Process

14. **Alpha: breaking changes are fine, compatibility shims are not.** No
    aliases, no deprecation paths.
15. **Docs move with the code in the same change.**
16. **Deletion needs evidence; results reports are honest** — exact counts,
    deviations named as findings.
17. **One behavior, one test; test behavior, not trivia.** float32
    gradchecks, following the workspace testing rules.
