# AI Agent Guidelines for GaussianBeam

Instructions for AI agents and coding assistants working in this Python toolkit for modeling Gaussian beam propagation with the ABCD matrix formalism.

## 1. Project Context

* This is a lightweight optics toolkit: Gaussian beams are described by the complex q parameter, and each optical element is a 2x2 ABCD ray-transfer matrix.
* Core dependencies are NumPy and SymPy only. Keep the core package this light; put heavier dependencies (e.g. Matplotlib for the GUI) behind optional extras in `pyproject.toml`.
* All physics code supports two modes via the `Mode` enum: `NUMERIC` (NumPy, complex floats) and `SYMBOLIC` (SymPy expressions). New physics features should preserve both paths.
* All internal quantities are SI (meters, radians). User-facing unit factors (`nm`, `um`, `mm`, `mrad`, ...) live in `gaussianbeam.units` and are plain floats multiplying to SI.

## 2. Testing is Encouraged

* Run existing examples freely to verify correctness after changes: `examples/numeric_demo.py` and `examples/symbolic_demo.py` are the primary sanity checks.
* Before finishing a task, run the relevant examples or write a small verification script to confirm behavior (e.g. compare against the analytic free-space formula `w(z) = w0 * sqrt(1 + (z/zR)^2)`).
* If you add new physics (a new optical element, a new beam property), prefer to add a minimal example or check demonstrating the expected result.

## 3. Running Code

* Always use the project interpreter `/home/jiaze/myenv/bin/python` (system Python lacks the dependencies).
* Scripts in `examples/` are safe to execute and are the primary way to exercise the package.
* Before running heavy numerical scans or optimizations (e.g. scripts under `data/`), do a quick dry-run with simple parameters to sanity-check behavior and estimate runtime.
* Cap each script invocation at 5 minutes. If a run exceeds 5 minutes, terminate it immediately, report the elapsed time, estimate how long the full run would likely take, and ask the user whether to continue.

## 4. Coding Guidelines

* Keep physics code explicit: document units, sign conventions (e.g. radius of curvature positive if the center is after the surface), and reference equations in docstrings.
* Prefer deterministic numeric defaults so examples remain reproducible.
* Keep the chain-style public API of `Beam` (`prop()`, `lens()`, `thick_lens()`, `flat_interface()`, `curved_interface()`, `copy()`) stable unless a breaking change is requested.
* When an element constructor receives non-numeric parameters it auto-switches to symbolic mode with a warning; preserve this behavior for new elements.

## 5. Git

* **Read-only git only.** Inspection commands (`status`, `log`, `diff`, `fetch`, `show`, etc.) are fine.
* **DO NOT** run commands that change git state—`add`, `commit`, `push`, `reset`, `checkout`, branch operations, etc.—unless the user explicitly asks.

## 6. Agent Logs

* Keep logs in `agent/logs/` (git-ignored), named `YYYYMMDD_HHMM_<title>.md` (e.g., `20260717_1530_gui_trace_sampling.md`).
* Log important errors, design decisions, failed attempts and why, and open questions.
* Logs are working notes; durable decisions belong in the relevant tracked document (e.g., instructions under `agent/instructions/`).
