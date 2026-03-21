# ADR-001 — `src/` Package Layout and Module Separation

## Status

Accepted

## Context

The project started as exploratory notebooks. When transitioning to a
reusable codebase in Phase 4, a decision was needed about how to organize
Python source code.

Options considered:
1. Flat module structure — all code in a single `fraud_detection.py` or
   a small set of files in the project root.
2. Monolithic `src/` package — one or two files under `src/` covering all
   responsibilities.
3. Layered `src/` package — separate subpackages for `data`, `features`,
   `models`, `pipelines`, and `utils`, each with a single responsibility.

## Decision

Option 3: a layered `src/` package with distinct subpackages.

```
src/
├── data/       loader, schema validation, temporal split
├── features/   feature registry, feature pipeline
├── models/     training factory, metrics, artifact management
├── pipelines/  end-to-end training orchestration
└── utils/      config constants, tracking, drift
```

## Consequences

**Positive:**
- Each module has a well-defined boundary and can be tested independently.
  Tests for `schema.py` do not need to know about `metrics.py`.
- The training pipeline (`pipelines/training_pipeline.py`) reads as an
  orchestration script — it calls the other modules but contains no
  business logic itself, making it easy to understand and modify.
- Adding new model types, feature sets, or monitoring metrics requires
  changes in one place only, without touching unrelated modules.
- Import paths make dependencies explicit: `from src.models.metrics import
  expected_loss` signals exactly what the calling module needs.

**Negative:**
- More files and directories than a flat structure. For a small project,
  this adds navigation overhead.
- Circular import risk if modules are not carefully scoped.
  (Mitigated: `utils/config.py` has no imports from other `src/` modules;
  `data/` and `features/` do not import from `models/`.)
- The `sys.path` injection in scripts (`PROJECT_ROOT` prepend) is a
  workaround for the absence of an installed package. A production codebase
  would install `src/` as a proper package via `pyproject.toml`.
