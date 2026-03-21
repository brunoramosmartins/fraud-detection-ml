# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the
fraud detection ML system.

An ADR documents a significant design decision: what was decided, the
context that made it necessary, and the consequences — including the
downsides of the chosen approach.

## Index

| ADR | Title | Status |
|---|---|---|
| [ADR-001](ADR-001-src-package-layout.md) | `src/` package layout and module separation | Accepted |
| [ADR-002](ADR-002-artifact-storage-contract.md) | Artifact storage contract and metadata schema | Accepted |
| [ADR-003](ADR-003-temporal-validation-only.md) | Temporal split as the only valid validation strategy | Accepted |
| [ADR-004](ADR-004-feature-contract-fail-fast.md) | Feature contract enforcement: fail-fast on missing columns | Accepted |
| [ADR-005](ADR-005-app-state-for-model-storage.md) | `app.state` for runtime model storage | Accepted |
