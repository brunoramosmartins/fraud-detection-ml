# Contributing

This document describes how to set up a development environment, run the
test suite, and follow the project's commit and branch conventions.

---

## Environment Setup

**Prerequisites:** Python 3.10, Git, Docker (optional).

```bash
# Clone the repository
git clone https://github.com/brunoramosmartins/fraud-detection-ml.git
cd fraud-detection-ml

# Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate      # Linux / macOS

# Install development dependencies (includes test and notebook extras)
pip install -r requirements-dev.txt
```

Place the IEEE-CIS dataset files in `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests use lightweight stubs — no real model artifacts or dataset files
are required to run the test suite.

---

## Training a Model

```bash
python scripts/train_model.py \
  --model gb \
  --config configs/model_gb_v1.yml \
  --dataset-version ieee-cis-original
```

The trained artifact is saved to `artifacts/models/`.

---

## Commit Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

Format: `type(scope): description`

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without behavior change |
| `test` | Test additions or corrections |
| `chore` | Tooling, config, CI |

Examples:

```
feat(api): add /health endpoint with model readiness check
fix(drift): handle zero-denominator in PSI computation
docs(reporting): add executive summary document
test(api): add 422 error case for missing features
```

---

## Branch Convention

```
feature/phase-XX-<short-description>
fix/<short-description>
docs/<short-description>
```

---

## Pull Requests

- One PR per phase or logical unit of work
- PR description must include: summary, context, list of changes, and
  acceptance criteria verification
- All tests must pass before merge
- No binary files, dataset files, or monitoring output artifacts in PRs
