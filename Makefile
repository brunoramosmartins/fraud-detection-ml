# Fraud Detection ML System — Makefile
# Requires: Python 3.10+, virtualenv at .venv/, dataset files in data/raw/
#
# Usage:
#   make setup     Install all dev dependencies
#   make train     Train the Gradient Boosting model
#   make test      Run the full test suite
#   make api       Start the scoring API on port 8000 (foreground)
#   make simulate  Send 5 batches of transactions to the API (API must be running)
#   make monitor   Run drift + performance monitoring on the latest predictions log
#   make demo      Full pipeline: train → test → print API instructions
#   make clean     Remove generated outputs (pycache, monitoring artifacts)

PYTHON    := .venv/Scripts/python
PIP       := .venv/Scripts/pip
PYTEST    := .venv/Scripts/pytest
UVICORN   := .venv/Scripts/uvicorn
CONFIG    := configs/model_lgbm_v2.yml
REFERENCE := data/raw/train_transaction.csv

.PHONY: setup train test api simulate monitor demo clean help

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	@echo ">>> Creating virtual environment..."
	python -m venv .venv
	@echo ">>> Installing project + dev dependencies (pyproject.toml)..."
	$(PIP) install -e ".[dev]"
	@echo ">>> Setup complete. Activate with: source .venv/Scripts/activate"

# ── Training ─────────────────────────────────────────────────────────────────
train:
	@echo ">>> Training LightGBM v2 model (served)..."
	$(PYTHON) scripts/train_model.py --model lgbm --config $(CONFIG)
	@echo ">>> Model artifact saved to artifacts/models/"

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	@echo ">>> Running test suite..."
	$(PYTEST) tests/ -v
	@echo ">>> All tests passed."

# ── API ──────────────────────────────────────────────────────────────────────
api:
	@echo ">>> Starting scoring API on http://localhost:8000"
	@echo ">>> Press Ctrl+C to stop."
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

# ── Simulation ───────────────────────────────────────────────────────────────
simulate:
	@echo ">>> Sending 5 batches of 100 transactions to the API..."
	@echo ">>> Make sure the API is running in another terminal (make api)."
	$(PYTHON) scripts/simulate_transactions.py \
		--api-url http://localhost:8000/predict \
		--batch-size 100 \
		--max-batches 5 \
		--sleep-seconds 1.0
	@echo ">>> Predictions saved to artifacts/monitoring/predictions/"

# ── Monitoring ───────────────────────────────────────────────────────────────
monitor:
	@echo ">>> Running drift and performance monitoring..."
	@LATEST=$$(ls -t artifacts/monitoring/predictions/predictions_*.csv 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "ERROR: No predictions log found. Run 'make simulate' first."; \
		exit 1; \
	fi; \
	echo ">>> Using predictions: $$LATEST"; \
	$(PYTHON) scripts/monitor_model.py \
		--reference-path $(REFERENCE) \
		--predictions-path $$LATEST \
		--psi-threshold 0.2
	@echo ">>> Reports saved to artifacts/monitoring/drift/ and artifacts/monitoring/performance/"

# ── Full Demo ────────────────────────────────────────────────────────────────
demo: train test
	@echo ""
	@echo "============================================================"
	@echo "  Demo pipeline complete."
	@echo "  Model trained and all tests passing."
	@echo ""
	@echo "  Next steps for a live demo:"
	@echo "    1. Terminal 1: make api"
	@echo "    2. Terminal 2: make simulate"
	@echo "    3. Terminal 2: make monitor"
	@echo ""
	@echo "  API health check:"
	@echo "    curl http://localhost:8000/health"
	@echo "============================================================"

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	@echo ">>> Removing pycache and generated outputs..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf artifacts/monitoring/predictions/*.csv 2>/dev/null || true
	rm -rf artifacts/monitoring/drift/*.json 2>/dev/null || true
	rm -rf artifacts/monitoring/performance/*.json 2>/dev/null || true
	@echo ">>> Clean complete."

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  setup     Install dev dependencies"
	@echo "  train     Train the served LightGBM v2 model"
	@echo "  test      Run pytest suite"
	@echo "  api       Start FastAPI on port 8000"
	@echo "  simulate  Send transaction batches to API"
	@echo "  monitor   Compute PSI and performance metrics"
	@echo "  demo      train + test + print instructions"
	@echo "  clean     Remove generated outputs"
