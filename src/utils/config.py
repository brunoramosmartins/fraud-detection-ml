from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
RUNS_DIR = ARTIFACTS_DIR / "runs"

RANDOM_STATE: int = 42
C_FP: float = 5.0

