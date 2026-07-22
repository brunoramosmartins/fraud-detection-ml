from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model(model_name: str, config: Dict[str, Any]) -> Any:
    """
    Factory for supported models.

    model_name: \"lr\", \"rf\", \"gb\", or \"lgbm\".
    config: hyperparameters dictionary.
    """
    name = model_name.lower()
    if name == "lgbm":
        # Imported lazily so the v1 models stay usable without lightgbm.
        from lightgbm import LGBMClassifier

        # Defaults are the EXP-003 configuration (ADR-006/ADR-007): the
        # serving-feasible Kaggle setup this served model reproduces.
        lgbm_params = config.get("lgbm", {})
        return LGBMClassifier(
            objective="binary",
            n_estimators=lgbm_params.get("n_estimators", 1000),
            learning_rate=lgbm_params.get("learning_rate", 0.05),
            num_leaves=lgbm_params.get("num_leaves", 192),
            min_data_in_leaf=lgbm_params.get("min_data_in_leaf", 100),
            feature_fraction=lgbm_params.get("feature_fraction", 0.8),
            bagging_fraction=lgbm_params.get("bagging_fraction", 0.8),
            bagging_freq=lgbm_params.get("bagging_freq", 1),
            random_state=lgbm_params.get("random_state", 42),
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "lr":
        lr_params = config.get("lr", {})
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=lr_params.get("C", 1.0),
                        class_weight="balanced",
                        max_iter=lr_params.get("max_iter", 1000),
                        solver="lbfgs",
                        n_jobs=-1,
                        random_state=lr_params.get("random_state", 42),
                    ),
                ),
            ]
        )
    if name == "rf":
        rf_params = config.get("rf", {})
        return RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 12),
            min_samples_leaf=rf_params.get("min_samples_leaf", 50),
            class_weight="balanced",
            n_jobs=-1,
            random_state=rf_params.get("random_state", 42),
        )
    if name == "gb":
        gb_params = config.get("gb", {})
        return GradientBoostingClassifier(
            n_estimators=gb_params.get("n_estimators", 80),
            max_depth=gb_params.get("max_depth", 5),
            learning_rate=gb_params.get("learning_rate", 0.1),
            min_samples_leaf=gb_params.get("min_samples_leaf", 100),
            subsample=gb_params.get("subsample", 0.8),
            random_state=gb_params.get("random_state", 42),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")

