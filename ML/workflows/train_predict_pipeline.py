from ML.dataset_loader import DatasetLoader
from ML.workflows.evaluation import ModelEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "data/neuroinsights.db"
REPORT_DIR = "reports"


class GradientBoostingModels:
    def __init__(self):
        self.models = {
            "lightgbm": None,
            "xgboost": None,
        }

    def train_lightgbm(self, X_train, y_train):
        logger.info("Training LightGBM...")
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "binary_logloss",
            "num_leaves": 100,
            "learning_rate": 0.1,
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.models["lightgbm"] = lgb.train(params, lgb_train, num_boost_round=500)
        logger.info("LightGBM training complete.")

    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost...")
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
        self.models["xgboost"] = xgb.train(params, dtrain, num_boost_round=500)
        logger.info("XGBoost training complete.")

    def save_model(self, model_name, path):
        if model_name not in self.models or self.models[model_name] is None:
            logger.error(f"Model {model_name} is not trained or does not exist.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.models[model_name], f)
        logger.info(f"{model_name} saved to {path}.")

    def load_model(self, model_name, path):
        try:
            with open(path, "rb") as f:
                self.models[model_name] = pickle.load(f)
            logger.info(f"{model_name} loaded from {path}.")
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")


def train_and_evaluate(models_to_train):
    loader = DatasetLoader(DB_PATH)
    X_train, X_test, y_train, y_test = loader.load_features()

    evaluator = ModelEvaluator(report_dir=REPORT_DIR)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    traditional_models = {
        "logistic_regression": LogisticRegression(max_iter=5000, solver="saga", C=0.01),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    gb_models = GradientBoostingModels()

    for model_name in models_to_train:
        predictions_proba = None
        feature_importances = None
        feature_names = X_train.columns.tolist()

        if model_name in ["lightgbm", "xgboost"]:
            if model_name == "lightgbm":
                gb_models.train_lightgbm(X_train, y_train)
                gb_models.save_model("lightgbm", f"{models_dir}/lightgbm.pkl")
                predictions_proba = gb_models.models["lightgbm"].predict(X_test)
                predictions = (predictions_proba > 0.5).astype(int)
                feature_importances = gb_models.models["lightgbm"].feature_importance(importance_type="gain")
            elif model_name == "xgboost":
                gb_models.train_xgboost(X_train, y_train)
                gb_models.save_model("xgboost", f"{models_dir}/xgboost.pkl")
                dtest = xgb.DMatrix(X_test, feature_names=feature_names)
                predictions_proba = gb_models.models["xgboost"].predict(dtest)
                predictions = (predictions_proba > 0.5).astype(int)
                feature_importances = gb_models.models["xgboost"].get_score(importance_type="gain")
                feature_importances = [feature_importances.get(f, 0) for f in feature_names]
        elif model_name in traditional_models:
            logger.info(f"Training {model_name}...")
            model = traditional_models[model_name]
            model.fit(X_train, y_train)
            model_path = f"{models_dir}/{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"{model_name} saved to {model_path}.")
            predictions = model.predict(X_test)
            predictions_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            feature_importances = (
                model.feature_importances_ if hasattr(model, "feature_importances_") else None
            )

        logger.info(f"Evaluating {model_name}...")
        metrics = evaluator.evaluate_model(
            y_true=y_test,
            y_pred=predictions,
            y_pred_proba=predictions_proba,
            average="binary",
            feature_importances=feature_importances,
            feature_names=feature_names,
            model_name=model_name,
        )
        logger.info(f"Metrics for {model_name}: {metrics}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "lightgbm", "xgboost"],
        help="Specify models to train (default: all).",
    )
    args = parser.parse_args()
    train_and_evaluate(args.models)
