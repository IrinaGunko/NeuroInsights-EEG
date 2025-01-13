import xgboost as xgb
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class XGBoostModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
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
        self.model = xgb.train(params, dtrain, num_boost_round=500)
        logger.info("XGBoost training complete.")

    def predict(self, X):
        logger.info("Predicting with XGBoost...")
        dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        return (self.model.predict(dtest) > 0.5).astype(int)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"XGBoost model saved to {path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"XGBoost model loaded from {path}")
