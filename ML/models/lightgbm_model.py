import lightgbm as lgb
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class LightGBMModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        logger.info("Training LightGBM...")
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "binary_logloss",
            "num_leaves": 100,
            "learning_rate": 0.1,
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params, lgb_train, num_boost_round=500)
        logger.info("LightGBM training complete.")

    def predict(self, X):
        logger.info("Predicting with LightGBM...")
        return (self.model.predict(X) > 0.5).astype(int)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"LightGBM model saved to {path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"LightGBM model loaded from {path}")
