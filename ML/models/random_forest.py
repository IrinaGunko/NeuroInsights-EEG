import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import os

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42, oob_score=True)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def inspect_feature_importance(self, feature_names):

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            self.logger.info("Feature Importances:")
            for idx in sorted_indices:
                self.logger.info(f"{feature_names[idx]}: {importances[idx]:.4f}")
        else:
            self.logger.warning("The model does not have feature importances available.")

    def log_oob_score(self):
        if hasattr(self.model, "oob_score_"):
            self.logger.info(f"OOB Score: {self.model.oob_score_:.4f}")
        else:
            self.logger.warning("OOB score is not enabled for this model.")


    def train(self, X_train, y_train, feature_names):
        self.logger.info("Starting training of RandomForestClassifier...")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed successfully.")

        self.inspect_feature_importance(feature_names)

        self.log_oob_score()

    def predict(self, X):
        self.logger.info("Predicting with the trained RandomForest model...")
        return self.model.predict(X)

    def predict_proba(self, X):
        self.logger.info("Predicting probabilities with the RandomForest model...")
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        self.logger.info("Evaluating the RandomForest model...")
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
        accuracy = accuracy_score(y_test, predictions)

        self.logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")

        return {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
        }

    def save_model(self, path="models/random_forest.pkl", overwrite=False):
        if not overwrite and os.path.exists(path):
            self.logger.warning(f"File already exists at {path}. Use overwrite=True to overwrite.")
            return
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved successfully at: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save the model: {e}")

    def load_model(self, path="models/random_forest.pkl"):
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded successfully from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load the model: {e}")
