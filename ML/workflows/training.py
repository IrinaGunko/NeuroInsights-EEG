import pickle
from ML.dataset_loader import DatasetLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, db_path, models):
        self.db_path = db_path
        self.models = models

    def train_model(self, model_name, label_column="cognitive_load_status", test_size=0.4, random_state=42):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not supported.")

        loader = DatasetLoader(self.db_path)
        X_train, X_test, y_train, y_test = loader.load_features(
            label_column=label_column, test_size=test_size, random_state=random_state
        )

        if label_column in X_train.columns:
            raise ValueError("Labels are included in the feature set. Remove them before training.")

        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)

        model_path = f"models/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model '{model_name}' saved to {model_path}.")

        return model, X_test, y_test
