import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}.")
            return model
        except FileNotFoundError:
            raise ValueError(f"Model file '{self.model_path}' not found. Train the model first.")

    def predict(self, input_features):
        logger.info("Predicting using the loaded model...")
        return self.model.predict(input_features)

    def predict_proba(self, input_features):
        if hasattr(self.model, "predict_proba"):
            logger.info("Predicting probabilities using the loaded model...")
            return self.model.predict_proba(input_features)
        else:
            logger.warning("This model does not support probability predictions.")
            return None
