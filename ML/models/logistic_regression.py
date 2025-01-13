import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel:
    def __init__(self, max_iter=5000, solver="saga", C=0.1, degree=2):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.model = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(max_iter=max_iter, solver=solver, C=C))
        ])

    def train(self, X_train, y_train):
        self.logger.info("Starting training of Logistic Regression model with Polynomial Features...")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed successfully.")

    def predict(self, X):
        self.logger.info("Predicting with the trained Logistic Regression model...")
        return self.model.predict(X)

    def predict_proba(self, X):
        self.logger.info("Predicting probabilities with the Logistic Regression model...")
        return self.model.predict_proba(X)

    def save_model(self, path="models/logistic_regression_poly.pkl"):
        try:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved successfully at: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save the model: {e}")

    def load_model(self, path="models/logistic_regression_poly.pkl"):
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded successfully from: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load the model: {e}")
