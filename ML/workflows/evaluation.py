from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


class ModelEvaluator:
    def __init__(self, evaluation_options=None, report_dir="reports"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        self.evaluation_options = evaluation_options or {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1_score": True,
            "roc_auc": True,
            "log_loss": True,
            "confusion_matrix": True,
            "classification_report": False,
            "plot_confusion_matrix": True,
            "plot_roc_curve": True,
        }

    def evaluate_model(
        self,
        y_true,
        y_pred,
        y_pred_proba=None,
        average="binary",
        feature_importances=None,
        feature_names=None,
        model_name="model",
    ):
        metrics = {}

        if self.evaluation_options.get("accuracy", True):
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        if self.evaluation_options.get("precision", True):
            metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
        if self.evaluation_options.get("recall", True):
            metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
        if self.evaluation_options.get("f1_score", True):
            metrics["f1_score"] = f1_score(y_true, y_pred, average=average, zero_division=0)

        if y_pred_proba is not None:
            if self.evaluation_options.get("roc_auc", True):
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average=average)
                except ValueError:
                    self.logger.warning("ROC AUC could not be computed. Check inputs for compatibility.")
            if self.evaluation_options.get("log_loss", True):
                try:
                    metrics["log_loss"] = log_loss(y_true, y_pred_proba)
                except ValueError:
                    self.logger.warning("Log loss could not be computed. Check inputs for compatibility.")

        if self.evaluation_options.get("confusion_matrix", True):
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            if self.evaluation_options.get("plot_confusion_matrix", True):
                self._plot_confusion_matrix(cm, model_name)

        if self.evaluation_options.get("plot_roc_curve", True) and y_pred_proba is not None:
            self._plot_roc_curve(y_true, y_pred_proba, model_name)

        if self.evaluation_options.get("classification_report", False):
            metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

        if feature_importances is not None and feature_names is not None:
            self._log_feature_importances(feature_importances, feature_names, model_name)

        report_path = os.path.join(self.report_dir, f"{model_name}_evaluation.json")
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Saved evaluation report to {report_path}")

        self.logger.info("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            if metric == "confusion_matrix":
                self.logger.info(f"{metric}: \n{value}")
            else:
                self.logger.info(f"{metric}: {value}")

        return metrics

    def _plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plot_path = os.path.join(self.report_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved confusion matrix plot to {plot_path}")

    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (area = {:.2f})".format(roc_auc_score(y_true, y_pred_proba)))
        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="best")
        plot_path = os.path.join(self.report_dir, f"{model_name}_roc_curve.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved ROC curve plot to {plot_path}")

    def _log_feature_importances(self, importances, feature_names, model_name):
        feature_importances = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        self.logger.info("Feature Importances:")
        for feature, importance in feature_importances:
            self.logger.info(f"{feature}: {importance:.4f}")

        import pandas as pd
        df = pd.DataFrame(feature_importances, columns=["Feature", "Importance"])
        file_path = os.path.join(self.report_dir, f"{model_name}_feature_importances.csv")
        df.to_csv(file_path, index=False)
        self.logger.info(f"Saved feature importances to {file_path}")
