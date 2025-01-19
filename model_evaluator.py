import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix
)

class ModelEvaluator:
    def __init__(self, model, X_test_list, y_test_reshaped):
        self.model = model
        self.X_test_list = X_test_list
        self.y_test_reshaped = y_test_reshaped
        self.evaluation_results = None

    def evaluate(self):
        y_pred = self.model.predict(self.X_test_list)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test_reshaped, axis=1)

        # Calculate metrics
        classification_report_str = classification_report(y_true_classes, y_pred_classes)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        cohen_kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
        mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
        confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)


        self.evaluation_results = {
            "classification_report": classification_report_str,
            "accuracy": accuracy,
            "f1": f1,
            "cohen_kappa": cohen_kappa,
            "mcc": mcc,
            "confusion_matrix": confusion_mat,
        }

    def get_evaluation_results(self):
        if self.evaluation_results is None:
            raise ValueError("You need to run `evaluate` before getting results.")
        return self.evaluation_results
