import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix

def evaluate_model_performance(model, X_test_list, y_test_reshaped):
    # Predict classes for test data
    y_pred = model.predict(X_test_list)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_reshaped, axis=1)

    # Calculate classification metrics
    classification_report_str = classification_report(y_true_classes, y_pred_classes)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    cohen_kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
    mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
    confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

    # Calculate various metrics
    TP = np.diag(confusion_mat)
    FP = confusion_mat.sum(axis=0) - TP
    FN = confusion_mat.sum(axis=1) - TP
    TN = confusion_mat.sum() - (TP + FP + FN)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return {
        "classification_report": classification_report_str,
        "accuracy": accuracy,
        "f1": f1,
        "cohen_kappa": cohen_kappa,
        "mcc": mcc,
        "confusion_matrix": confusion_mat,
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "ACC": ACC
    }
