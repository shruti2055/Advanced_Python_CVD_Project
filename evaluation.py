"""Evaluation Module.

Contains functions for evaluating models on test data.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Parameters
    ----------
    model : trained model
        The trained model.
    X_test : pd.DataFrame or np.ndarray
        Test features.
    y_test : pd.Series or np.ndarray
        Test target.

    Returns
    -------
    dict
        Dictionary with evaluation metrics: accuracy, precision, recall, f1_score.
    """
    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    return metrics, y_pred


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics for predictions."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }


def get_confusion_matrix(y_test, y_pred):
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_test : pd.Series or np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    np.ndarray
        Confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    return cm