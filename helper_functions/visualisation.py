import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, log_loss, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)


def plot_training_curves(history: dict) -> None:
    """Plots training loss, validation F1, and validation ROC-AUC curves.

    Creates a 1x3 figure: the first panel shows train vs. validation loss,
    the second shows validation F1 score, and the third shows validation
    ROC-AUC, all plotted against epoch number.

    Args:
        history (dict): Training history dictionary with keys ``'train_loss'``,
            ``'val_loss'``, ``'val_f1'``, and ``'val_roc_auc'``, each mapping
            to a list of per-epoch scalar values.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, history['val_f1'], color='green')
    axes[1].set_title('Val F1 Score')
    axes[1].set_xlabel('Epoch')

    axes[2].plot(epochs, history['val_roc_auc'], color='darkorange')
    axes[2].set_title('Val ROC-AUC')
    axes[2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(stats: dict, preds_bin: np.ndarray, title: str) -> None:
    """Plots a seaborn heatmap of the binary confusion matrix.

    Args:
        stats (dict): Stats dictionary containing a ``'labels'`` key with
            ground-truth binary labels as a numpy array.
        preds_bin (np.ndarray): Binarised model predictions (0 or 1).
        title (str): Title displayed above the confusion matrix plot.
    """
    cm = confusion_matrix(stats['labels'], preds_bin)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def print_results(stats: dict, preds_bin: np.ndarray, title: str) -> None:
    """Prints a formatted classification metrics summary for the neural network model.

    Displays loss, accuracy, precision, recall, F1, and ROC-AUC, sourcing
    pre-computed values from ``stats`` where available.

    Args:
        stats (dict): Stats dictionary with keys ``'loss'``, ``'f1'``, and
            ``'roc_auc'`` (floats), and ``'labels'`` (np.ndarray of
            ground-truth labels).
        preds_bin (np.ndarray): Binarised model predictions (0 or 1) used
            to compute accuracy, precision, and recall.
        title (str): Header line printed above the metrics block.
    """
    print(title)
    print(f"Loss      : {stats['loss']:.4f}")
    print(f"Accuracy  : {accuracy_score(stats['labels'], preds_bin):.4f}")
    print(f"Precision : {precision_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"F1        : {stats['f1']:.4f}")
    print(f"ROC-AUC   : {stats['roc_auc']:.4f}")


def visualize_xgb_model(y_test: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> None:
    """Prints classification metrics and plots a confusion matrix for an XGBoost model.

    Computes and displays loss, accuracy, precision, recall, F1, and ROC-AUC,
    then renders a seaborn heatmap of the confusion matrix.

    Args:
        y_test (np.ndarray): Ground-truth binary labels.
        y_pred (np.ndarray): Binarised model predictions (0 or 1).
        y_pred_prob (np.ndarray): Raw predicted probabilities used for log
            loss and ROC-AUC computation.
    """
    print(f"Loss      : {log_loss(y_test, y_pred_prob):.4f}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title("Confusion Matrix")
    plt.show()

