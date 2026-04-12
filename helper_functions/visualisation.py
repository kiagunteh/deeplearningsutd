import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, log_loss, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)


def plot_training_curves(history):
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


def plot_confusion_matrix(stats, preds_bin, title):
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


def print_results(stats, preds_bin, title):
    print(title)
    print(f"Loss      : {stats['loss']:.4f}")
    print(f"Accuracy  : {accuracy_score(stats['labels'], preds_bin):.4f}")
    print(f"Precision : {precision_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"F1        : {stats['f1']:.4f}")
    print(f"ROC-AUC   : {stats['roc_auc']:.4f}")


def visualize_xgb_model(y_test, y_pred, y_pred_prob):
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

