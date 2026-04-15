import torch, copy
import numpy as np
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score


def run_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    loss_func: Callable,
    device: torch.device,
    training: bool,
) -> dict:
    """Runs one forward (and optionally backward) pass over a data loader.

    Sets the model to train or eval mode accordingly, accumulates predictions
    and labels across all batches, and computes aggregate metrics.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): Optimizer used for parameter
            updates (only applied when ``training=True``).
        data_loader (torch.utils.data.DataLoader): DataLoader yielding
            ``(X_batch, y_batch)`` tuples.
        loss_func (callable): Loss function with signature
            ``loss_func(preds, targets) -> torch.Tensor``.
        device (torch.device): Device to move batches onto before inference.
        training (bool): If ``True``, performs a backward pass and parameter
            update; otherwise runs in inference mode with no gradients.

    Returns:
        dict: A dictionary with the following keys:

        * ``'loss'`` (float): Mean loss over the full dataset.
        * ``'f1'`` (float): F1 score at a 0.5 decision threshold.
        * ``'roc_auc'`` (float): Area under the ROC curve.
        * ``'probs'`` (np.ndarray): Raw predicted probabilities, shape
          ``(n_samples,)``.
        * ``'labels'`` (np.ndarray): Ground-truth labels, shape
          ``(n_samples,)``.
    """
    model.train() if training else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss  = loss_func(preds, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(X_batch)
            all_probs.append(preds.cpu().detach().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_probs  = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_preds  = (all_probs >= 0.5).astype(int)

    return {
        'loss':    total_loss / len(data_loader.dataset),
        'f1':      f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'probs':   all_probs,
        'labels':  all_labels,
    }


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_func: Callable,
    device: torch.device,
    num_epoch: int,
    patience: int,
    verbose: bool = True,
) -> tuple[dict, nn.Module]:
    """Trains a model with early stopping and learning-rate scheduling.

    Runs up to ``num_epoch`` training epochs, evaluating on the validation set
    after each one. Tracks the best validation loss and restores those weights
    before returning. Training halts early if validation loss does not improve
    for ``patience`` consecutive epochs.

    Args:
        model (torch.nn.Module): Model to train (modified in-place).
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning-rate
            scheduler stepped on validation loss each epoch.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_func (callable): Loss function passed to ``run_epoch``.
        device (torch.device): Device to run training on.
        num_epoch (int): Maximum number of training epochs.
        patience (int): Number of epochs without improvement before early stopping.
        verbose (bool, optional): If ``True``, prints per-epoch metrics and
            improvement messages. Defaults to ``True``.

    Returns:
        tuple[dict, torch.nn.Module]: A ``(history, model)`` pair where:

        * ``history`` is a dict with lists ``'train_loss'``, ``'val_loss'``,
          ``'val_f1'``, and ``'val_roc_auc'``, one entry per completed epoch.
        * ``model`` is the input model with best-validation-loss weights loaded.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_roc_auc': []}
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0

    for epoch in range(1, num_epoch + 1):
        train_stats = run_epoch(model=model, optimizer=optimizer, data_loader=train_loader, loss_func=loss_func, device=device, training=True)
        val_stats = run_epoch(model=model, optimizer=optimizer, data_loader=val_loader, loss_func=loss_func, device=device, training=False)

        scheduler.step(val_stats['loss'])

        history['train_loss'].append(train_stats['loss'])
        history['val_loss'].append(val_stats['loss'])
        history['val_f1'].append(val_stats['f1'])
        history['val_roc_auc'].append(val_stats['roc_auc'])

        if verbose:
            print(
                f"Epoch [{epoch:03d}/{num_epoch}]  "
                f"Train Loss: {train_stats['loss']:.4f}  |  "
                f"Val Loss: {val_stats['loss']:.4f}  |  "
                f"Val F1: {val_stats['f1']:.4f}  |  "
                f"Val AUC: {val_stats['roc_auc']:.4f}"
            )

        # Save best model in memory
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"  ✓  New best val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    # Reload best weights
    model.load_state_dict(best_weights)
    print("\nTraining complete. Best weights reloaded.")
    return history, model


def weighted_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | torch.Tensor,
) -> torch.Tensor:
    """Computes binary cross-entropy loss with up-weighting for the positive class.

    Assigns ``pos_weight`` to samples where ``target == 1`` (malicious) and
    ``1.0`` to samples where ``target == 0`` (benign), then returns the
    weighted mean loss.

    Args:
        pred (torch.Tensor): Predicted probabilities, shape ``(batch, 1)``.
        target (torch.Tensor): Ground-truth binary labels, shape ``(batch, 1)``.
        pos_weight (float | torch.Tensor): Scalar weight applied to positive
            (malicious) samples to compensate for class imbalance.

    Returns:
        torch.Tensor: Scalar weighted mean BCE loss.
    """
    weights = torch.where(target == 1, pos_weight, torch.ones_like(target))
    return (nn.functional.binary_cross_entropy(pred, target, reduction='none') * weights).mean()

