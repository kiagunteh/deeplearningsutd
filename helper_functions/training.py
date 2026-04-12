import torch, copy
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score


def run_epoch(model, optimizer, data_loader, loss_func, device, training):
    """
    One forward (and optionally backward) pass over *loader*.
    Returns a dict with loss, f1, roc_auc, probs, labels.
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


def train(model, optimizer, scheduler, train_loader, val_loader, loss_func, device, num_epoch, patience, verbose=True):
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


def weighted_bce(pred, target, pos_weight):
    """BCE loss with up-weighting of the malicious (positive) class."""
    weights = torch.where(target == 1, pos_weight, torch.ones_like(target))
    return (nn.functional.binary_cross_entropy(pred, target, reduction='none') * weights).mean()

