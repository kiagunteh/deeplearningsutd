"""
Hyperparameter tuning for NetworkAnomalyDetector using Optuna.

Results are stored in tuning.db (SQLite).
Best model weights are saved to tuning/best_model.pth.
Best hyperparameters are saved to tuning/best_params.json.

Usage:
    python tuning.py                  # run 50 trials (default)
    python tuning.py --n-trials 100   # run 100 trials
    python tuning.py --resume         # resume existing study
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from helper_functions.model import NetworkAnomalyDetector, PacketsDataset
from helper_functions.preprocessing import one_hot_encode, get_preprocessor, load_data
from helper_functions.training import run_epoch, weighted_bce

SEED = 42
_pending_weights: dict = {}  # trial.number -> state_dict; consumed by BestModelCallback
DATA_PATH = "data/UNSW-NB15_1.csv"
DB_URL = "sqlite:///tuning.db"
STUDY_NAME = "anomaly_detector_tuning"
SAVE_DIR = "tuning"

NUM_EPOCHS = 50
PATIENCE = 7

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess():
    df = load_data(DATA_PATH)
    df.drop(
        ["srcip", "dstip", "stime", "ltime", "stcpb", "dtcpb", "sport", "dsport", "attack_cat"],
        axis=1, inplace=True
    )
    df_ohe = one_hot_encode(df, columns=["proto_group", "state", "service"])

    y = df_ohe.pop("label").values.astype(np.float32)
    X = df_ohe

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_val, _, y_val, _ = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    preprocessor = get_preprocessor()
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)

    return X_train, X_val, y_train, y_val


def make_loss_func(y_train, device):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    raw_ratio = n_neg / n_pos
    pos_weight_value = max(1.5, raw_ratio)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    return lambda pred, target: weighted_bce(pred, target, pos_weight)


def objective(trial: optuna.Trial) -> float:
    # --- Suggest hyperparameters ---
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.7)
    scheduler_patience = trial.suggest_int("scheduler_patience", 2, 5)

    # --- DataLoaders ---
    train_loader = DataLoader(
        PacketsDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        PacketsDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    # --- Model, optimiser, scheduler ---
    model = NetworkAnomalyDetector(input_dim=INPUT_DIM, dropout_p=dropout_p).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        run_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                  loss_func=loss_func, device=DEVICE, training=True)
        val_stats = run_epoch(model=model, optimizer=optimizer, data_loader=val_loader,
                              loss_func=loss_func, device=DEVICE, training=False)

        scheduler.step(val_stats["loss"])

        preds_bin = (val_stats["probs"] >= 0.5).astype(int)
        val_f1 = f1_score(val_stats["labels"], preds_bin, zero_division=0)

        # Report to Optuna for pruning (report val_loss so lower = better with minimize direction)
        trial.report(val_stats["loss"], epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    trial.set_user_attr("best_val_f1", best_val_f1)
    trial.set_user_attr("best_val_loss", best_val_loss)

    _pending_weights[trial.number] = best_weights

    return best_val_loss  # minimise


class BestModelCallback:
    """Saves the best model weights whenever a new best trial is found."""

    def __init__(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.best_val = float("inf")

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        weights = _pending_weights.pop(trial.number, None)
        if trial.value is not None and trial.value < self.best_val:
            self.best_val = trial.value
            if weights is not None:
                torch.save(weights, os.path.join(self.save_dir, "best_model.pth"))
                print(
                    f"\n  [Callback] New best trial #{trial.number} | "
                    f"val_loss={trial.value:.4f} | "
                    f"val_f1={trial.user_attrs.get('best_val_f1', 0):.4f} | "
                    f"Weights saved to {self.save_dir}/best_model.pth"
                )
                # Save best params alongside the model
                params_path = os.path.join(self.save_dir, "best_params.json")
                with open(params_path, "w") as f:
                    json.dump(
                        {
                            "trial_number": trial.number,
                            "val_loss": trial.value,
                            "val_f1": trial.user_attrs.get("best_val_f1"),
                            "params": trial.params,
                        },
                        f,
                        indent=2,
                    )


def run_tuning(n_trials: int = 50, resume: bool = False):
    os.makedirs(SAVE_DIR, exist_ok=True)

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    if resume:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL, sampler=sampler, pruner=pruner)
        print(f"Resuming study '{STUDY_NAME}' with {len(study.trials)} existing trials.")
    else:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_URL,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    callback = BestModelCallback(save_dir=SAVE_DIR)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial: #{best.number}")
    print(f"  val_loss : {best.value:.4f}")
    print(f"  val_f1   : {best.user_attrs.get('best_val_f1', 'N/A')}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    params_path = os.path.join(SAVE_DIR, "best_params.json")
    print(f"\nBest params saved to  : {params_path}")
    print(f"Best model weights    : {os.path.join(SAVE_DIR, 'best_model.pth')}")
    print(f"Full study DB         : tuning.db  (open with optuna-dashboard)")

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study from tuning.db")
    args = parser.parse_args()

    print("Loading and preprocessing data (done once)...")
    X_train, X_val, y_train, y_val = load_and_preprocess()
    INPUT_DIM = X_train.shape[1]
    loss_func = make_loss_func(y_train, DEVICE)
    print(f"Input dim: {INPUT_DIM} | Train: {len(X_train)} | Val: {len(X_val)} | Device: {DEVICE}")

    run_tuning(n_trials=args.n_trials, resume=args.resume)
