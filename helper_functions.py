import torch, copy
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, f1_score, roc_auc_score
)

class NetworkAnomalyDetector(nn.Module):
    """
    Fully-connected binary classifier for network anomaly detection.

    Architecture:
        Input  -> Linear(256) -> BN -> ReLU -> Dropout
               -> Linear(128) -> BN -> ReLU -> Dropout
               -> Linear(64)  -> BN -> ReLU -> Dropout
               -> Linear(32)  -> BN -> ReLU -> Dropout
               -> Linear(1)   -> Sigmoid

    Args:
        input_dim (int):   Number of features after preprocessing.
        dropout_p (float): Dropout probability for each hidden block.
    """

    def __init__(self, input_dim: int, dropout_p: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            # Block 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Block 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # Output
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        # He uniform initialisation for all Linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class PacketsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train(model, optimizer, scheduler, train_loader, val_loader, loss_func, device, num_epoch, patience):
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

        print(
            f"Epoch [{epoch:03d}/{num_epoch}]  "
            f"Train Loss: {train_stats['loss']:.4f}  |  "
            f"Val Loss: {val_stats['loss']:.4f}  |  "
            f"Val F1: {val_stats['f1']:.4f}  |  "
            f"Val AUC: {val_stats['roc_auc']:.4f}"
        )

        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
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


def one_hot_encode(df, columns):
    # create a copy of the dataframe
    df = df.copy()

    # create map of groups to protocol (necessary due to the sheer number of protocols)
    protocol_groups = {
        'common_transport': ['tcp', 'udp', 'sctp'],
        'routing': ['ospf', 'eigrp', 'egp', 'igp', 'nsfnet-igp', 'dgp', 'idrp', 'idpr', 'idpr-cmtp', 'sdrp', 'mhrp'],
        'tunneling': ['gre', 'ipip', 'l2tp', 'encap', 'etherip', 'mobile', 'ipcomp', 'ipnip', 'ip', 'micp'],
        'ipv6_family': ['ipv6', 'ipv6-frag', 'ipv6-route', 'ipv6-no', 'ipv6-opts'],
        'multicast': ['igmp', 'pim', 'vrrp', 'pgm', 'cbt'],
        'link_layer': ['arp', 'ax.25', 'fc', 'srp', 'il', 'ipx-n-ip'],
        'security': ['skip', 'tlsp', 'rsvp', 'kryptolan', 'secure-vmtp', 'aes-sp3-d', 'swipe', 'pri-enc'],
        'legacy': ['ggp', 'st2', 'argus', 'chaos', 'nvp', 'pup', 'xnet', 'mux', 'dcn', 'hmp', 'prm', 'trunk-1', 'trunk-2', 'xns-idp', 'leaf-1', 'leaf-2', 'irtp', 'rdp', 'netblt', 'mfe-nsp', 'merit-inp', '3pc', 'ddp', 'tp++', 'narp', 'any', 'cftp', 'sat-expak', 'ippc', 'sat-mon', 'cpnx', 'wsn', 'pvp', 'br-sat-mon', 'sun-nd', 'wb-mon', 'vmtp', 'ttp', 'vines', 'tcf', 'sprite-rpc', 'larp', 'mtp', 'bbn-rcc', 'bna', 'visa', 'ipcv', 'cphb', 'iso-tp4', 'wb-expak', 'sep', 'xtp', 'unas', 'iso-ip', 'aris', 'a/n', 'snp', 'compaq-peer', 'zero', 'ddx', 'iatp', 'stp', 'uti', 'sm', 'smp', 'isis', 'ptp', 'fire', 'crtp', 'crudp', 'sccopmce', 'iplt', 'pipe', 'sps', 'ib', 'emcon', 'gmtp', 'ifmp', 'pnni', 'qnx', 'scps']
    }

    # create a mapping of protocols to group
    proto_to_group = {proto: group for group, protos in protocol_groups.items() for proto in protos}

    if "proto_group" in columns:
        # map protocol to their respective group
        df['proto_group'] = df['proto'].map(proto_to_group).fillna('legacy')

    # one hot encode columns
    df_ohe = pd.get_dummies(df, columns=columns)

    # convert boolean columns to int
    bool_cols = df_ohe.select_dtypes(include='bool').columns
    df_ohe[bool_cols] = df_ohe[bool_cols].astype(int)
    df_ohe.drop(columns=["proto"], inplace=True)

    return df_ohe


def get_preprocessor():
    log_scale_cols = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload', 'dttl', 'sintpkt', 'dintpkt', 'sjit', 'djit', 'smeansz', 'dmeansz', 'ct_flw_http_mthd', 'trans_depth', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    std_scale_cols = ['tcprtt', 'synack', 'ackdat']

    minmax_cols = ['sttl', 'swin', 'dwin']

    # Columns needing clip before log
    clip_cols = {
        'sloss': 0.99,
        'dloss': 0.99,
        'res_bdy_len': 0.99
    }

    # Add clipped columns to log group
    log_scale_cols += list(clip_cols.keys())

    # Build pipeline for Log + StandardScalar
    log_transformer = Pipeline([
        ('log1p',  FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('log_scale',   log_transformer,  log_scale_cols),
        ('std_scale',   StandardScaler(), std_scale_cols),
        ('minmax',      MinMaxScaler(),   minmax_cols)
    ], remainder='passthrough')

    return preprocessor


def load_data(file):
    labels = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]

    df = pd.read_csv(file)
    df.columns = labels
    return df


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


def weighted_bce(pred, target, pos_weight):
    """BCE loss with up-weighting of the malicious (positive) class."""
    weights = torch.where(target == 1, pos_weight, torch.ones_like(target))
    return (nn.functional.binary_cross_entropy(pred, target, reduction='none') * weights).mean()


def print_results(stats, preds_bin, title):
    print(title)
    print(f"Loss      : {stats['loss']:.4f}")
    print(f"Accuracy  : {accuracy_score(stats['labels'], preds_bin):.4f}")
    print(f"Precision : {precision_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(stats['labels'], preds_bin, zero_division=0):.4f}")
    print(f"F1        : {stats['f1']:.4f}")
    print(f"ROC-AUC   : {stats['roc_auc']:.4f}")

