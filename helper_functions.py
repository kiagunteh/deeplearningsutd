import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score


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


def one_hot_encode(df, columns):
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


def normalize(df, X_train, X_val, X_test):
    log_scale_cols = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload', 'dttl', 'sintpkt', 'dintpkt', 'sjit', 'djit', 'smeansz', 'dmeansz', 'ct_flw_http_mthd', 'trans_depth', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    std_scale_cols = ['tcprtt', 'synack', 'ackdat']

    minmax_cols = ['sttl', 'swin', 'dwin']

    # Columns needing clip before log
    clip_cols = {
        'sloss': 0.99,
        'dloss': 0.99,
        'res_bdy_len': 0.99
    }

    # Apply clipping before the pipeline
    for col, quantile in clip_cols.items():
        cap = X_train[col].quantile(quantile)  # fit on train only
        X_train[col] = X_train[col].clip(upper=cap)
        X_val[col]   = X_val[col].clip(upper=cap)
        X_test[col]  = X_test[col].clip(upper=cap)

    # Add clipped columns to log group
    log_scale_cols += list(clip_cols.keys())

    # Add binary flag for res_bdy_len before dropping into pipeline
    df['has_response_body'] = (df['res_bdy_len'] > 0).astype(np.float32)

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

    # Fit on train only, transform all splits
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test


def load_data(file):
    labels = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]

    df = pd.read_csv(file)
    df.columns = labels
    return df

