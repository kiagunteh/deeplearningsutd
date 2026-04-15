import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def one_hot_encode(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encodes specified columns in a DataFrame.

    Groups network protocols into broader categories before encoding when
    ``'proto_group'`` is included in ``columns``, then applies
    ``pd.get_dummies`` and drops the original ``proto`` column.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least a ``proto``
            column and all columns listed in ``columns``.
        columns (list[str]): Column names to one-hot encode. Include
            ``'proto_group'`` to trigger protocol-grouping logic.

    Returns:
        pd.DataFrame: A copy of ``df`` with the specified columns replaced
        by their one-hot encoded counterparts (boolean columns cast to int)
        and the original ``proto`` column removed.
    """
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


def get_preprocessor() -> ColumnTransformer:
    """Builds a sklearn ``ColumnTransformer`` for numerical feature preprocessing.

    Applies three distinct transformations depending on the feature group:

    * **log_scale** - log1p transform followed by ``StandardScaler`` for
      high-skew count/byte columns and clipped outlier columns.
    * **std_scale** - ``StandardScaler`` for approximately normal TCP timing
      columns.
    * **minmax** - ``MinMaxScaler`` for bounded integer fields.

    All remaining columns are passed through unchanged.

    Returns:
        sklearn.compose.ColumnTransformer: Fitted-ready preprocessor pipeline.
    """
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


def load_data(file: str) -> pd.DataFrame:
    """Loads a UNSW-NB15 CSV file and assigns canonical column names.

    Args:
        file (str | os.PathLike): Path to the CSV file. The file must contain
            exactly 49 columns in the original UNSW-NB15 field order (no
            header required; any existing header will be overwritten).

    Returns:
        pd.DataFrame: DataFrame with 49 named columns matching the
        UNSW-NB15 feature set, ending with ``attack_cat`` and ``label``.
    """
    labels = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"]

    df = pd.read_csv(file)
    df.columns = labels
    return df