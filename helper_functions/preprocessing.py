import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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