"""The data of the columns need to be read. This is a dummy version"""
#TODO : read from a valid csv
import pandas as pd
import numpy as np

csv = pd.read_csv("model/comnet14-flows.csv")
csv = csv.iloc[: , :-5]
df_split = np.array_split(csv,16)
train_data = df_split[1]
test_data = df_split[4]

#The data with what we want to teach our model
transfer_data = df_split[5]

CSV_HEADER = [
            "id",
            "expiration_id",
            "src_ip",
            "src_mac",
            "src_oui",
            "src_port",
            "dst_ip",
            "dst_mac",
            "dst_oui",
            "dst_port",
            "protocol",
            "ip_version",
            "vlan_id",
            "tunnel_id",
            "bidirectional_first_seen_ms",
            "bidirectional_last_seen_ms",
            "bidirectional_duration_ms",
            "bidirectional_packets",
            "bidirectional_bytes",
            "src2dst_first_seen_ms",
            "src2dst_last_seen_ms",
            "src2dst_duration_ms",
            "src2dst_packets",
            "src2dst_bytes",
            "dst2src_first_seen_ms",
            "dst2src_last_seen_ms",
            "dst2src_duration_ms",
            "dst2src_packets",
            "dst2src_bytes",
            "bidirectional_min_ps",
            "bidirectional_mean_ps",
            "bidirectional_stddev_ps",
            "bidirectional_max_ps",
            "src2dst_min_ps",
            "src2dst_mean_ps",
            "src2dst_stddev_ps",
            "src2dst_max_ps",
            "dst2src_min_ps",
            "dst2src_mean_ps",
            "dst2src_stddev_ps",
            "dst2src_max_ps",
            "bidirectional_min_piat_ms",
            "bidirectional_mean_piat_ms",
            "bidirectional_stddev_piat_ms",
            "bidirectional_max_piat_ms",
            "src2dst_min_piat_ms",
            "src2dst_mean_piat_ms",
            "src2dst_stddev_piat_ms",
            "src2dst_max_piat_ms",
            "dst2src_min_piat_ms",
            "dst2src_mean_piat_ms",
            "dst2src_stddev_piat_ms",
            "dst2src_max_piat_ms",
            "bidirectional_syn_packets",
            "bidirectional_cwr_packets",
            "bidirectional_ece_packets",
            "bidirectional_urg_packets",
            "bidirectional_ack_packets",
            "bidirectional_psh_packets",
            "bidirectional_rst_packets",
            "bidirectional_fin_packets",
            "src2dst_syn_packets",
            "src2dst_cwr_packets",
            "src2dst_ece_packets",
            "src2dst_urg_packets",
            "src2dst_ack_packets",
            "src2dst_psh_packets",
            "src2dst_rst_packets",
            "src2dst_fin_packets",
            "dst2src_syn_packets",
            "dst2src_cwr_packets",
            "dst2src_ece_packets",
            "dst2src_urg_packets",
            "dst2src_ack_packets",
            "dst2src_psh_packets",
            "dst2src_rst_packets",
            "dst2src_fin_packets",
            "splt_direction",
            "splt_ps",
            "splt_piat_ms",
            "application_name",
            "application_category_name",
            "application_is_guessed",
            "application_confidence",
            #"requested_server_name",
            #"client_fingerprint",
            #"server_fingerprint",
            #"user_agent",
            #"content_type",
            ]


NUMERIC_FEATURE_NAMES = [
                        "id",
                        "expiration_id",
                        "src_port",
                        "dst_port",
                        "protocol",
                        "ip_version",
                        "vlan_id",
                        "tunnel_id",
                        "bidirectional_first_seen_ms",
                        "bidirectional_last_seen_ms",
                        "bidirectional_duration_ms",
                        "bidirectional_packets",
                        "bidirectional_bytes",
                        "src2dst_first_seen_ms",
                        "src2dst_last_seen_ms",
                        "src2dst_duration_ms",
                        "src2dst_packets",
                        "src2dst_bytes",
                        "dst2src_first_seen_ms",
                        "dst2src_last_seen_ms",
                        "dst2src_duration_ms",
                        "dst2src_packets",
                        "dst2src_bytes",
                        "bidirectional_min_ps",
                        "bidirectional_mean_ps",
                        "bidirectional_stddev_ps",
                        "bidirectional_max_ps",
                        "src2dst_min_ps",
                        "src2dst_mean_ps",
                        "src2dst_stddev_ps",
                        "src2dst_max_ps",
                        "dst2src_min_ps",
                        "dst2src_mean_ps",
                        "dst2src_stddev_ps",
                        "dst2src_max_ps",
                        "bidirectional_min_piat_ms",
                        "bidirectional_mean_piat_ms",
                        "bidirectional_stddev_piat_ms",
                        "bidirectional_max_piat_ms",
                        "src2dst_min_piat_ms",
                        "src2dst_mean_piat_ms",
                        "src2dst_stddev_piat_ms",
                        "src2dst_max_piat_ms",
                        "dst2src_min_piat_ms",
                        "dst2src_mean_piat_ms",
                        "dst2src_stddev_piat_ms",
                        "dst2src_max_piat_ms",
                        "bidirectional_syn_packets",
                        "bidirectional_cwr_packets",
                        "bidirectional_ece_packets",
                        "bidirectional_urg_packets",
                        "bidirectional_ack_packets",
                        "bidirectional_psh_packets",
                        "bidirectional_rst_packets",
                        "bidirectional_fin_packets",
                        "src2dst_syn_packets",
                        "src2dst_cwr_packets",
                        "src2dst_ece_packets",
                        "src2dst_urg_packets",
                        "src2dst_ack_packets",
                        "src2dst_psh_packets",
                        "src2dst_rst_packets",
                        "src2dst_fin_packets",
                        "dst2src_syn_packets",
                        "dst2src_cwr_packets",
                        "dst2src_ece_packets",
                        "dst2src_urg_packets",
                        "dst2src_ack_packets",
                        "dst2src_psh_packets",
                        "dst2src_rst_packets",
                        "dst2src_fin_packets",
                        "application_is_guessed",
                        "application_confidence",
                        ]

# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "src_ip": sorted(list(train_data["src_ip"].unique())),
    "src_mac": sorted(list(train_data["src_mac"].unique())),
    "src_oui": sorted(list(train_data["src_oui"].unique())),
    "dst_ip": sorted(list(train_data["dst_ip"].unique())),
    "dst_mac": sorted(list(train_data["dst_mac"].unique())),
    "dst_oui": sorted(list(train_data["dst_oui"].unique())),
    "splt_direction": sorted(list(train_data["splt_direction"].unique())),
    "splt_ps": sorted(list(train_data["splt_ps"].unique())),
    "splt_piat_ms": sorted(list(train_data["splt_piat_ms"].unique())),
    #"application_name": sorted(list(train_data["application_name"].unique())), ##Label shouldnt be listed here
    "application_category_name": sorted(list(train_data["application_category_name"].unique())),
}

#TODO : find solution if one of the columns has no values, because now it would break the model
'''
# A list of the columns to ignore from the dataset. #Dropped them before
IGNORE_COLUMN_NAMES = [
"requested_server_name",
"client_fingerprint",
"server_fingerprint",
"user_agent",
"content_type"
]
'''
# Drop last N columns of dataframe instead of ignore
#train_data = train_data.iloc[: , :-5]


# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

# A list of all the input features.
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"] #+ IGNORE_COLUMN_NAMES (if have any)
    for feature_name in CSV_HEADER
]

# The name of the target feature.
TARGET_FEATURE_NAME = "application_name"
# A list of the labels of the target features.
label = "application_name"
classes = train_data[label].unique().tolist()
print(f"Label classes: {classes}")
TARGET_LABELS = classes
#TARGET_LABELS = ['RDP', 'NTP', 'NetBIOS', 'HTTP.Google', 'DNS.WindowsUpdate']