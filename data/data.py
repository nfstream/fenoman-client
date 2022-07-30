from patterns.singleton import singleton
import pandas as pd
from sklearn.model_selection import train_test_split
from configuration.data_configuration import *
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats


@singleton
class Data:
    def __init__(self, df: str = TEST_DATAFRAME) -> None:
        self.__data = pd.read_csv(df)

        # TODO át kell ezt az egészet írni mert ez igy nem jó átemeltem a comnetből

        comnet14 = self.__data

        self.__data_reduced = self.__data.drop(DROP_VARIABLES, axis='columns')
        comnet14_reduced = self.__data_reduced

        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='bidirectional')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='first_seen')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='last_seen')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_syn_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_ece_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_ack_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_rst_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_cwr_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_urg_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_psh_')), axis='columns')
        comnet14_reduced = comnet14_reduced.drop(list(comnet14_reduced.filter(regex='_fin_')), axis='columns')

        # Give meaning to the values of the protocol feature
        comnet14_reduced['protocol'] = comnet14_reduced['protocol'].map({6: 'TCP', 17: 'UDP'})
        # Give meaning to the values of the expiration_id feature
        comnet14_reduced['expiration_id'] = comnet14_reduced['expiration_id'].map({0: 'idle', 1: 'active'})

        # Drop non TCP or UDP traffic
        comnet14_reduced = comnet14_reduced.dropna()
        comnet14 = comnet14[comnet14['protocol'] != 1]

        def outlier_removal(df):
            # Calculate the Z-scores
            z = np.abs(stats.zscore(df)).fillna(
                0)  # Constant results in NaN ==> change to 0 otherwise would drop every row

            # By setting variable 'coef', different number of outliers will be eliminated
            # Here we use the Three Sigma Rule
            coef = 3
            df_z = df[(z < coef).all(axis=1)]

            return df_z

        comnet14_w_outliers = comnet14_reduced.copy()
        comnet14_w_outliers['protocol'] = comnet14['protocol']  # Remove the legible encoding
        comnet14_w_outliers['expiration_id'] = comnet14['expiration_id']  # Remove the legible encoding

        label_encoder = LabelEncoder()
        comnet14_w_outliers['application_name'] = label_encoder.fit_transform(
            comnet14_w_outliers['application_name'])  # LabelEncode text data

        comnet14_wo_outliers = outlier_removal(comnet14_w_outliers)
        target = comnet14_wo_outliers['application_name']
        comnet14_wo_outliers = comnet14_wo_outliers.drop(['application_name'], axis='columns')

        n_features = 12

        def selector(score_function, X, y, n_features):
            selector = SelectKBest(score_function, k=n_features)
            selector.fit_transform(X, y)
            cols = selector.get_support(indices=True)
            return X.iloc[:, cols]

        X_f = selector(f_classif, comnet14_wo_outliers, target, n_features)
        print(X_f.columns)

        # scale
        comnet_scaled_minmax = MinMaxScaler().fit_transform(X_f)

        comnet_scaled_minmax = pd.DataFrame(comnet_scaled_minmax,
                                            columns=X_f.columns,
                                            index=X_f.index)

        # normalize
        comnet_normalized = Normalizer().fit_transform(X_f)

        comnet_normalized = pd.DataFrame(comnet_normalized,
                                         columns=X_f.columns,
                                         index=X_f.index)

        protocol_df = X_f['protocol'].map({6: 'TCP', 17: 'UDP'})
        protocol_encoded = pd.get_dummies(protocol_df, prefix='protocol')

        # Drop the non-encoded feature
        comnet_scaled_minmax_encoded = comnet_scaled_minmax.drop(['protocol'], axis='columns')
        comnet_normalized_encoded = comnet_normalized.drop(['protocol'], axis='columns')

        # Add to scaled DataFrames
        comnet_scaled_minmax_encoded = comnet_scaled_minmax_encoded.join(protocol_encoded)
        comnet_normalized_encoded = comnet_normalized_encoded.join(protocol_encoded)

        self.__data = comnet_scaled_minmax_encoded.join(target)

    def load_data(self, target_field: str = TARGET_VARIABLE):
        train, validation = train_test_split(self.__data, test_size=TRAIN_VALIDATION_SPLIT)

        def __separate_target(df):
            return df.loc[:, ~df.columns.isin([target_field])], df[target_field]

        x_train, y_train = __separate_target(train)
        x_val, y_val = __separate_target(validation)

        return x_train, y_train, x_val, y_val


data = Data()
