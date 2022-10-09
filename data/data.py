from patterns.singleton import singleton
import pandas as pd
from sklearn.model_selection import train_test_split
from configuration.data_configuration import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Tuple, Any


@singleton
class Data:
    def __init__(self, df: str = TEST_DATAFRAME, n_features: int = N_FEATURES) -> None:
        """
        This Data class is implemented directly for NFStream use-cases.

        :param df: dataframe input from NFStream
        """
        self.__data = pd.read_csv(df)

        nfstream_data = self.__data

        self.__data_reduced = self.__data.drop(DROP_VARIABLES, axis='columns')
        nfstream_data_reduced = self.__data_reduced

        for regex_filter in REDUCE_REGEX_VARIABLES:
            nfstream_data_reduced = nfstream_data_reduced.drop(
                list(nfstream_data_reduced.filter(regex=regex_filter)), axis='columns'
            )

        # Give meaning to the values of the protocol feature
        nfstream_data_reduced['protocol'] = nfstream_data_reduced['protocol'].map({6: 'TCP', 17: 'UDP'})
        # Give meaning to the values of the expiration_id feature
        nfstream_data_reduced['expiration_id'] = nfstream_data_reduced['expiration_id'].map({0: 'idle', 1: 'active'})

        # Drop non TCP or UDP traffic
        nfstream_data_reduced = nfstream_data_reduced.dropna()
        nfstream_data = nfstream_data[nfstream_data['protocol'] != 1]

        nfstream_w_outliers = nfstream_data_reduced.copy()
        nfstream_w_outliers['protocol'] = nfstream_data['protocol']  # Remove the legible encoding
        nfstream_w_outliers['expiration_id'] = nfstream_data['expiration_id']  # Remove the legible encoding

        target = nfstream_w_outliers['application_name']
        nfstream_wo_outliers = nfstream_w_outliers.drop(['application_name'], axis='columns')

        def selector(score_function, X, y, n_features: int) -> pd.DataFrame:
            """
            Selects the K best element.

            :param score_function: function to calculate out the score
            :param X: input fields
            :param y: prediction field
            :param n_features: number of features
            :return:
            """
            selector = SelectKBest(score_function, k=n_features)
            selector.fit_transform(X, y)
            cols = selector.get_support(indices=True)
            return X.iloc[:, cols]

        X_f = selector(f_classif, nfstream_wo_outliers, target, n_features)
        # scale
        nfstream_scaled = MinMaxScaler().fit_transform(X_f)
        nfstream_scaled = pd.DataFrame(nfstream_scaled,
                                            columns=X_f.columns,
                                            index=X_f.index)

        protocol_df = X_f['protocol'].map({6: 'TCP', 17: 'UDP'})
        protocol_encoded = pd.get_dummies(protocol_df, prefix='protocol')

        # Drop the non-encoded feature
        nfstream_scaled_minmax_encoded = nfstream_scaled.drop(['protocol'], axis='columns')

        # Add to scaled DataFrames
        nfstream_scaled_minmax_encoded = nfstream_scaled_minmax_encoded.join(protocol_encoded)

        self.__data = nfstream_scaled_minmax_encoded.join(target)

    def load_data(self, target_field: str = TARGET_VARIABLE) -> Tuple[Any, Any, Any, Any]:
        """
        Load train and validation test data split.

        :param target_field: y field
        :return: train and validation split
        """
        train, validation = train_test_split(self.__data, test_size=TRAIN_VALIDATION_SPLIT)

        def __separate_target(df):
            return df.loc[:, ~df.columns.isin([target_field])], df[target_field]

        x_train, y_train = __separate_target(train)
        x_val, y_val = __separate_target(validation)

        return x_train, y_train, x_val, y_val


data = Data()
