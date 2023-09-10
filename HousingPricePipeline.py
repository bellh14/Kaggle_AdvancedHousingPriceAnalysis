import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class HousingPricePipeline:

    def __init__(self, df, impute_strategy, drop_num_feature_threshold=None, drop_cat_feature_threshold=None,
                 cat_null_strategy=None):
        self.numerical_df = None
        self.cat_df = None
        self.df = df
        self.labels = None
        self.drop_num_feature_threshold = drop_num_feature_threshold
        self.drop_cat_feature_threshold = drop_cat_feature_threshold
        self.impute_strategy = impute_strategy
        self.cat_null_strategy = cat_null_strategy

    def get_categorical_columns(self):
        categories = self.df.select_dtypes(include=[object]).columns
        return categories

    def get_numerical_columns(self):
        numerical = self.df.select_dtypes(include=[np.number]).columns
        return numerical

    def get_features_null_counts(self):
        null_counts = self.df.isnull().sum()
        return null_counts

    def onehot_encode(self, features=None, all_cat=True):
        if all_cat:
            cat_encoder = OneHotEncoder(sparse_output=False)
            # todo
        return self

    def preprocessing_pipeline(self):
        num_pipeline = make_pipeline(SimpleImputer(
            strategy='median'), StandardScaler())
        cat_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown="ignore"))

        preprocessing = make_column_transformer(
            (num_pipeline, make_column_selector(dtype_include=np.number)),
            (cat_pipeline, make_column_selector(dtype_include=object)),
        )
        self.split_labels()
        return preprocessing

    def split_labels(self):
        self.labels = self.df.pop('SalePrice')

    def drop(self):
        null_counts = self.get_features_null_counts()
        num_feature_list = self.get_numerical_columns()
        cat_feature_list = self.get_categorical_columns()
        for num_feature in num_feature_list:
            if self.drop_num_feature_threshold is not None and \
                    null_counts[num_feature] >= self.drop_num_feature_threshold:
                self.drop_feature(num_feature)

        for cat_feature in cat_feature_list:
            if self.drop_cat_feature_threshold is not None and \
                    null_counts[cat_feature] >= self.drop_cat_feature_threshold:
                self.drop_feature(cat_feature)

    def drop_feature(self, feature):
        self.df.drop(feature, axis=1, inplace=True)

    def drop_list_features(self, features):
        for feature in features:
            self.df.drop(feature, axis=1, inplace=True)

    def create_numerical_pipeline(self):
        self.num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ])

    def load_data(self, df):
        train_inputs = df.to_numpy()
        train_results = self.labels.to_numpy()
        training_inputs = [np.reshape(x, (9, 1)) for x in train_inputs]
        training_results = [
            self.vectorize_sale_price(y) for y in train_results]
        # training_results = train_results
        training_data = zip(training_inputs, training_results)
        test_inputs = [np.reshape(x, (9, 1)) for x in test_in]
        # test_results = [vectorized_result(y) for y in test_re]
        test_results = test_re
        test_data = zip(test_inputs, test_results)
        return training_data, test_data

    def vectorize_sale_price(self, j):
        e = np.zeros((755000, 1))
        e[j] = 1.0
        return e
