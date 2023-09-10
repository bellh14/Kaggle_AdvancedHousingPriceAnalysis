# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import HousingPricePipeline as hp
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError


class AdvancedHousingPrices:

    def __init__(self):
        self.folder_name = "house-prices-advanced-regression-techniques/"
        self.training_data = pd.DataFrame(
            pd.read_csv(self.folder_name + "train.csv"))
        self.training_data_copy = pd.DataFrame(
            pd.read_csv(self.folder_name + "train_copy.csv")).sample(frac=0.8, random_state=0)
        self.train_data = self.training_data_copy.copy()
        self.train_data_labels = self.train_data.pop('SalePrice')
        self.training_data_cat = []
        self.train_data_labels = []
        self.training_data_numerical = []
        self.pipeline = None
        self.attributes = ["MSSubClass", 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
                           'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                           '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea', 'WoodDeckSF',
                           'OpenPorchSF', 'EnclosedPorch', 'MoSold', 'YrSold', 'SalePrice', 'Age', 'TotFlrSF']

    def ols_trend_line(self):
        fig = px.scatter(
            self.training_data, x='OverallQual', y='SalePrice',
            opacity=0.75, trendline='ols'
        )
        fig.show()

    def histogram(self):
        fig = px.histogram(self.training_data, x="SalePrice", y='OverallQual', nbins=30, color='SaleCondition',
                           marginal='violin', hover_data=self.training_data.columns, text_auto=True, histfunc='avg')
        fig.show()

    def correlation_matrix(self):
        corr_matrix = self.training_data_copy.corr()
        return corr_matrix

    def changed_correlation_matrix(self):
        self.training_data["Age"] = self.training_data["YrSold"] - \
            self.training_data["YearBuilt"]
        self.training_data["TotFlrSF"] = self.training_data['1stFlrSF'] + \
            self.training_data['2ndFlrSF']
        corr_matrix = self.training_data[self.attributes].corr()
        return corr_matrix

    def scatter_matrix(self):
        scatter_matrix(
            self.training_data[self.attributes], figsize=(38.4, 21.6))
        plt.show()

    def clean_data(self, update):
        # dropping since very few rows actually have data
        self.training_data.drop("Alley", axis=1, inplace=True)  # 91/1460
        self.training_data.drop('PoolQC', axis=1, inplace=True)  # 1/1460
        self.training_data.drop('Fence', axis=1, inplace=True)  # 281/1460
        self.training_data.drop('MiscFeature', axis=1, inplace=True)  # 54/1460
        # 770/1460 might keep since most is due to no fireplace
        self.training_data.drop('FireplaceQu', axis=1, inplace=True)
        self.training_data.drop('MasVnrType', axis=1, inplace=True)  # 588/1460

        imputer = SimpleImputer(strategy='median')
        # iterImputer = IterativeImputer()  # experimental
        numerical_features = self.training_data_copy.select_dtypes(include=[
                                                                   np.number])
        imputer.fit(numerical_features)
        x = imputer.transform(numerical_features)
        self.training_data_numerical = pd.DataFrame(
            x, columns=numerical_features.columns, index=numerical_features.index)
        # print(training_data_transform["LotFrontage"].head(10))
        self.training_data_copy = self.training_data_numerical.join(
            self.training_data_cat)
        print(self.training_data_copy.info())
        # print(self.training_data_numerical)
        # print(self.training_data_cat)

        if update:
            self.update_copy_csv()

    def encode_categories(self):
        cat_encoder = OneHotEncoder(sparse_output=False)
        cat_features = self.training_data_copy.select_dtypes(include=[object])
        self.training_data_cat = pd.DataFrame(cat_encoder.fit_transform(cat_features),
                                              columns=cat_encoder.get_feature_names_out(), index=cat_features.index)

    def update_copy_csv(self):
        self.training_data_copy.to_csv(
            self.folder_name + "train_copy.csv", index=False)

    def linear_regression(self):
        x = self.training_data["1stFlrSF"].to_numpy().reshape(-1, 1)
        y = self.training_data["SalePrice"].to_numpy()
        model = LinearRegression()
        model.fit(x, y)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        fig = px.scatter(self.training_data, x="1stFlrSF",
                         y="SalePrice", opacity=.65)
        fig.add_traces(go.Scatter(x=x_range, y=y_range, name='test'))
        fig.show()

    def run_pipeline(self):
        self.pipeline = hp.HousingPricePipeline(
            self.training_data, 'median', drop_cat_feature_threshold=700)
        self.pipeline.drop()
        return self.pipeline.preprocessing_pipeline()
        # housing_prepared = preprocessing.fit_transform(pipeline.df)

    def linear_model_test(self):
        lin_reg = make_pipeline(self.run_pipeline(), LinearRegression())
        lin_reg.fit(self.pipeline.df, self.pipeline.labels)
        predictions = lin_reg.predict(self.pipeline.df)
        print(predictions[:10].round(0))  # nearest one
        print(self.pipeline.labels.iloc[:10].values)
        lin_rmse = mean_squared_error(
            self.pipeline.labels, predictions, squared=False)
        print(lin_rmse)

    def decision_tree(self):
        tree_reg = make_pipeline(
            self.run_pipeline(), DecisionTreeRegressor(random_state=42))
        tree_reg.fit(self.pipeline.df, self.pipeline.labels)
        predictions = tree_reg.predict(self.pipeline.df)
        tree_rmses = -cross_val_score(tree_reg, self.pipeline.df, self.pipeline.labels,
                                      scoring="neg_root_mean_squared_error", cv=10)
        print(pd.Series(tree_rmses).describe())

    def random_forest(self):
        forest_reg = make_pipeline(self.run_pipeline(), RandomForestRegressor(max_features=0.3,
                                                                              random_state=42, n_jobs=8))
        forest_rmses = -cross_val_score(forest_reg, self.pipeline.df, self.pipeline.labels,
                                        scoring="neg_root_mean_squared_error", cv=10)
        print(pd.Series(forest_rmses).describe())

    def grid_search(self):
        full_pipeline = Pipeline([
            ('preprocessing', self.run_pipeline()),
            ('random_forest', RandomForestRegressor(
                random_state=42, n_jobs=-1, n_estimators=100)),
        ])

        param_grid = [
            {'random_forest__max_features': [1.0, 'sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             'random_forest__min_samples_split': [2, 4, 5, 10, 15, 20, 25, 50, 100, 250, 500, 1000]},
        ]
        grid_search = GridSearchCV(
            full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
        grid_search.fit(self.pipeline.df, self.pipeline.labels)
        print(grid_search.best_params_)
        cv_res = pd.DataFrame(grid_search.cv_results_)
        print(cv_res.sort_values(by='mean_test_score',
              ascending=False, inplace=True).head(10))

    def random_forest_tuning(self):
        max_features = [2, 4, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 1.0, 'sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        min_sample_splits = [2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 50, 100, 250, 500, 1000]
        preprocessing = self.run_pipeline()
        for max_feature in max_features:
            for min_sample_split in min_sample_splits:
                forest_reg = make_pipeline(preprocessing, RandomForestRegressor(
                    max_features=max_feature, min_samples_split=min_sample_split, random_state=14, n_jobs=32, n_estimators=200))
                forest_rmses = -cross_val_score(forest_reg, self.pipeline.df, self.pipeline.labels,
                                                scoring="neg_root_mean_squared_error", cv=10)
                print(pd.Series(forest_rmses).describe())
                print(max_feature, min_sample_split)

    def correlation_visualize(self):
        sp_corr_df = self.training_data_copy.corr(
        )['SalePrice'].sort_values(ascending=False)
        fig = px.bar(sp_corr_df)
        fig.show()

    def normalizer(self):
        # data = self.clean_data(False)
        self.train_data = self.train_data.transpose()
        normalizer = layers.Normalization(axis=-1)
        normalizer = layers.Normalization()
        normalizer.adapt(self.train_data.to_numpy())
        print(normalizer.mean.numpy())
        return normalizer

    def build_dnn_model(self, norm) -> keras.Sequential:
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss=MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(0.001))

        return model

    def dnn_model(self):
        dnn_model = self.build_dnn_model(self.normalizer())
        dnn_model.summary()
        history = dnn_model.fit(
            self.train_data,
            self.train_data_labels,
            validation_split=0.2,
            verbose=0, epochs=100)
        plot_loss(history)
        # test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
        # print(test_results)


if __name__ == "__main__":
    housing_prices = AdvancedHousingPrices()
    # housing_prices.normalizer()
    #housing_prices.dnn_model()
    # housing_prices.histogram()
    # print(housing_prices.training_data)
    # housing_prices.ols_trend_line()
    # corr = housing_prices.correlation_matrix()
    # print(corr['SalePrice'].sort_values(ascending=False))
    # housing_prices.scatter_matrix()
    # housing_prices.linear_regression()
    # corr = housing_prices.changed_correlation_matrix()
    # print(corr['LotFrontage'].sort_values(ascending=False))
    # print(corr)
    # print(housing_prices.training_data.info())
    # housing_prices.clean_data()
    # housing_prices.encode_categories()
    # housing_prices.clean_data(True)
    # null_counts = housing_prices.training_data.isnull().sum()
    # print(null_counts['Id'])
    # housing_prices.linear_model_test()
    # housing_prices.decision_tree()
    housing_prices.random_forest_tuning()

    # print(housing_prices.training_data_copy.corr()["SalePrice"].sort_values(ascending=False))
    # housing_prices.correlation_visualize()
    # housing_prices.grid_search()
