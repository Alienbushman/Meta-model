import os
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
    mutual_info_score
)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb


def metrics_regression(y_test, y_pred):
    """
    Prints the standard of the metrics
    :param y_test: the true labels of the test set
    :param y_pred: the predicted labels of the test set
    :return: None, this prints out the results of the metrics
    """
    r2 = r2_score(y_test, y_pred)
    MAE = median_absolute_error(y_test, y_pred)
    RSE = mean_squared_error(y_test, y_pred)
    sum_preds = y_pred.sum()
    sum_actual = y_test.sum()
    return r2, MAE, RSE, sum_preds, sum_actual


def run_datasets(df_train, df_target, target, apply_preprocessing=False, reporter_object=None):
    reporter_object.normalized = apply_preprocessing

    if apply_preprocessing:
        df_train = normalize_data_exclude_target(df_train, target)
        df_target = normalize_data_exclude_target(df_target, target)
    X_train, y_train = label_feature_split(df_train, target)
    X_test, y_test = label_feature_split(df_target, target)
    reporter_object.actual_sum = y_test.sum()

    return run_generic_models_regression(X_train, y_train, X_test, y_test, reporter_object)


def split_into_bins(df, bins=12, column=None):
    """
    This method adds which bin the column falls into based on the column and bins
    :param df: dataframe to be used
    :param column: the column which needs to be split
    :param bins: a list of the bins the dataframe is split into
    :return: the original dataframe with the new columns
    """
    list_of_persentiles = []
    for i in range(1, bins):
        list_of_persentiles.append(i / bins)
    if column is None:
        dataset = pd.DataFrame({'predictions': df})
        percentile_outputs = list(dataset.predictions.describe(percentiles=list_of_persentiles))
    else:
        percentile_outputs = list(df[column].describe(percentiles=list_of_persentiles))

    count = percentile_outputs.pop(0)
    mean = percentile_outputs.pop(0)
    std = percentile_outputs.pop(0)
    return count, mean, std, percentile_outputs


def run_generic_models_regression(X_train, y_train, X_test, y_test, reporter_object):
    # models from https://arxiv.org/abs/1708.05070 slightly adaped for regression and speeds
    GBC = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    RFC = RandomForestRegressor(n_estimators=500, max_features=0.25)
    SVM = SVR(C=0.01, gamma=0.1, kernel="poly", degree=3, coef0=10.0)
    ETC = ExtraTreesRegressor(n_estimators=1000, max_features="log2")
    LR = LogisticRegression(C=1.5, penalty="l1", fit_intercept=True)
    # Models that were not included in the paper not from SKlearn
    XGC = XGBRegressor()
    CBC = CatBoostRegressor(silent=True, task_type="GPU")
    light_gb = lgb.LGBMRegressor()
    # Commenting out the later models variable will run all the variables

    models = [(ETC, "Extra tree classifier"), (RFC, "random forest classifier"), (GBC, "gradient boosted classifier"),
              (XGC, "XGBoost"), (light_gb, "Light GBM")]
    models = [(RFC, "random forest regressor")]
    for model, name in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2, MAE, RSE, sum_preds, sum_actual = metrics_regression(y_test, y_pred)
        count, mean, std, persentile_outputs = split_into_bins(y_pred)
        reporter_object.add_model(name, r2, MAE, RSE, sum_preds, count, mean, std, persentile_outputs)
    return reporter_object


def distance_metrics(df1, df2, column):
    series_1, series_2 = basic_drop_series(df1[column], df2[column])

    series_1, series_2 = same_length_lists(series_1, series_2)
    series_1, series_2 = remove_negative(series_1, series_2)

    kl_divergence_result = KL_divergence(series_1, series_2)
    wasserstein_distance_result = calculate_wasserstein_distance(series_1, series_2)
    hellinger_distance_result = hellinger(series_1, series_2)
    ks_test_result = ks_test(series_1, series_2)[1]
    return kl_divergence_result, wasserstein_distance_result, hellinger_distance_result, ks_test_result


class model_results(object):
    def __init__(self, model_name, r2, MAE, MSE, predicted_sum, count, mean, std, bins):
        self.model_name = model_name
        self.r2 = r2
        self.MAE = MAE
        self.MSE = MSE
        self.predicted_sum = predicted_sum
        self.std = std
        self.mean = mean
        self.count = count
        self.bins = bins


class model_data(object):
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.normalized = None
        self.actual_sum = None
        self.models = []
        # per variable
        self.std_train = []
        self.mean_train = []
        self.count_train = []
        self.bins_train = []

        self.std_test = []
        self.mean_test = []
        self.count_test = []
        self.bins_test = []

        self.kl_divergence = []
        self.wasserstein_distance = []
        self.hellinger_distance = []
        self.ks_test = []

    def add_model(self, model_name, r2, MAE, MSE, predicted_sum, count, mean, std, bins):
        self.models.append(model_results(model_name, r2, MAE, MSE, predicted_sum, count, mean, std, bins))

    def print_model_details(self):
        print('Train dataset: {} Test dataset: {}'.format(self.train_dataset, self.test_dataset))
        print('Data normalization is {}'.format(self.normalized))
        print('The target amount is {}'.format(self.actual_sum))
        print('The following is a list of the models used and results obtained')
        print('Format name'.ljust(25) + '\t r2 \t MSE \t\t MAE \t predicted sum')
        for model in self.models:
            print('{}\t{}\t{}\t{}\t{}'.format(model.model_name.ljust(25), round(model.r2, 4), round(model.MSE, 2),
                                              round(model.MAE, 2), round(model.predicted_sum, 2)))

    def output_to_csv(self, filename, features):

        new_model_outputs = []
        for model in self.models:
            dictionary = {'train_dataset': self.train_dataset, 'test_dataset': self.test_dataset,
                          'normalized': self.normalized,
                          'actual_sum': self.actual_sum, 'model_name': model.model_name, 'r2': model.r2,
                          'MSE': model.MSE,
                          'MAE': model.MAE, 'predicted_sum': model.predicted_sum, 'entries_amount': model.count,
                          'standard_deviation': model.std,
                          'mean': model.mean}
            i = 0
            for bin_value in model.bins:
                dictionary['model_bin_number_' + str(i)] = bin_value
                i += 1
            i = 0
            for feature in features:
                dictionary[feature + '_std_train'] = self.std_train[i]
                dictionary[feature + '_mean_train'] = self.mean_train[i]
                dictionary[feature + '_count_train'] = self.count_train[i]

                dictionary[feature + '_std_test'] = self.std_test[i]
                dictionary[feature + '_mean'] = self.mean_test[i]
                dictionary[feature + '_count'] = self.count_test[i]

                dictionary[feature + '_kl_divergence'] = self.kl_divergence[i]
                dictionary[feature + '_wasserstein_distance'] = self.wasserstein_distance[i]
                dictionary[feature + '_hellinger_distance'] = self.hellinger_distance[i]
                dictionary[feature + '_ks_test'] = self.ks_test[i]

                j = 0
                for bin_value in self.bins_train[i]:
                    dictionary[feature + '_bin_number_train_' + str(j)] = bin_value
                    j += 1
                j = 0
                for bin_value in self.bins_test[i]:
                    dictionary[feature + '_bin_number_test_' + str(j)] = bin_value
                    j += 1
                i += 1
            new_model_outputs.append(dictionary)
        df = pd.DataFrame(new_model_outputs)

        df.to_csv(filename, mode='a', header=(not os.path.exists(filename)))


def same_length_lists(l1, l2):
    l1, l2 = sorted_list(l1, l2)
    if len(l1) > len(l2):
        s, l = l2, l1
    elif len(l1) < len(l2):
        s, l = l1, l2
    else:
        return l1, l2

    s_len = len(s)
    l_len = len(l)
    ratio = l_len / s_len
    keep_value = []
    for j in range(s_len):
        get_index = int(round((j + 0.5) * ratio))
        if (get_index >= l_len):
            get_index = l_len - 1
        keep_value.append(l[get_index])
    return s, keep_value


def remove_negative(series_1, series_2):
    min_value = min(min(series_1), min(series_2))
    if min_value <= 0:
        add_value = (abs(min_value) + 1)
        series_1 = [x + add_value for x in series_1]
        series_2 = [x + add_value for x in series_2]
    return series_1, series_2


def KL_divergence(df1, df2):
    return mutual_info_score(df1, df2)


def calculate_wasserstein_distance(series_1, series_2):
    return wasserstein_distance(series_1, series_2)


def hellinger(p, q):
    import math
    return sum([(math.sqrt(t[0]) - math.sqrt(t[1])) * (math.sqrt(t[0]) - math.sqrt(t[1])) \
                for t in zip(p, q)]) / math.sqrt(2.)


def ks_test(series_1, series_2):
    return ks_2samp(series_1, series_2)


def sorted_list(series_1, series_2):
    sorted_1 = sorted(series_1)
    sorted_2 = sorted(series_2)
    return sorted_1, sorted_2


def label_feature_split(df, column):
    label = df[[column]].values.ravel()
    feature = df.drop([column], axis=1)
    return feature, label


def normalize_data_exclude_target(df, target):
    target_df = df[target]
    df = df.drop(target, axis=1)
    columns_encode = list(df.select_dtypes(include=['float', 'int']))
    for col in columns_encode:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df_merged = pd.merge(df, target_df, how='inner', left_index=True, right_index=True)
    return df_merged


def get_name(string):
    return string.rsplit('/', 1)[1][0:-4]


def basic_drop_series(df1, df2):
    return df1.dropna(), df2.dropna()
