"""A module to run the meta-model on the results of the trained models"""
import re
import numpy
import shap
import pandas as pd
from catboost import CatBoostRegressor


def preprocessing(df, target, column_names=None, bad_columns=None, apply_one_hot=True, using_xgboost=False):
    """A function that applies generic preprocessing to the datasets"""
    target_df = df[target]
    df.drop(target, inplace=True, axis=1)

    df = keep_relevant_columns(df, column_names)
    df = drop_bad_columns(df, bad_columns)
    df = df.dropna()

    if apply_one_hot:
        df = encode_one_hot(df)
        df = df.astype('float64')
        df = df.apply(pd.to_numeric, errors='coerce')
    else:
        df = df.apply(pd.to_numeric, errors='coerce')
    if using_xgboost:
        df = process_column_names_xgboost(df)
    df_merged = pd.merge(df, target_df, how='inner', left_index=True, right_index=True)
    return df_merged


def label_feature_split(df, column):
    """A function that splits the features and the label"""
    label = df[[column]].values.ravel()
    feature = df.drop([column], axis=1)
    return feature, label


def analyse_generic_models_regression(X_train, y_train, X_test, y_test, plot_shap=False,run_gpu=False):
    """A function that trains the meta-model on the results of the different models"""
    if (run_gpu):
        CBC = CatBoostRegressor(silent=True, task_type="GPU")
    else:
        CBC = CatBoostRegressor(silent=True, task_type="CPU")

    CBC.fit(X_train, y_train)
    y_pred = CBC.predict(X_test)
    if plot_shap:
        shap_values = shap.TreeExplainer(CBC).shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
    X_test['actual_result'] = y_test
    X_test['predicted_result'] = y_pred
    return X_test, numpy.corrcoef(y_pred, y_test)[0, 1]


def split_dataset(df, dataset):
    """A function that splits the dataset into the target and training datasets"""
    test_df = df[df["['test_dataset']_" + dataset] == 1]
    train_df = df[df["['test_dataset']_" + dataset] != 1]
    return train_df, test_df


def drop_bad_columns(df, columns=None):
    """A function that removes the columns which should not be included"""
    if columns is not None:
        return df.drop(columns, axis=1)
    return df


def keep_relevant_columns(df, column_names=None):
    """A function for isolating relevant columns if it is required"""
    if column_names is None:
        return df
    return df[column_names]


def encode_one_hot(df):
    """A function for applying one hot encoding to any columns where it is applicable"""
    columns_to_encode = list(df.select_dtypes(include=['category', 'object']))
    for col in columns_to_encode:
        if len(df[col].unique()) < 100:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=[col])], axis=1)
        df.drop(col, inplace=True, axis=1)
    return df


def process_column_names_xgboost(df):
    """A function to process the features to meet XGboost's requirements with regards to which characters can be
    included in naming conventions """
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                  df.columns.values]
    return df
