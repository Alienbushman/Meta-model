import numpy
import pandas as pd
from scipy.stats import kendalltau
from meta_model import preprocessing, label_feature_split, analyse_generic_models_regression, split_dataset

SELECTED_SECTION = 0.1

print('starting with section ' + str(SELECTED_SECTION))
correlation_coefficient = []
window = SELECTED_SECTION
dataset = 'friedman_10_noise_' + str(SELECTED_SECTION) + '_window_size_2'
directory = 'toy_datasets/ran_datasets/' + dataset + '.csv'
dataset_names = dataset + '_seed_'
numpy.random.seed(42)
dataset = pd.read_csv(directory)
target = 'MAE'
processed_features_df = preprocessing(dataset, target, bad_columns=['MSE', 'r2', 'mean', 'actual_sum'],
                                      using_xgboost=False)

kendalltau_results = []
coefficient_results = []
for i in range(11):
    train_df, validation_df = split_dataset(processed_features_df, dataset_names + str(i))
    X_train, y_train = label_feature_split(train_df, target)
    X_validation, y_validation = label_feature_split(validation_df, target)
    analyse_df, correlation_coefficient = analyse_generic_models_regression(X_train, y_train, X_validation, y_validation)

    analyse_df = analyse_df.sort_values(by=['predicted_result'])
    analyse_df['predicted_rankings'] = analyse_df.reset_index().index.values
    analyse_df = analyse_df.sort_values(by=['actual_result'])
    analyse_df['actual_rankings'] = analyse_df.reset_index().index.values
    coefficient_results.append(correlation_coefficient)
    kendalltau_results.append(kendalltau(analyse_df['predicted_rankings'], analyse_df['actual_rankings'])[0])

print('coeficients')
for coefficient in coefficient_results:
    print(coefficient)
print('klenditau output')
for klenditau_output in kendalltau_results:
    print(klenditau_output)
print('section completed')
