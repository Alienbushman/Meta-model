# Meta-model

# How to run
Start a jupyter notebook

Generate data using the notebook 'creating_toy_datasets_friedman.ipynb'
Run the underlying models for the regression algorithm using the notebook 'run_models_regression.ipynb'
To run the meta-model run the jupyter notebook 'meta_model_regression_analysis.ipynb'


You can create a new example by creating datasets using the file jypyter notebook 'creating_toy_datasets_friedman'
and changing the variables [n_samples,random_state, amount_datasets, noise, window_size, shift_dataset]

To run the different datasets, open the notebook 'run_models_regression' and specify which datasets it should run by editing the variable 'run_dataset'

You can output different results of the meta-model by altering the 'meta_model_regression_analysis' notebook (which includes a feature importance tool as well as the ability to visualize the results)


