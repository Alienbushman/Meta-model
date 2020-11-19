# Meta-model

# How to run
Start a jupyter notebook server

Generate data using the notebook 'Creating_toy_datasets_friedman.ipynb'
Run the underlying models for the regression algorithm using the notebook 'Run_models_regression.ipynb'
To run the meta-model run the jupyter notebook 'Run_meta_model.ipynb'


You can create a new example by creating datasets using the jypyter notebook 'Creating_toy_datasets_friedman'
and changing the variables [n_samples,random_state, amount_datasets, noise, window_size, shift_dataset]

To run the different datasets, open the notebook 'Run_models_regression' and specify which datasets it should run by editing the variable 'dataset_name', sending different parameters to the 'dataset_name_generator'

You can output different results of the meta-model by altering the 'Run_meta_model' notebook (which includes a feature importance tool as well as the ability to visualize the results)


