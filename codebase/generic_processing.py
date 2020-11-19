"""A module to store common methods used across sections"""


def dataset_name_generator(noise=0, window_size=2, shift_dataset=0, n_samples=10000, number_datasets=10):
    """A method for consistently  generating names given the same parameters"""
    dataset_name = 'friedman_' + str(number_datasets)
    if (noise > 0):
        dataset_name += '_noise_' + str(noise)
    if (window_size > 0):
        dataset_name += '_window_size_' + str(window_size)
    if (shift_dataset > 0):
        dataset_name += '_shift_dataset_' + str(shift_dataset)
    if (n_samples != 10000):
        dataset_name += '_samples_' + str(n_samples)
    return dataset_name
