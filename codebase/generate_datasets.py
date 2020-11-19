"""A module for generating different friedman datasets"""
import numpy as np
import numbers


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def generate_friedmen_ranges(number_datasets=10, n_samples=10000, window_size=2,
                             noise=0.1, shift_dataset=0):
    """The method for generating friedman datasets"""
    datasets_feature = []
    datasets_target = []
    for i in range(number_datasets + 1):
        generator = check_random_state(i)
        X = generator.rand(n_samples, 4)/window_size
        if shift_dataset > 0:
            dataset_range = generate_X_values(X)
        else:
            dataset_range = generate_X_values((X + i / number_datasets * (window_size - 1) / window_size))
        datasets_feature.append(dataset_range)
        datasets_target.append(generate_y(dataset_range, i, n_samples, noise, shift_dataset))
    return datasets_feature, datasets_target


def generate_X_values(X):
    """Generating X values from a list of random variables for friedman datasets"""
    X[:, 0] *= 100
    X[:, 1] *= 520 * np.pi
    X[:, 1] += 40 * np.pi
    X[:, 3] *= 10
    X[:, 3] += 1
    return X


def generate_y(X, i, n_samples, noise, shift_dataset):
    """Generate Y values for the friedman dataset"""
    generator = check_random_state(i)

    y = np.arctan((X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]) + noise * generator.randn(n_samples)

    y += i * shift_dataset
    return y
