import numpy as np


def zero_initialize(sizes):
    """Zero initializes all weights and biases. Note that
    this is not the best practise to follow"""
    weights = [np.zeros(x, y)
               for x, y in zip(sizes[1:], sizes[:-1])]
    biases = [np.zeros(x) for x in sizes[1:]]
    return (weights, biases)


def random_initialize(sizes):
    """initialize randomly using Gaussian distribution with mean 0 and variance 1"""
    weights = [np.random.randn(x, y)
               for x, y in zip(sizes[1:], sizes[:-1])]
    biases = [np.random.randn(x) for x in sizes[1:]]
    return (weights, biases)
