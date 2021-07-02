import numpy as np


def add_ones(x):
    if len(x.shape) == 1:
        return np.array([x[0], x[1], 1])
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
