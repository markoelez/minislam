import numpy as np


def add_ones(x):
    if len(x.shape) == 1: return np.array([x[0], x[1], 1])
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def pose_Rt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t.ravel()
    return ret
