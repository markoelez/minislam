import numpy as np


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.f = fx # assume fx == fy
        self.cx = cx
        self.cy = cy
        self.pp = (cx, cy)

        self.K = np.array([[fx, 0,cx],
                           [ 0,fy,cy],
                           [ 0, 0, 1]])

        self.Kinv = np.linalg.inv(self.K)

    def __str__(self):
        return np.array2string(self.K)

    # 3D world --> 2D image
    def normalize_pts(self, pts):
        pass
    
    # 2D image --> 3D world
    def denormalize_pts(self, pts):
        h = add_ones(pts)
        return np.dot(self.Kinv, h.T).T[:, 0:2]

def add_ones(x):
    if len(x.shape) == 1:
        return np.array([x[0], x[1], 1])
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

