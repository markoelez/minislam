import numpy as np


class Camera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

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
        pass
