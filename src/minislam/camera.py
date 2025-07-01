import numpy as np

from minislam.util import add_ones


class Camera:
  def __init__(self, width, height, fx, fy, cx, cy):
    self.width, self.height = width, height
    self.fx, self.fy = fx, fy
    self.cx, self.cy = cx, cy
    self.pp = (cx, cy)

    self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    self.Kinv = np.linalg.inv(self.K)

  def __str__(self):
    return np.array2string(self.K)

  # 3D world --> 2D image
  def normalize_pts(self, pts):
    raise NotImplementedError()

  # 2D image --> 3D world
  def denormalize_pts(self, pts):
    h = add_ones(pts)
    return np.dot(self.Kinv, h.T).T[:, 0:2]
