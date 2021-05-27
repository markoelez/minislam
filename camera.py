#!/usr/bin/env python3

class Camera:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.width = width
        self.height = height

        self.focal_length = fx

        self.intrinsic = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]])
