#!/usr/bin/env python3
import cv2
import sys
import numpy as np
np.set_printoptions(suppress=True)
from dataset import ImageLoader, VideoLoader
from display2d import Display as Display2D
from display3d import Display as Display3D
from camera import Camera
from visual_odometry import VisualOdometry


window_name = 'window'


# transform point to display nicely in frame
def fit_point(img_shape, pt):
    w, h = img_shape
    x, y, z = pt

    return (int(x + w // 2), int(h - (h // 4) + z))


if __name__ == '__main__':
    
    fxy = 718.856 # focal length: fx = fy
    cx, cy = 607.2,  185.2

    W = 1241
    H = 376

    camera = Camera(W, H, fxy, fxy, W // 2, H // 2)

    #loader = VideoLoader('videos/kitti1.mp4')
    #loader = VideoLoader('videos/kitti_datasets/kitti00/video.mp4')
    loader = VideoLoader('videos/kitti_datasets/kitti06/video.mp4')

    display2D = Display2D(W, H)
    display3D = Display3D()

    vo = VisualOdometry(camera)
    
    tw, th = 800, 800
    mapd = np.zeros((tw, th, 3))

    draw = True
    
    for i, img in enumerate(loader):
        img = cv2.resize(img, (W, H))

        vo.process_frame(img, i)
        
        if draw:
            display2D.paint(vo.draw_img)
        
        if i > 1:
            x, y = fit_point((tw, th), vo.translations[-1])
            
            if draw:
                cv2.circle(mapd, (x, y), 2, (0, 255, 0), 1)

                tmp = mapd.copy()
                cv2.putText(tmp, f'x: {x} y: {y}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, 8)
            
                cv2.imshow('Trajectory', tmp)
                
                display3D.update(vo)
        
    if draw:
        cv2.destroyAllWindows()
