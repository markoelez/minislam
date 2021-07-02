#!/usr/bin/env python3
import cv2
import sys
import numpy as np
np.set_printoptions(suppress=True)
from dataset import ImageLoader, VideoLoader
from display import Display, Display3D
from camera import Camera
from visual_odometry import VisualOdometry


window_name = 'window'


# transform point to display nicely in frame
def fit_point(img_shape, pt):
    w, h = img_shape
    x, y, z = pt

    return (int(x + w // 2), int(h + z))


if __name__ == '__main__':
    
    fxy = 270 # focal length: fx = fy
    W = 1920 // 2
    H = 1080 // 2

    camera = Camera(W, H, fxy, fxy, W // 2, H // 2)

    loader = VideoLoader('videos/kitti1.mp4')
    #loader = ImageLoader('videos/one')

    display2D = Display(W, H)

    vo = VisualOdometry(camera)
    
    tw, th = 800, 800
    mapd = np.zeros((tw, th, 3))
    
    for i, img in enumerate(loader):
        img = cv2.resize(img, (W, H))

        vo.process_frame(img, i)
        
        display2D.paint(vo.draw_img)
        
        if i > 1:
            x, y = fit_point((tw, th), vo.translations[-1])

            cv2.circle(mapd, (x, y), 3, (0, 255, 0), 1)

            mapd = mapd.copy()
            cv2.putText(mapd, f'x: {x} y: {y}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
            cv2.imshow('Trajectory', mapd)
        
    cv2.destroyAllWindows()
