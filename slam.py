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


if __name__ == '__main__':
    
    fxy = 270 # focal length: fx = fy
    W = 1920 // 2
    H = 1080 // 2

    camera = Camera(W, H, fxy, fxy, W // 2, H // 2)

    loader = VideoLoader('videos/test.mp4')
    #loader = ImageLoader('videos/one')

    display2D = Display(W, H)

    vo = VisualOdometry(camera)
    
    for i, img in enumerate(loader):
        img = cv2.resize(img, (W, H))

        vo.process_frame(img, i)
        
        #print('-' * 50)
        #print(img.shape)
        #print(vo.draw_img.shape)
        #print(vo.cur_img.shape)
        #print('-' * 50)
        display2D.paint(vo.draw_img)
        
    cv2.destroyAllWindows()
