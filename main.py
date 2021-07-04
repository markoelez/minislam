#!/usr/bin/env python3
import cv2
import sys
import os
import configparser
import argparse
import numpy as np
np.set_printoptions(suppress=True)
from dataset import ImageLoader, VideoLoader
from display2d import Display as Display2D
from display3d import Display as Display3D
from camera import Camera
from visual_odometry import VisualOdometry


CONFIG_PATH = 'config.ini'

# transform point to display nicely in frame
def fit_point(img_shape, pt):
    w, h = img_shape
    x, y, z = pt
    return (int(x + w // 2), int(h - (h // 4) + z))


def main(camera, data_loader):
    width, height = camera.width, camera.height

    display2D = Display2D(width, height)
    display3D = Display3D()

    vo = VisualOdometry(camera)
    
    tw, th = 800, 800
    mapd = np.zeros((tw, th, 3))

    draw = True
    
    for i, img in enumerate(data_loader):
        img = cv2.resize(img, (width, height))

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

if __name__ == '__main__':
    config = configparser.ConfigParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_section',
                        metavar='config_section',
                        type=str,
                        help='The associated config section in the config.ini file')
    args = vars(parser.parse_args())
    section = args['config_section']

    config.read(CONFIG_PATH)

    params = config[section]

    # focal length
    fx, fy = float(params['fx']), float(params['fy'])

    # principle point
    cx, cy = float(params['cx']), float(params['cy'])

    # camera dimensions
    w, h = int(params['w']), int(params['h'])

    camera = Camera(w, h, fx, fy, cx, cy)
    
    path = params['path']
    if os.path.isdir(path):
        loader = ImageLoader(path)
    elif os.path.isfile(path):
        loader = VideoLoader(path)

    assert loader

    main(camera, loader)

