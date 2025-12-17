#!/usr/bin/env python3
import os
import argparse

import cv2
import yaml
import numpy as np

from minislam.camera import Camera
from minislam.dataset import DataLoader, ImageLoader, VideoLoader
from minislam.display import Display
from minislam.odometry import VisualOdometry
from minislam.display3d import Display3D

np.set_printoptions(suppress=True)


def parse_cfg(path: str, dataset: str) -> dict:
  with open(path, "r") as fp:
    cfg = yaml.safe_load(fp)
  assert dataset in cfg["datasets"]
  return cfg["datasets"][dataset]


def _main(camera: Camera, data_loader: DataLoader):
  width, height = camera.width, camera.height

  # 2D display (OpenCV) - features + top-down trajectory
  display = Display(width, height)

  # 3D display (pygame/OpenGL)
  display3d = Display3D(width=800, height=600)

  vo = VisualOdometry(camera)

  for i, img in enumerate(data_loader):
    img = cv2.resize(img, (width, height))
    vo.process_frame(img, i)

    # Update 2D display
    key = display.show(vo)
    if key == 27:  # ESC to quit
      break

    # Update 3D display
    if not display3d.update(vo):
      break

  cv2.destroyAllWindows()
  display3d.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", metavar="dataset", type=str, help="The target dataset in config.yaml", default="test2")
  args = vars(parser.parse_args())
  cfg = parse_cfg("config.yaml", args["dataset"])

  # focal length
  fx, fy = float(cfg["fx"]), float(cfg["fy"])

  # principle point
  cx, cy = float(cfg["cx"]), float(cfg["cy"])

  # camera dimensions
  w, h = int(cfg["w"]), int(cfg["h"])

  camera = Camera(w, h, fx, fy, cx, cy)

  path = cfg["path"]
  if os.path.isdir(path):
    loader = ImageLoader(path)
  elif os.path.isfile(path):
    loader = VideoLoader(path)
  else:
    raise ValueError(f"Invalid path: {path}")

  _main(camera, loader)
