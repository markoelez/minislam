#!/usr/bin/env python3
import os
import argparse

import cv2
import yaml
import numpy as np

from minislam.camera import Camera
from minislam.dataset import DataLoader, ImageLoader, VideoLoader
from minislam.odometry import VisualOdometry
from minislam.display2d import Display as Display2D
from minislam.display3d import Display as Display3D

np.set_printoptions(suppress=True)


def fit_point(img_shape: tuple[float, float], pt: tuple[float, float, float]) -> tuple[float, float]:
  w, h = img_shape
  x, _, z = pt
  return (int(x + w // 2), int(h - (h // 4) + z))


def parse_cfg(path: str, dataset: str) -> dict:
  with open(path, "r") as fp:
    cfg = yaml.safe_load(fp)
  assert dataset in cfg["datasets"]
  return cfg["datasets"][dataset]


def _main(camera: Camera, data_loader: DataLoader):
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
        cv2.circle(mapd, (x, y), 2, (0, 255, 0), 1)  # type: ignore

        tmp = mapd.copy()
        cv2.putText(tmp, f"x: {x} y: {y}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, 8)

        cv2.imshow("Trajectory", tmp)

        display3D.update(vo)

  if draw:
    cv2.destroyAllWindows()


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
