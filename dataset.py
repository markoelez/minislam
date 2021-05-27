#!/usr/bin/env python3
import os
import cv2
from collections import deque


class ImageLoader:

    def __init__(self, base_dir):
        assert os.path.exists(base_dir)
        self.base_dir = base_dir

        self.images = deque(sorted([x for x in os.listdir(base_dir) if x.endswith('.png')]))

    def __next__(self):
        try:
            return cv2.imread(os.path.join(self.base_dir, self.images.popleft()))
        except IndexError as e:
            raise StopIteration from e

    def __iter__(self):
        return self


if __name__ == '__main__':

    loader = ImageLoader('videos/one/')

    for x in loader:
        cv2.namedWindow('window')
        cv2.moveWindow('window', 90, 180)
        cv2.imshow('window', x)
        cv2.waitKey(200)

    cv2.destroyAllWindows()
