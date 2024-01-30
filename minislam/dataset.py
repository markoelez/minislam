#!/usr/bin/env python3
import os
import cv2
from abc import abstractmethod
from collections import deque


class DataLoader():
    def __init__(self, path: str):
        assert os.path.exists(path)
        self.path = path

    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()


class ImageLoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)
        self.images = deque(sorted([x for x in os.listdir(path) if x.endswith('.png')]))
        assert self.images
        self.height, self.width, _ = cv2.imread(os.path.join(self.path, self.images[-1])).shape

    def __next__(self):
        try:
            return cv2.imread(os.path.join(self.path, self.images.popleft()))
        except IndexError as e:
            raise StopIteration from e

    def __iter__(self):
        return self


class VideoLoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)
        self.cap = cv2.VideoCapture(path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __iter__(self):
        return self
