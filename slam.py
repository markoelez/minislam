#!/usr/bin/env python3
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
from dataset import ImageLoader


window_name = 'window'

class Frame:

    def __init__(self, img):
        self.original_img = img
        self.img = img

    def detect_keypoints(self, n_features=5000):
        orb = cv2.ORB_create(nfeatures=n_features, scoreType=cv2.ORB_FAST_SCORE)
        kp = orb.detect(self.img, None)
        kp, des = orb.compute(self.img, kp)

        #img = cv2.drawKeypoints(self.img, kp, None, color=(0, 255, 0), flags=0)

        self.kp, self.des = np.array(kp), np.array(des)

        return (kp, des)


def paint(img):
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 90, 180)
    cv2.imshow(window_name, img)
    cv2.waitKey(50)

def paint_matches(f1, f2, matches):

    f1_mask = np.array([x.queryIdx for x in matches])
    f2_mask = np.array([x.trainIdx for x in matches])
    
    pts1 = np.array([x.pt for x in f1.kp[f1_mask]])
    pts2 = np.array([x.pt for x in f2.kp[f2_mask]])

    # ransac
    model, inliers = ransac((pts1, pts2),
                            AffineTransform,
                            min_samples=4,
                            residual_threshold=8,
                            max_trials=1000
                            )

    pts1 = pts1[inliers]
    pts2 = pts2[inliers]

    for p1, p2 in zip(pts1, pts2):
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        cv2.circle(f2.img, (x1, y1), 2, (0, 0, 255), 1)
        cv2.circle(f2.img, (x2, y2), 2, (255, 0, 0), 1)
        cv2.line(f2.img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return f2.img

def match_keypoints(f1, f2):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good

if __name__ == '__main__':

    loader = ImageLoader('videos/one/')

    prev = None
    for x in loader:
        f = Frame(x)
        f.detect_keypoints(n_features=50000)

        if prev:
            matches = match_keypoints(prev, f)
            img = paint_matches(prev, f, matches)
            paint(img)
        prev = f

    cv2.destroyAllWindows()
