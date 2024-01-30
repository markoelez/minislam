import cv2
import numpy as np


class FeatureManager:
    def __init__(self):
        self.feature_extractor = cv2.ORB_create()

    def detect(self, frame):
        self.feature_extractor = cv2.ORB_create()
        pts = cv2.goodFeaturesToTrack(frame, 2000, qualityLevel=0.01, minDistance=3)
        return [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=5) for p in pts]

    def compute(self, frame, kps):
        kps, des = self.feature_extractor.compute(frame, kps)
        self.kps, self.des = np.array(kps), np.array(des)
        return (self.kps, self.des)

    def detect_and_compute(self, frame):
        kps = self.detect(frame)
        return self.compute(frame, kps)

    def get_matches(self, cur_kps, ref_kps, cur_des, ref_des):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(ref_des, cur_des, k=2)

        # apply ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        assert len(good) > 8

        # return matching keypoints
        ref_mask = np.array([x.queryIdx for x in good])
        cur_mask = np.array([x.trainIdx for x in good])

        ref_pts = np.array([x.pt for x in ref_kps[ref_mask]])
        cur_pts = np.array([x.pt for x in cur_kps[cur_mask]])

        return (ref_pts, cur_pts)
