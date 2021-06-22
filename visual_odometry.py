import cv2
import numpy as np
from feature_manager import FeatureManager


class VisualOdometry:
    def __init__(self, camera):
        self.camera = camera
        self.feature_manager = FeatureManager()

        self.cur_frame_id = None

        self.poses = []

        self.cur_img = None
        self.ref_img = None
        self.draw_img = None

        self.cur_kps = None
        self.ref_kps = None

        self.cur_des = None
        self.ref_des = None

        self.cur_matched_kps = None
        self.ref_matched_kps = None

        self.translations = []

        self.cur_R = np.eye(3, 3)
        self.cur_t = np.zeros((3, 1))

    def process_frame(self, img, frame_id):
        # setup
        cur_frame_id = frame_id
        if img.ndim > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             
        self.cur_img = img
        
        # process
        self.cur_kps, self.cur_des = self.feature_manager.detect_and_compute(img)
        if frame_id == 0:
            self.draw_img = cv2.cvtColor(self.cur_img, cv2.COLOR_GRAY2RGB)
        else:
            self.cur_matched_kps, self.ref_matched_kps = self.feature_manager.get_matches(self.cur_kps,
                                                                                          self.ref_kps,
                                                                                          self.cur_des,
                                                                                          self.ref_des)
            self.draw_img = self.draw_features(self.cur_img)

        # update reference
        self.ref_img = self.cur_img
        self.ref_kps = self.cur_kps
        self.ref_des = self.cur_des

    def estimate_pose(self):
        pass

    def draw_features(self, img): 
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for p1, p2 in zip(self.ref_matched_kps, self.cur_matched_kps):
            x1, y1 = map(int, p1)
            x2, y2 = map(int, p2)
            cv2.circle(draw_img, (x1, y1), 1, (0, 255, 0), 1)
            cv2.circle(draw_img, (x2, y2), 1, (0, 255, 0), 1)
            cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return draw_img
