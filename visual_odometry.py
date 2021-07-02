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

            R, t, mask = self.estimate_pose()

            self.cur_matched_kps = self.cur_matched_kps[mask]
            self.ref_matched_kps = self.ref_matched_kps[mask]

            self.draw_img = self.draw_features(self.cur_img)

            self.cur_t = self.cur_t + self.cur_R.dot(t) 
            self.cur_R = self.cur_R.dot(R)

            self.translations.append(self.cur_t)
            
            '''
            print(self.cur_R)
            print(self.cur_t)
            print('=' * 30)
            '''


        # update reference
        self.ref_img = self.cur_img
        self.ref_kps = self.cur_kps
        self.ref_des = self.cur_des

    def estimate_pose(self, use_fundamental_matrix=False):
        if use_fundamental_matrix:
            cur_kps = self.cur_matched_kps
            ref_kps = self.ref_matched_kps
            F, mask = cv2.findFundamentalMat(cur_kps,
                                             ref_kps,
                                             method=cv2.RANSAC)
            E = np.dot(self.camera.K.T, F).dot(self.camera.K)
        else:
            cur_kps = self.camera.denormalize_pts(self.cur_matched_kps)
            ref_kps = self.camera.denormalize_pts(self.ref_matched_kps)
            E, mask = cv2.findEssentialMat(cur_kps,
                                           ref_kps,
                                           focal=1,
                                           pp=(0., 0.),
                                           method=cv2.RANSAC,
                                           prob=0.999,
                                           threshold=0.0003)

        _, R, t, mask = cv2.recoverPose(E, cur_kps, ref_kps, focal=1, pp=(0., 0.))

        mask = [i for i, v in enumerate(mask) if v > 0]
        return R, t, mask

    def draw_features(self, img): 
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for p1, p2 in zip(self.ref_matched_kps, self.cur_matched_kps):
            x1, y1 = map(int, p1)
            x2, y2 = map(int, p2)
            cv2.circle(draw_img, (x1, y1), 1, (0, 255, 0), 1)
            cv2.circle(draw_img, (x2, y2), 1, (0, 255, 0), 1)
            cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return draw_img
