import cv2
import numpy as np

from minislam.util import pose_Rt
from minislam.features import FeatureManager
from minislam.loop_closure import LoopClosureDetector, LoopClosureCandidate


class VisualOdometry:
  def __init__(self, camera, enable_loop_closure: bool = True):
    self.camera = camera
    self.feature_manager = FeatureManager()

    self.scale = 0.8

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
    self.poses = []

    self.cur_R = np.eye(3, 3)
    self.cur_t = np.zeros((3, 1))

    # Loop closure detection with smart keyframe selection
    self.enable_loop_closure = enable_loop_closure
    self.loop_closure_detector = LoopClosureDetector(
      min_frame_gap=50,
      similarity_threshold=0.75,
      min_inliers=50,
      min_inlier_ratio=0.3,
      # Smart keyframe selection parameters
      min_keyframe_gap=5,
      min_parallax=15.0,  # pixels - minimum feature displacement
      max_parallax=100.0,  # pixels - force keyframe if exceeded
      min_tracked_ratio=0.5,  # force keyframe if tracking drops below 50%
      min_features=50,  # force keyframe if features drop below this
    )
    self.last_loop_closure: LoopClosureCandidate | None = None

  def process_frame(self, img, frame_id):
    # setup
    if img.ndim > 2:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    self.cur_img = img

    # process
    self.cur_kps, self.cur_des = self.feature_manager.detect_and_compute(img)
    if frame_id == 0:
      self.draw_img = cv2.cvtColor(self.cur_img, cv2.COLOR_GRAY2RGB)
    else:
      self.cur_matched_kps, self.ref_matched_kps = self.feature_manager.get_matches(self.cur_kps, self.ref_kps, self.cur_des, self.ref_des)

      R, t, mask = self.estimate_pose()

      self.cur_matched_kps = self.cur_matched_kps[mask]
      self.ref_matched_kps = self.ref_matched_kps[mask]

      self.draw_img = self.draw_features(self.cur_img)

      self.cur_t = self.cur_t + (self.scale * self.cur_R.dot(t))
      self.cur_R = self.cur_R.dot(R)

      self.translations.append(self.cur_t)
      pose = pose_Rt(self.cur_R, self.cur_t)
      self.poses.append(pose)

      # Loop closure detection
      if self.enable_loop_closure:
        self.last_loop_closure = self.loop_closure_detector.process_frame(
          frame_id=frame_id,
          pose=pose,
          keypoints=self.cur_kps,
          descriptors=self.cur_des,
        )

        if self.last_loop_closure is not None:
          print(
            f"[Loop Closure] Detected! Frame {self.last_loop_closure.query_frame_id} "
            f"matches frame {self.last_loop_closure.match_frame_id} "
            f"({self.last_loop_closure.num_inliers} inliers)"
          )

    # update reference
    self.ref_img = self.cur_img
    self.ref_kps = self.cur_kps
    self.ref_des = self.cur_des

  def estimate_pose(self, use_fundamental_matrix=False):
    if use_fundamental_matrix:
      cur_kps = self.cur_matched_kps
      ref_kps = self.ref_matched_kps
      F, mask = cv2.findFundamentalMat(cur_kps, ref_kps, method=cv2.RANSAC)  # type: ignore
      E = np.dot(self.camera.K.T, F).dot(self.camera.K)
    else:
      cur_kps = self.camera.denormalize_pts(self.cur_matched_kps)
      ref_kps = self.camera.denormalize_pts(self.ref_matched_kps)
      E, mask = cv2.findEssentialMat(
        cur_kps,
        ref_kps,
        focal=1,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.003,
      )

    _, R, t, mask = cv2.recoverPose(E, cur_kps, ref_kps, focal=1, pp=(0.0, 0.0))  # type: ignore

    inlier_indices = np.where(mask.ravel() > 0)[0]
    return R, t, inlier_indices

  def draw_features(self, img):
    draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for p1, p2 in zip(self.ref_matched_kps, self.cur_matched_kps):  # type: ignore
      x1, y1 = map(int, p1)
      x2, y2 = map(int, p2)
      cv2.circle(draw_img, (x1, y1), 1, (255, 0, 0), 1)
      cv2.circle(draw_img, (x2, y2), 1, (255, 0, 0), 1)
      cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw loop closure indicator
    if self.last_loop_closure is not None:
      cv2.putText(
        draw_img,
        f"LOOP CLOSURE: {self.last_loop_closure.match_frame_id}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
      )

    return draw_img

  @property
  def loop_closures(self) -> list[tuple[int, int]]:
    """Get all detected loop closure pairs."""
    return self.loop_closure_detector.get_loop_closure_pairs()

  @property
  def keyframes(self):
    """Get all stored keyframes."""
    return self.loop_closure_detector.keyframes

  @property
  def num_keyframes(self) -> int:
    """Get the number of keyframes."""
    return self.loop_closure_detector.num_keyframes
