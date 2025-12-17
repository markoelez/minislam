from dataclasses import field, dataclass

import cv2
import numpy as np


@dataclass
class Keyframe:
  """Represents a keyframe with its associated data."""

  frame_id: int
  pose: np.ndarray
  keypoints: np.ndarray
  descriptors: np.ndarray
  descriptor_mean: np.ndarray = field(default_factory=lambda: np.array([]))

  def __post_init__(self):
    # Compute mean descriptor for fast similarity comparison
    if len(self.descriptors) > 0:
      self.descriptor_mean = np.mean(self.descriptors.astype(np.float32), axis=0)


@dataclass
class LoopClosureCandidate:
  """Represents a detected loop closure."""

  query_frame_id: int
  match_frame_id: int
  query_pose: np.ndarray
  match_pose: np.ndarray
  num_inliers: int
  relative_pose: np.ndarray


class LoopClosureDetector:
  """Detects loop closures using descriptor matching and geometric verification."""

  def __init__(
    self,
    min_frame_gap: int = 50,
    similarity_threshold: float = 0.75,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.3,
    keyframe_interval: int = 5,
  ):
    """
    Initialize loop closure detector.

    Args:
      min_frame_gap: Minimum frames between current and candidate for loop closure
      similarity_threshold: Minimum descriptor similarity score (0-1)
      min_inliers: Minimum inlier matches for valid loop closure
      min_inlier_ratio: Minimum ratio of inliers to total matches
      keyframe_interval: Add keyframe every N frames
    """
    self.min_frame_gap = min_frame_gap
    self.similarity_threshold = similarity_threshold
    self.min_inliers = min_inliers
    self.min_inlier_ratio = min_inlier_ratio
    self.keyframe_interval = keyframe_interval

    self.keyframes: list[Keyframe] = []
    self.loop_closures: list[LoopClosureCandidate] = []

    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

  def should_add_keyframe(self, frame_id: int) -> bool:
    """Determine if this frame should be stored as a keyframe."""
    return frame_id % self.keyframe_interval == 0

  def add_keyframe(self, frame_id: int, pose: np.ndarray, keypoints: np.ndarray, descriptors: np.ndarray) -> None:
    """Add a new keyframe to the database."""
    kf = Keyframe(frame_id=frame_id, pose=pose.copy(), keypoints=keypoints.copy(), descriptors=descriptors.copy())
    self.keyframes.append(kf)

  def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
    """
    Compute similarity between two descriptor sets using mean descriptor correlation.
    Returns value between 0 (different) and 1 (identical).
    """
    if len(desc1) == 0 or len(desc2) == 0:
      return 0.0

    mean1 = np.mean(desc1.astype(np.float32), axis=0)
    mean2 = np.mean(desc2.astype(np.float32), axis=0)

    # Normalize and compute correlation
    norm1 = np.linalg.norm(mean1)
    norm2 = np.linalg.norm(mean2)

    if norm1 == 0 or norm2 == 0:
      return 0.0

    correlation = np.dot(mean1, mean2) / (norm1 * norm2)
    # Convert from [-1, 1] to [0, 1]
    return float((correlation + 1) / 2)

  def find_candidates(self, frame_id: int, descriptors: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    """
    Find potential loop closure candidates based on descriptor similarity.

    Returns:
      List of (keyframe_index, similarity_score) tuples, sorted by similarity
    """
    candidates = []

    for i, kf in enumerate(self.keyframes):
      # Skip recent frames
      if frame_id - kf.frame_id < self.min_frame_gap:
        continue

      similarity = self.compute_similarity(descriptors, kf.descriptors)

      if similarity >= self.similarity_threshold:
        candidates.append((i, similarity))

    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]

  def verify_loop_closure(
    self,
    query_kps: np.ndarray,
    query_des: np.ndarray,
    match_kf: Keyframe,
  ) -> tuple[bool, int, np.ndarray | None]:
    """
    Geometrically verify a loop closure candidate using RANSAC.

    Returns:
      (is_valid, num_inliers, relative_pose or None)
    """
    if len(query_des) < 8 or len(match_kf.descriptors) < 8:
      return False, 0, None

    # Match descriptors
    try:
      matches = self.matcher.knnMatch(query_des, match_kf.descriptors, k=2)
    except cv2.error:
      return False, 0, None

    # Apply stricter ratio test (0.7 instead of 0.75)
    good_matches = []
    for match_pair in matches:
      if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.7 * n.distance:
          good_matches.append(m)

    if len(good_matches) < self.min_inliers:
      return False, len(good_matches), None

    # Extract matched points
    query_pts = np.array([query_kps[m.queryIdx].pt for m in good_matches])
    match_pts = np.array([match_kf.keypoints[m.trainIdx].pt for m in good_matches])

    # Compute fundamental matrix with RANSAC (stricter threshold)
    F, mask = cv2.findFundamentalMat(query_pts, match_pts, cv2.RANSAC, 2.0, 0.99)

    if F is None or mask is None:
      return False, 0, None

    num_inliers = int(np.sum(mask))

    # Check minimum inliers
    if num_inliers < self.min_inliers:
      return False, num_inliers, None

    # Check inlier ratio - ensures geometric consistency
    inlier_ratio = num_inliers / len(good_matches)
    if inlier_ratio < self.min_inlier_ratio:
      return False, num_inliers, None

    # Create relative pose placeholder (would need camera intrinsics for full pose)
    relative_pose = np.eye(4)

    return True, num_inliers, relative_pose

  def detect(
    self,
    frame_id: int,
    pose: np.ndarray,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
  ) -> LoopClosureCandidate | None:
    """
    Attempt to detect a loop closure for the current frame.

    Args:
      frame_id: Current frame ID
      pose: Current camera pose (4x4 matrix)
      keypoints: Current frame keypoints
      descriptors: Current frame descriptors

    Returns:
      LoopClosureCandidate if loop closure detected, None otherwise
    """
    # Find candidate keyframes
    candidates = self.find_candidates(frame_id, descriptors)

    for kf_idx, similarity in candidates:
      kf = self.keyframes[kf_idx]

      # Verify geometrically
      is_valid, num_inliers, relative_pose = self.verify_loop_closure(keypoints, descriptors, kf)

      if is_valid and relative_pose is not None:
        loop_closure = LoopClosureCandidate(
          query_frame_id=frame_id,
          match_frame_id=kf.frame_id,
          query_pose=pose.copy(),
          match_pose=kf.pose.copy(),
          num_inliers=num_inliers,
          relative_pose=relative_pose,
        )
        self.loop_closures.append(loop_closure)
        return loop_closure

    return None

  def process_frame(
    self,
    frame_id: int,
    pose: np.ndarray,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
  ) -> LoopClosureCandidate | None:
    """
    Process a frame for loop closure detection.
    Adds keyframes as needed and checks for loop closures.

    Args:
      frame_id: Current frame ID
      pose: Current camera pose (4x4 matrix)
      keypoints: Current frame keypoints
      descriptors: Current frame descriptors

    Returns:
      LoopClosureCandidate if loop closure detected, None otherwise
    """
    loop_closure = None

    # Check for loop closure (only if we have enough keyframes)
    if len(self.keyframes) > self.min_frame_gap // self.keyframe_interval:
      loop_closure = self.detect(frame_id, pose, keypoints, descriptors)

    # Add keyframe if needed
    if self.should_add_keyframe(frame_id):
      self.add_keyframe(frame_id, pose, keypoints, descriptors)

    return loop_closure

  def get_loop_closure_pairs(self) -> list[tuple[int, int]]:
    """Get list of (frame_id, frame_id) pairs for all detected loop closures."""
    return [(lc.query_frame_id, lc.match_frame_id) for lc in self.loop_closures]
