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


@dataclass
class KeyframeDecision:
  """Information about keyframe selection decision."""

  should_add: bool
  reason: str
  parallax: float = 0.0
  tracked_ratio: float = 1.0


class LoopClosureDetector:
  """Detects loop closures using descriptor matching and geometric verification."""

  def __init__(
    self,
    min_frame_gap: int = 50,
    similarity_threshold: float = 0.75,
    min_inliers: int = 50,
    min_inlier_ratio: float = 0.3,
    # Keyframe selection parameters
    min_keyframe_gap: int = 5,
    min_parallax: float = 15.0,
    max_parallax: float = 100.0,
    min_tracked_ratio: float = 0.5,
    min_features: int = 50,
  ):
    """
    Initialize loop closure detector.

    Args:
      min_frame_gap: Minimum frames between current and candidate for loop closure
      similarity_threshold: Minimum descriptor similarity score (0-1)
      min_inliers: Minimum inlier matches for valid loop closure
      min_inlier_ratio: Minimum ratio of inliers to total matches
      min_keyframe_gap: Minimum frames between keyframes
      min_parallax: Minimum median feature displacement (pixels) for new keyframe
      max_parallax: Maximum parallax before forcing keyframe (prevents drift)
      min_tracked_ratio: Minimum ratio of tracked features before forcing keyframe
      min_features: Minimum features before forcing keyframe
    """
    self.min_frame_gap = min_frame_gap
    self.similarity_threshold = similarity_threshold
    self.min_inliers = min_inliers
    self.min_inlier_ratio = min_inlier_ratio

    # Keyframe selection
    self.min_keyframe_gap = min_keyframe_gap
    self.min_parallax = min_parallax
    self.max_parallax = max_parallax
    self.min_tracked_ratio = min_tracked_ratio
    self.min_features = min_features

    self.keyframes: list[Keyframe] = []
    self.loop_closures: list[LoopClosureCandidate] = []
    self.last_keyframe_id: int = -1

    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

  def compute_parallax(
    self,
    cur_keypoints: np.ndarray,
    cur_descriptors: np.ndarray,
    ref_keypoints: np.ndarray,
    ref_descriptors: np.ndarray,
  ) -> tuple[float, int, int]:
    """
    Compute parallax (median feature displacement) between current frame and reference.

    Returns:
      (median_parallax, num_matches, num_ref_features)
    """
    if len(cur_descriptors) < 8 or len(ref_descriptors) < 8:
      return 0.0, 0, len(ref_descriptors)

    try:
      matches = self.matcher.knnMatch(cur_descriptors, ref_descriptors, k=2)
    except cv2.error:
      return 0.0, 0, len(ref_descriptors)

    # Apply ratio test
    good_matches = []
    for match_pair in matches:
      if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
          good_matches.append(m)

    if len(good_matches) < 8:
      return 0.0, len(good_matches), len(ref_descriptors)

    # Compute displacements
    displacements = []
    for m in good_matches:
      pt1 = cur_keypoints[m.queryIdx].pt
      pt2 = ref_keypoints[m.trainIdx].pt
      dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
      displacements.append(dist)

    median_parallax = float(np.median(displacements))
    return median_parallax, len(good_matches), len(ref_descriptors)

  def should_add_keyframe(
    self,
    frame_id: int,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
  ) -> KeyframeDecision:
    """
    Determine if this frame should be stored as a keyframe using smart criteria.

    Criteria:
    1. Minimum frame gap since last keyframe
    2. Sufficient parallax (feature displacement)
    3. Tracking quality (not losing too many features)
    """
    # First keyframe
    if not self.keyframes:
      return KeyframeDecision(should_add=True, reason="first_keyframe")

    # Check minimum frame gap
    frames_since_keyframe = frame_id - self.last_keyframe_id
    if frames_since_keyframe < self.min_keyframe_gap:
      return KeyframeDecision(should_add=False, reason="too_soon")

    # Get last keyframe for comparison
    last_kf = self.keyframes[-1]

    # Compute parallax against last keyframe
    parallax, num_matches, num_ref = self.compute_parallax(keypoints, descriptors, last_kf.keypoints, last_kf.descriptors)

    tracked_ratio = num_matches / max(num_ref, 1)

    # Check if we're losing too many features (tracking failure)
    if len(keypoints) < self.min_features:
      return KeyframeDecision(
        should_add=True,
        reason="low_features",
        parallax=parallax,
        tracked_ratio=tracked_ratio,
      )

    # Check tracking ratio
    if tracked_ratio < self.min_tracked_ratio:
      return KeyframeDecision(
        should_add=True,
        reason="low_tracking",
        parallax=parallax,
        tracked_ratio=tracked_ratio,
      )

    # Check if parallax is too high (force keyframe to prevent drift)
    if parallax > self.max_parallax:
      return KeyframeDecision(
        should_add=True,
        reason="high_parallax",
        parallax=parallax,
        tracked_ratio=tracked_ratio,
      )

    # Check if we have sufficient parallax for good triangulation
    if parallax >= self.min_parallax:
      return KeyframeDecision(
        should_add=True,
        reason="sufficient_parallax",
        parallax=parallax,
        tracked_ratio=tracked_ratio,
      )

    return KeyframeDecision(
      should_add=False,
      reason="insufficient_parallax",
      parallax=parallax,
      tracked_ratio=tracked_ratio,
    )

  def add_keyframe(self, frame_id: int, pose: np.ndarray, keypoints: np.ndarray, descriptors: np.ndarray) -> None:
    """Add a new keyframe to the database."""
    kf = Keyframe(frame_id=frame_id, pose=pose.copy(), keypoints=keypoints.copy(), descriptors=descriptors.copy())
    self.keyframes.append(kf)
    self.last_keyframe_id = frame_id

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
    if len(self.keyframes) > self.min_frame_gap // self.min_keyframe_gap:
      loop_closure = self.detect(frame_id, pose, keypoints, descriptors)

    # Smart keyframe selection
    decision = self.should_add_keyframe(frame_id, keypoints, descriptors)
    if decision.should_add:
      self.add_keyframe(frame_id, pose, keypoints, descriptors)

    return loop_closure

  def get_loop_closure_pairs(self) -> list[tuple[int, int]]:
    """Get list of (frame_id, frame_id) pairs for all detected loop closures."""
    return [(lc.query_frame_id, lc.match_frame_id) for lc in self.loop_closures]

  @property
  def num_keyframes(self) -> int:
    """Get the number of keyframes."""
    return len(self.keyframes)
