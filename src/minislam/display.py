"""Unified display module using OpenCV only."""

import cv2
import numpy as np


class Display:
  """Combined display for features, trajectory, and top-down map view."""

  def __init__(self, frame_width: int, frame_height: int, map_size: int = 600, window_name: str = "MiniSLAM"):
    self.frame_width = frame_width
    self.frame_height = frame_height
    self.map_size = map_size
    self.window_name = window_name

    # Create and position window at top-left
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 0, 0)

    # Trajectory map (top-down view) - dark background
    self.trajectory_map = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    self.map_scale = 1.0
    self.map_center = np.array([map_size // 2, map_size // 2])

    # Track bounds for auto-scaling and centering
    self.min_x, self.max_x = 0.0, 0.0
    self.min_z, self.max_z = 0.0, 0.0
    self.center_x, self.center_z = 0.0, 0.0  # World center of trajectory

    # Store all translations for redrawing
    self.all_translations: list[tuple[float, float, float]] = []
    self.loop_closures: list[tuple[int, int]] = []

    # Colors (BGR format)
    self.bg_color = (15, 15, 20)
    self.grid_color = (35, 35, 45)
    self.trajectory_color = (180, 230, 80)  # Cyan-green
    self.trajectory_glow = (60, 80, 30)  # Darker glow
    self.current_color = (80, 120, 255)  # Orange-red
    self.start_color = (80, 255, 80)  # Green
    self.loop_closure_color = (0, 200, 255)  # Yellow-orange

  def world_to_map(self, x: float, z: float) -> tuple[int, int]:
    """Convert world coordinates to map pixel coordinates."""
    # Offset by trajectory center so the trajectory is centered on the map
    px = int(self.map_center[0] + (x - self.center_x) * self.map_scale)
    py = int(self.map_center[1] - (z - self.center_z) * self.map_scale)
    return (px, py)

  def update_scale(self):
    """Update map scale and center based on trajectory bounds."""
    if not self.all_translations:
      return

    xs = [t[0] for t in self.all_translations]
    zs = [t[2] for t in self.all_translations]

    self.min_x, self.max_x = min(xs), max(xs)
    self.min_z, self.max_z = min(zs), max(zs)

    # Calculate trajectory center
    self.center_x = (self.min_x + self.max_x) / 2
    self.center_z = (self.min_z + self.max_z) / 2

    # Calculate range (actual extent of trajectory)
    range_x = (self.max_x - self.min_x) + 20  # Add padding
    range_z = (self.max_z - self.min_z) + 20

    max_range = max(range_x, range_z, 1.0)
    self.map_scale = (self.map_size * 0.8) / max_range

  def _draw_grid(self):
    """Draw a subtle grid on the map."""
    grid_spacing = 50
    for i in range(0, self.map_size, grid_spacing):
      cv2.line(self.trajectory_map, (i, 0), (i, self.map_size), self.grid_color, 1)
      cv2.line(self.trajectory_map, (0, i), (self.map_size, i), self.grid_color, 1)

    # Draw center crosshair
    cx, cy = self.map_size // 2, self.map_size // 2
    cv2.line(self.trajectory_map, (cx - 10, cy), (cx + 10, cy), (50, 50, 60), 1)
    cv2.line(self.trajectory_map, (cx, cy - 10), (cx, cy + 10), (50, 50, 60), 1)

  def redraw_trajectory(self):
    """Redraw the entire trajectory map with nice styling."""
    # Fill background
    self.trajectory_map[:] = self.bg_color

    # Draw grid
    self._draw_grid()

    if len(self.all_translations) < 2:
      # Draw start point even with no trajectory
      if self.all_translations:
        t = self.all_translations[0]
        p = self.world_to_map(t[0], t[2])
        cv2.circle(self.trajectory_map, p, 6, self.start_color, -1, cv2.LINE_AA)
      return

    # Convert to points array for polylines
    points = [self.world_to_map(t[0], t[2]) for t in self.all_translations]
    pts = np.array(points, dtype=np.int32)

    # Draw glow effect (thicker, darker line behind)
    cv2.polylines(self.trajectory_map, [pts], False, self.trajectory_glow, 4, cv2.LINE_AA)

    # Draw main trajectory line
    cv2.polylines(self.trajectory_map, [pts], False, self.trajectory_color, 2, cv2.LINE_AA)

    # Draw loop closures
    for query_idx, match_idx in self.loop_closures:
      q_idx = query_idx - 1
      m_idx = match_idx - 1

      if 0 <= q_idx < len(self.all_translations) and 0 <= m_idx < len(self.all_translations):
        p1 = self.world_to_map(self.all_translations[q_idx][0], self.all_translations[q_idx][2])
        p2 = self.world_to_map(self.all_translations[m_idx][0], self.all_translations[m_idx][2])

        # Dashed effect - draw loop closure line
        cv2.line(self.trajectory_map, p1, p2, self.loop_closure_color, 2, cv2.LINE_AA)

        # Small markers at connection points
        cv2.circle(self.trajectory_map, p1, 4, self.loop_closure_color, -1, cv2.LINE_AA)
        cv2.circle(self.trajectory_map, p2, 4, self.loop_closure_color, -1, cv2.LINE_AA)

    # Draw start position (green)
    start_p = points[0]
    cv2.circle(self.trajectory_map, start_p, 7, (40, 120, 40), -1, cv2.LINE_AA)
    cv2.circle(self.trajectory_map, start_p, 7, self.start_color, 2, cv2.LINE_AA)

    # Draw current position (orange with glow)
    curr_p = points[-1]
    cv2.circle(self.trajectory_map, curr_p, 10, (40, 60, 120), -1, cv2.LINE_AA)  # Glow
    cv2.circle(self.trajectory_map, curr_p, 6, self.current_color, -1, cv2.LINE_AA)

  def update(self, vo) -> np.ndarray:
    """Update display with current visual odometry state."""
    if vo.translations:
      self.all_translations = [(t[0, 0], t[1, 0], t[2, 0]) for t in vo.translations]
      self.loop_closures = vo.loop_closures

      if len(self.all_translations) % 10 == 0:
        self.update_scale()

      self.redraw_trajectory()

    # Get feature image
    feature_img = vo.draw_img if vo.draw_img is not None else np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    # Resize map to match frame height (use INTER_AREA for better downscaling)
    map_display = cv2.resize(self.trajectory_map, (self.frame_height, self.frame_height), interpolation=cv2.INTER_AREA)

    # Add labels with background for readability
    self._draw_label(feature_img, "Features", (10, 25), 0.7)
    self._draw_label(map_display, "Trajectory", (10, 25), 0.6)

    # Add stats
    if self.all_translations:
      t = self.all_translations[-1]
      stats = f"X:{t[0]:.1f}  Z:{t[2]:.1f}"
      self._draw_label(map_display, stats, (10, 50), 0.45, color=(200, 200, 200))

      frame_text = f"Frame: {len(self.all_translations)}"
      self._draw_label(map_display, frame_text, (10, 75), 0.45, color=(150, 150, 150))

    if self.loop_closures:
      lc_text = f"Loops: {len(self.loop_closures)}"
      self._draw_label(map_display, lc_text, (10, 100), 0.45, color=self.loop_closure_color)

    return np.hstack([feature_img, map_display])

  def _draw_label(self, img: np.ndarray, text: str, pos: tuple, scale: float, color=(255, 255, 255)):
    """Draw text with a subtle shadow for readability."""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    # Text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

  def show(self, vo) -> int:
    """Update and display. Returns key pressed."""
    combined = self.update(vo)
    cv2.imshow(self.window_name, combined)
    return cv2.waitKey(1)
