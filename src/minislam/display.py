"""Unified display module with consolidated single window layout."""

import cv2
import numpy as np

from minislam.display3d import Display3D


class Display:
  """
  Consolidated display with layout:
  - Top: input dataset (features)
  - Bottom left: 3D map (interactive)
  - Bottom right: top-down trajectory map
  """

  def __init__(self, frame_width: int, frame_height: int, window_name: str = "MiniSLAM"):
    self.frame_width = frame_width
    self.frame_height = frame_height
    self.window_name = window_name

    # Bottom panels will be square, matching frame height
    self.panel_size = frame_height

    # Initialize 3D renderer (offscreen)
    self.display3d = Display3D(width=self.panel_size, height=self.panel_size, offscreen=True)

    # Create and position window at top-left
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 0, 0)

    # Set up mouse callback
    cv2.setMouseCallback(window_name, self._mouse_callback)

    # Mouse state for 3D interaction
    self.mouse_down = False
    self.last_mouse_x = 0
    self.last_mouse_y = 0

    # Store 3D view bounds in the combined image
    self.view3d_x = 0
    self.view3d_y = frame_height
    self.view3d_width = frame_width // 2
    self.view3d_height = frame_height

    # Trajectory map state
    self.map_scale = 1.0
    self.map_center = np.array([self.panel_size // 2, self.panel_size // 2])
    self.min_x, self.max_x = 0.0, 0.0
    self.min_z, self.max_z = 0.0, 0.0
    self.center_x, self.center_z = 0.0, 0.0

    # Store translations
    self.all_translations: list[tuple[float, float, float]] = []
    self.loop_closures: list[tuple[int, int]] = []

    # Smoothing
    self.smooth_window = 7

  def _mouse_callback(self, event, x, y, flags, param):
    """Handle mouse events for 3D view interaction."""
    x_in_bounds = self.view3d_x <= x < self.view3d_x + self.view3d_width
    y_in_bounds = self.view3d_y <= y < self.view3d_y + self.view3d_height
    in_3d_view = x_in_bounds and y_in_bounds

    if event == cv2.EVENT_LBUTTONDOWN:
      if in_3d_view:
        self.mouse_down = True
        self.last_mouse_x = x
        self.last_mouse_y = y
        self.display3d.auto_follow = False

    elif event == cv2.EVENT_LBUTTONUP:
      self.mouse_down = False

    elif event == cv2.EVENT_MOUSEMOVE:
      if self.mouse_down and in_3d_view:
        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        self.display3d.rotate(dx, dy)
        self.last_mouse_x = x
        self.last_mouse_y = y

    elif event == cv2.EVENT_MOUSEWHEEL:
      if in_3d_view:
        delta = 1 if flags > 0 else -1
        self.display3d.zoom(delta)
        self.display3d.auto_follow = False

    elif event == cv2.EVENT_RBUTTONDOWN:
      if in_3d_view:
        self.display3d.auto_follow = True

  def world_to_map(self, x: float, z: float) -> tuple[int, int]:
    """Convert world coordinates to map pixel coordinates."""
    px = int(self.map_center[0] + (x - self.center_x) * self.map_scale)
    py = int(self.map_center[1] + (z - self.center_z) * self.map_scale)
    return (px, py)

  def update_scale(self):
    """Update map scale and center based on trajectory bounds."""
    if not self.all_translations:
      return

    xs = [t[0] for t in self.all_translations]
    zs = [t[2] for t in self.all_translations]

    self.min_x, self.max_x = min(xs), max(xs)
    self.min_z, self.max_z = min(zs), max(zs)

    self.center_x = (self.min_x + self.max_x) / 2
    self.center_z = (self.min_z + self.max_z) / 2

    range_x = (self.max_x - self.min_x) + 30
    range_z = (self.max_z - self.min_z) + 30

    max_range = max(range_x, range_z, 1.0)
    self.map_scale = (self.panel_size * 0.75) / max_range

  def _smooth_trajectory(self, translations: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Apply moving average smoothing to trajectory."""
    if len(translations) < self.smooth_window:
      return translations

    smoothed = []
    half_win = self.smooth_window // 2

    for i in range(len(translations)):
      start = max(0, i - half_win)
      end = min(len(translations), i + half_win + 1)

      avg_x = sum(t[0] for t in translations[start:end]) / (end - start)
      avg_y = sum(t[1] for t in translations[start:end]) / (end - start)
      avg_z = sum(t[2] for t in translations[start:end]) / (end - start)

      smoothed.append((avg_x, avg_y, avg_z))

    return smoothed

  def _render_top_view(self, vo) -> np.ndarray:
    """Render the top-down trajectory view from scratch."""
    size = self.panel_size
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Dark gradient background
    for y in range(size):
      intensity = int(10 + (y / size) * 8)
      img[y, :] = (intensity, intensity, intensity + 2)

    center = size // 2

    # Draw grid with fade
    grid_step = 40
    for i in range(0, size, grid_step):
      # Fade based on distance from center
      dist = abs(i - center) / center
      alpha = int(25 * (1 - dist * 0.5))

      # Vertical lines
      color = (alpha, alpha, alpha + 5)
      cv2.line(img, (i, 0), (i, size), color, 1)
      # Horizontal lines
      cv2.line(img, (0, i), (size, i), color, 1)

    if len(self.all_translations) < 2:
      if self.all_translations:
        t = self.all_translations[0]
        p = self.world_to_map(t[0], t[2])
        # Start marker
        cv2.circle(img, p, 8, (0, 255, 100), -1, cv2.LINE_AA)
        cv2.circle(img, p, 10, (0, 200, 80), 2, cv2.LINE_AA)
      return img

    # Smooth the trajectory
    smoothed = self._smooth_trajectory(self.all_translations)
    points = [self.world_to_map(t[0], t[2]) for t in smoothed]
    n = len(points)

    # Draw trajectory with neon green gradient glow
    # Multiple passes for glow effect
    for glow_size, glow_alpha in [(8, 0.15), (5, 0.3), (3, 0.5)]:
      for i in range(n - 1):
        progress = i / max(n - 1, 1)
        # Neon green with slight color shift along path
        g = int(255 - progress * 30)
        b = int(50 + progress * 30)
        color = (b, g, 0)  # BGR - neon green

        # Blend with alpha simulation
        glow_color = (int(color[0] * glow_alpha), int(color[1] * glow_alpha), int(color[2] * glow_alpha))
        cv2.line(img, points[i], points[i + 1], glow_color, glow_size, cv2.LINE_AA)

    # Draw main trajectory line with gradient
    for i in range(n - 1):
      progress = i / max(n - 1, 1)
      # Neon green gradient
      g = int(255 - progress * 20)
      r = int(progress * 40)
      color = (0, g, r)  # BGR

      thickness = 2
      cv2.line(img, points[i], points[i + 1], color, thickness, cv2.LINE_AA)

    # Draw loop closures as cyan arcs
    for query_idx, match_idx in self.loop_closures:
      q_idx = query_idx - 1
      m_idx = match_idx - 1

      if 0 <= q_idx < len(points) and 0 <= m_idx < len(points):
        p1 = points[q_idx]
        p2 = points[m_idx]

        # Draw curved arc
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        # Offset perpendicular to line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1, int(np.sqrt(dx * dx + dy * dy)))
        offset = length // 6

        # Control point for curve (length is guaranteed >= 1)
        ctrl_x = mid_x - int(dy * offset / length)
        ctrl_y = mid_y + int(dx * offset / length)

        # Draw bezier-like curve with line segments
        prev = p1
        for t in np.linspace(0, 1, 15):
          # Quadratic bezier
          x = int((1 - t) ** 2 * p1[0] + 2 * (1 - t) * t * ctrl_x + t**2 * p2[0])
          y = int((1 - t) ** 2 * p1[1] + 2 * (1 - t) * t * ctrl_y + t**2 * p2[1])
          cv2.line(img, prev, (x, y), (255, 200, 0), 1, cv2.LINE_AA)
          prev = (x, y)

        # Small markers at endpoints
        cv2.circle(img, p1, 4, (255, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 4, (255, 200, 0), -1, cv2.LINE_AA)

    # Draw start position (green with glow)
    start_p = points[0]
    cv2.circle(img, start_p, 12, (0, 80, 0), -1, cv2.LINE_AA)  # Outer glow
    cv2.circle(img, start_p, 8, (0, 180, 50), -1, cv2.LINE_AA)  # Inner
    cv2.circle(img, start_p, 5, (100, 255, 150), -1, cv2.LINE_AA)  # Bright center

    # Draw current position (red with glow)
    curr_p = points[-1]
    # Outer glow layers
    cv2.circle(img, curr_p, 18, (0, 0, 60), -1, cv2.LINE_AA)
    cv2.circle(img, curr_p, 14, (0, 0, 100), -1, cv2.LINE_AA)
    cv2.circle(img, curr_p, 10, (0, 0, 180), -1, cv2.LINE_AA)
    # Bright center
    cv2.circle(img, curr_p, 6, (0, 80, 255), -1, cv2.LINE_AA)
    cv2.circle(img, curr_p, 3, (100, 150, 255), -1, cv2.LINE_AA)

    # Draw direction indicator at current position
    if n >= 2:
      # Get direction from last few points
      dx = points[-1][0] - points[-2][0]
      dy = points[-1][1] - points[-2][1]
      length = max(1, np.sqrt(dx * dx + dy * dy))
      if length > 0.5:
        # Normalize and extend
        dx, dy = dx / length * 15, dy / length * 15
        arrow_end = (int(curr_p[0] + dx), int(curr_p[1] + dy))
        cv2.arrowedLine(img, curr_p, arrow_end, (0, 100, 255), 2, cv2.LINE_AA, tipLength=0.4)

    # Add labels and stats
    self._draw_label(img, "Top View", (10, 25), 0.5, (200, 200, 200))

    if self.all_translations:
      t = self.all_translations[-1]
      stats = f"X:{t[0]:.1f}  Z:{t[2]:.1f}"
      self._draw_label(img, stats, (10, 50), 0.4, (150, 150, 150))

      frame_text = f"Frame: {len(self.all_translations)}"
      self._draw_label(img, frame_text, (10, 70), 0.35, (120, 120, 120))

    kf_text = f"KF: {vo.num_keyframes}"
    self._draw_label(img, kf_text, (10, 90), 0.35, (0, 200, 100))

    if self.loop_closures:
      lc_text = f"Loops: {len(self.loop_closures)}"
      self._draw_label(img, lc_text, (10, 110), 0.35, (255, 200, 0))

    return img

  def _draw_label(self, img: np.ndarray, text: str, pos: tuple, scale: float, color=(255, 255, 255)):
    """Draw text with shadow for readability."""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    # Text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

  def update(self, vo) -> np.ndarray:
    """Update display with current visual odometry state."""
    if vo.translations:
      self.all_translations = [(t[0, 0], t[1, 0], t[2, 0]) for t in vo.translations]
      self.loop_closures = vo.loop_closures

      if len(self.all_translations) % 10 == 0:
        self.update_scale()

    # Get feature image (top panel)
    feature_img = vo.draw_img if vo.draw_img is not None else np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    # Render 3D view (bottom left)
    map_3d = self.display3d.render(vo)

    # Render top view (bottom right)
    map_2d = self._render_top_view(vo)

    # Add labels to other panels
    self._draw_label(feature_img, "Input", (10, 25), 0.6, (200, 200, 200))

    self._draw_label(map_3d, "3D View", (10, 25), 0.5, (200, 200, 200))
    if self.display3d.auto_follow:
      self._draw_label(map_3d, "Auto-follow ON", (10, self.panel_size - 15), 0.35, (100, 255, 100))
    else:
      self._draw_label(map_3d, "Drag:rotate  Scroll:zoom  RClick:reset", (10, self.panel_size - 15), 0.3, (120, 120, 120))

    # Resize panels to fit layout
    bottom_width = self.frame_width
    panel_target_width = bottom_width // 2
    map_3d_resized = cv2.resize(map_3d, (panel_target_width, self.panel_size), interpolation=cv2.INTER_AREA)
    map_2d_resized = cv2.resize(map_2d, (bottom_width - panel_target_width, self.panel_size), interpolation=cv2.INTER_AREA)

    # Update 3D view bounds for mouse interaction
    self.view3d_x = 0
    self.view3d_y = self.frame_height
    self.view3d_width = panel_target_width
    self.view3d_height = self.panel_size

    # Compose layout
    bottom_row = np.hstack([map_3d_resized, map_2d_resized])
    combined = np.vstack([feature_img, bottom_row])

    return combined

  def show(self, vo) -> int:
    """Update and display. Returns key pressed."""
    if not self.display3d.process_events():
      return 27

    combined = self.update(vo)
    cv2.imshow(self.window_name, combined)
    return cv2.waitKey(1)

  def close(self):
    """Clean up resources."""
    cv2.destroyAllWindows()
    self.display3d.close()
