"""3D visualization using pygame and OpenGL with offscreen rendering support."""

import math

import cv2
import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
from pygame.locals import HIDDEN, OPENGL, DOUBLEBUF


class Display3D:
  """3D map viewer using pygame and OpenGL with interactive orbital camera."""

  def __init__(self, width: int = 800, height: int = 600, offscreen: bool = False):
    self.width = width
    self.height = height
    self.offscreen = offscreen

    pygame.init()

    if offscreen:
      flags = DOUBLEBUF | OPENGL | HIDDEN
    else:
      flags = DOUBLEBUF | OPENGL

    pygame.display.set_caption("3D Map Viewer")
    pygame.display.set_mode((width, height), flags)

    self._init_gl()

    # Orbital camera parameters
    self.camera_distance = 50.0
    self.camera_yaw = 45.0
    self.camera_pitch = 30.0

    # Camera target
    self.target = np.array([0.0, 0.0, 0.0])
    self.auto_follow = True

    # Camera limits
    self.min_distance = 5.0
    self.max_distance = 500.0
    self.min_pitch = -89.0
    self.max_pitch = 89.0

    # Trajectory tube parameters
    self.tube_segments = 8
    self.tube_radius = 0.15

  def _init_gl(self):
    """Initialize OpenGL settings."""
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)

    # Enable smooth lines and points
    gl.glEnable(gl.GL_LINE_SMOOTH)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

    # Enable blending for smooth lines
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Enable lighting
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)
    gl.glEnable(gl.GL_COLOR_MATERIAL)
    gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

    # Light position and properties
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.5, 1.0, 0.5, 0.0])
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

    # Enable fog for depth
    gl.glEnable(gl.GL_FOG)
    gl.glFogi(gl.GL_FOG_MODE, gl.GL_LINEAR)
    gl.glFogfv(gl.GL_FOG_COLOR, [0.08, 0.08, 0.1, 1.0])
    gl.glFogf(gl.GL_FOG_START, 30.0)
    gl.glFogf(gl.GL_FOG_END, 200.0)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45, self.width / self.height, 0.1, 5000.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

  def rotate(self, dx: float, dy: float):
    """Rotate camera based on mouse delta."""
    sensitivity = 0.3
    self.camera_yaw += dx * sensitivity
    self.camera_pitch -= dy * sensitivity
    self.camera_pitch = max(self.min_pitch, min(self.max_pitch, self.camera_pitch))
    self.camera_yaw = self.camera_yaw % 360.0

  def zoom(self, delta: float):
    """Zoom camera in/out."""
    zoom_speed = 5.0
    self.camera_distance -= delta * zoom_speed
    self.camera_distance = max(self.min_distance, min(self.max_distance, self.camera_distance))

  def _get_camera_position(self) -> np.ndarray:
    """Calculate camera position from orbital parameters."""
    yaw_rad = math.radians(self.camera_yaw)
    pitch_rad = math.radians(self.camera_pitch)

    x = self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
    y = self.camera_distance * math.sin(pitch_rad)
    z = self.camera_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

    return self.target + np.array([x, y, z])

  def _get_perpendicular_vectors(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get two perpendicular vectors to the given direction."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Choose an up vector that's not parallel to direction
    if abs(direction[1]) < 0.9:
      up = np.array([0.0, 1.0, 0.0])
    else:
      up = np.array([1.0, 0.0, 0.0])

    right = np.cross(direction, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, direction)
    up = up / (np.linalg.norm(up) + 1e-8)

    return right, up

  def _draw_tube_segment(self, p1: np.ndarray, p2: np.ndarray, color1: tuple, color2: tuple, radius: float):
    """Draw a tube segment between two points with gradient color."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
      return

    right, up = self._get_perpendicular_vectors(direction)

    # Generate circle points
    angles = np.linspace(0, 2 * np.pi, self.tube_segments, endpoint=False)

    gl.glBegin(gl.GL_QUAD_STRIP)
    for angle in np.append(angles, angles[0]):
      # Calculate normal and position on circle
      normal = right * math.cos(angle) + up * math.sin(angle)
      offset = normal * radius

      # Point on first circle
      gl.glNormal3f(*normal)
      gl.glColor3f(*color1)
      gl.glVertex3f(*(p1 + offset))

      # Point on second circle
      gl.glColor3f(*color2)
      gl.glVertex3f(*(p2 + offset))

    gl.glEnd()

  def _draw_trajectory_tube(self, translations: np.ndarray):
    """Draw the trajectory as a 3D tube with gradient coloring."""
    if len(translations) < 2:
      return

    # Smooth the trajectory
    if len(translations) > 5:
      smoothed = np.copy(translations)
      for i in range(2, len(translations) - 2):
        smoothed[i] = np.mean(translations[i - 2 : i + 3], axis=0)
      translations = smoothed

    n = len(translations)

    for i in range(n - 1):
      # Progress along trajectory (0 to 1)
      t1 = i / max(n - 1, 1)
      t2 = (i + 1) / max(n - 1, 1)

      # Color gradient: cyan at start -> magenta at end
      color1 = (0.0 + t1 * 0.8, 0.7 - t1 * 0.4, 1.0 - t1 * 0.2)
      color2 = (0.0 + t2 * 0.8, 0.7 - t2 * 0.4, 1.0 - t2 * 0.2)

      self._draw_tube_segment(translations[i], translations[i + 1], color1, color2, self.tube_radius)

  def _draw_trajectory_shadow(self, translations: np.ndarray, ground_y: float = 0.0):
    """Draw a shadow of the trajectory on the ground."""
    if len(translations) < 2:
      return

    gl.glDisable(gl.GL_LIGHTING)
    gl.glColor4f(0.0, 0.0, 0.0, 0.3)

    gl.glBegin(gl.GL_LINE_STRIP)
    for point in translations:
      gl.glVertex3f(point[0], ground_y + 0.01, point[2])
    gl.glEnd()

    gl.glEnable(gl.GL_LIGHTING)

  def _draw_camera_frustum(self, pose: np.ndarray, scale: float = 0.1, filled: bool = False):
    """Draw a camera frustum at the given pose."""
    gl.glPushMatrix()
    gl.glMultMatrixf(pose.T.flatten())

    # Frustum vertices
    apex = np.array([0, 0, 0])
    bl = np.array([-scale, -scale * 0.75, scale * 1.5])
    br = np.array([scale, -scale * 0.75, scale * 1.5])
    tr = np.array([scale, scale * 0.75, scale * 1.5])
    tl = np.array([-scale, scale * 0.75, scale * 1.5])

    if filled:
      # Draw filled faces
      gl.glBegin(gl.GL_TRIANGLES)

      # Front face
      normal = np.cross(br - bl, tl - bl)
      normal = normal / (np.linalg.norm(normal) + 1e-8)
      gl.glNormal3f(*normal)
      gl.glVertex3f(*bl)
      gl.glVertex3f(*br)
      gl.glVertex3f(*tr)
      gl.glVertex3f(*bl)
      gl.glVertex3f(*tr)
      gl.glVertex3f(*tl)

      # Side faces
      for v1, v2 in [(bl, br), (br, tr), (tr, tl), (tl, bl)]:
        normal = np.cross(v2 - apex, v1 - apex)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        gl.glNormal3f(*normal)
        gl.glVertex3f(*apex)
        gl.glVertex3f(*v1)
        gl.glVertex3f(*v2)

      gl.glEnd()
    else:
      # Draw wireframe
      gl.glDisable(gl.GL_LIGHTING)
      gl.glLineWidth(1.5)

      gl.glBegin(gl.GL_LINES)
      for v in [bl, br, tr, tl]:
        gl.glVertex3f(*apex)
        gl.glVertex3f(*v)
      gl.glEnd()

      gl.glBegin(gl.GL_LINE_LOOP)
      for v in [bl, br, tr, tl]:
        gl.glVertex3f(*v)
      gl.glEnd()

      gl.glEnable(gl.GL_LIGHTING)

    gl.glPopMatrix()

  def _draw_sphere(self, center: np.ndarray, radius: float, slices: int = 12, stacks: int = 8):
    """Draw a sphere at the given position."""
    gl.glPushMatrix()
    gl.glTranslatef(*center)

    quadric = glu.gluNewQuadric()
    glu.gluQuadricNormals(quadric, glu.GLU_SMOOTH)
    glu.gluSphere(quadric, radius, slices, stacks)
    glu.gluDeleteQuadric(quadric)

    gl.glPopMatrix()

  def _draw_grid(self, translations: np.ndarray):
    """Draw a ground grid centered on trajectory."""
    gl.glDisable(gl.GL_LIGHTING)

    # Calculate grid center from trajectory
    if len(translations) > 0:
      center_x = np.mean(translations[:, 0])
      center_z = np.mean(translations[:, 2])
    else:
      center_x, center_z = 0, 0

    grid_size = 100
    grid_step = 5

    gl.glBegin(gl.GL_LINES)
    for i in range(-grid_size, grid_size + 1, grid_step):
      # Fade grid lines based on distance from center
      dist = abs(i) / grid_size
      alpha = 0.3 * (1.0 - dist * 0.7)
      gl.glColor4f(0.3, 0.3, 0.4, alpha)

      gl.glVertex3f(center_x + i, 0, center_z - grid_size)
      gl.glVertex3f(center_x + i, 0, center_z + grid_size)
      gl.glVertex3f(center_x - grid_size, 0, center_z + i)
      gl.glVertex3f(center_x + grid_size, 0, center_z + i)
    gl.glEnd()

    gl.glEnable(gl.GL_LIGHTING)

  def _draw_axes(self):
    """Draw coordinate axes at origin."""
    gl.glDisable(gl.GL_LIGHTING)
    gl.glLineWidth(2.0)

    gl.glBegin(gl.GL_LINES)
    # X axis - red
    gl.glColor3f(0.9, 0.2, 0.2)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(3, 0, 0)

    # Y axis - green
    gl.glColor3f(0.2, 0.9, 0.2)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(0, 3, 0)

    # Z axis - blue
    gl.glColor3f(0.2, 0.2, 0.9)
    gl.glVertex3f(0, 0, 0)
    gl.glVertex3f(0, 0, 3)
    gl.glEnd()

    gl.glLineWidth(1.0)
    gl.glEnable(gl.GL_LIGHTING)

  def _draw_loop_closure(self, pos1: np.ndarray, pos2: np.ndarray):
    """Draw a loop closure connection as a curved arc."""
    gl.glDisable(gl.GL_LIGHTING)
    gl.glColor3f(1.0, 0.9, 0.2)
    gl.glLineWidth(2.0)

    # Draw as a subtle arc above the trajectory
    height = np.linalg.norm(pos2 - pos1) * 0.15

    gl.glBegin(gl.GL_LINE_STRIP)
    for t in np.linspace(0, 1, 20):
      # Parabolic arc
      p = pos1 * (1 - t) + pos2 * t
      p[1] += height * 4 * t * (1 - t)
      gl.glVertex3f(*p)
    gl.glEnd()

    # Draw small spheres at endpoints
    gl.glEnable(gl.GL_LIGHTING)
    gl.glColor3f(1.0, 0.9, 0.2)
    self._draw_sphere(pos1, 0.2)
    self._draw_sphere(pos2, 0.2)

  def render(self, vo) -> np.ndarray:
    """Render the 3D view and return as numpy array (BGR for OpenCV)."""
    poses = np.array(vo.poses) if vo.poses else np.array([])
    translations = np.array(vo.translations).reshape(-1, 3) if vo.translations else np.array([])
    loop_closures = vo.loop_closures

    # Update target to follow current position
    if self.auto_follow and len(translations) > 0:
      current_pos = translations[-1]
      self.target = self.target * 0.9 + current_pos * 0.1

    # Clear
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore[arg-type]
    gl.glClearColor(0.08, 0.08, 0.1, 1.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    # Set up camera
    camera_pos = self._get_camera_position()
    glu.gluLookAt(
      camera_pos[0],
      camera_pos[1],
      camera_pos[2],
      self.target[0],
      self.target[1],
      self.target[2],
      0,
      1,
      0,
    )

    # Update light position relative to camera
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [camera_pos[0], camera_pos[1] + 50, camera_pos[2], 1.0])

    # Draw ground grid
    self._draw_grid(translations)

    # Draw axes at origin
    self._draw_axes()

    # Draw trajectory shadow on ground
    if len(translations) > 0:
      self._draw_trajectory_shadow(translations, ground_y=0.0)

    # Draw trajectory as 3D tube
    if len(translations) > 1:
      self._draw_trajectory_tube(translations)

    # Draw start sphere (green)
    if len(translations) > 0:
      gl.glColor3f(0.2, 0.9, 0.3)
      self._draw_sphere(translations[0], 0.3)

    # Draw current position sphere (orange/red glow)
    if len(translations) > 0:
      gl.glColor3f(1.0, 0.4, 0.1)
      self._draw_sphere(translations[-1], 0.35)

    # Draw previous camera poses (subtle green)
    if len(poses) >= 2:
      gl.glColor4f(0.2, 0.7, 0.3, 0.6)
      # Only draw every Nth pose to avoid clutter
      step = max(1, len(poses) // 50)
      for i in range(0, len(poses) - 1, step):
        self._draw_camera_frustum(poses[i], scale=0.08)

    # Draw current pose (red, larger)
    if len(poses) >= 1:
      gl.glColor3f(1.0, 0.3, 0.2)
      self._draw_camera_frustum(poses[-1], scale=0.12, filled=True)

    # Draw loop closures
    if loop_closures and len(translations) > 0:
      for query_frame, match_frame in loop_closures:
        query_idx = query_frame - 1
        match_idx = match_frame - 1
        if 0 <= query_idx < len(translations) and 0 <= match_idx < len(translations):
          self._draw_loop_closure(translations[query_idx], translations[match_idx])

    # Read pixels
    gl.glFinish()
    pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
    image = np.flipud(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not self.offscreen:
      pygame.display.flip()

    return image

  def _should_quit(self, event) -> bool:
    """Check if an event indicates the application should quit."""
    if event.type == pygame.QUIT:
      return True
    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
      return True
    return False

  def update(self, vo) -> bool:
    """Update and render the 3D view (legacy interface)."""
    for event in pygame.event.get():
      if self._should_quit(event):
        return False

    self.render(vo)
    pygame.display.flip()
    return True

  def process_events(self) -> bool:
    """Process pygame events. Returns False if should quit."""
    for event in pygame.event.get():
      if self._should_quit(event):
        return False
    return True

  def close(self):
    """Clean up pygame."""
    pygame.quit()
