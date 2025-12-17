"""3D visualization using pygame and OpenGL."""

import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
from pygame.locals import OPENGL, DOUBLEBUF


class Display3D:
  """3D map viewer using pygame and OpenGL."""

  def __init__(self, width: int = 800, height: int = 600):
    self.width = width
    self.height = height

    pygame.init()
    pygame.display.set_caption("3D Map Viewer")
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)

    self._init_gl()

    self.camera_x = 0.0
    self.camera_y = 50.0
    self.camera_z = -80.0
    self.current_pose = np.eye(4)

  def _init_gl(self):
    """Initialize OpenGL settings."""
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45, self.width / self.height, 0.1, 5000.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

  def _draw_camera_frustum(self, pose: np.ndarray, scale: float = 0.1):
    """Draw a camera frustum at the given pose."""
    gl.glPushMatrix()
    gl.glMultMatrixf(pose.T.flatten())

    vertices = np.array(
      [
        [0, 0, 0],
        [-scale, -scale, scale],
        [scale, -scale, scale],
        [scale, scale, scale],
        [-scale, scale, scale],
      ]
    )

    gl.glBegin(gl.GL_LINES)
    for i in range(1, 5):
      gl.glVertex3f(*vertices[0])
      gl.glVertex3f(*vertices[i])
    gl.glEnd()

    gl.glBegin(gl.GL_LINE_LOOP)
    for i in range(1, 5):
      gl.glVertex3f(*vertices[i])
    gl.glEnd()

    gl.glPopMatrix()

  def _draw_trajectory(self, translations: np.ndarray):
    """Draw the trajectory as a line strip."""
    if len(translations) < 2:
      return

    gl.glBegin(gl.GL_LINE_STRIP)
    for point in translations:
      gl.glVertex3f(point[0], point[1], point[2])
    gl.glEnd()

  def _draw_loop_closure(self, pos1: np.ndarray, pos2: np.ndarray):
    """Draw a line connecting two loop closure positions."""
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(pos1[0], pos1[1], pos1[2])
    gl.glVertex3f(pos2[0], pos2[1], pos2[2])
    gl.glEnd()

  def update(self, vo) -> bool:
    """
    Update and render the 3D view.
    Returns False if window was closed.
    """
    # Handle pygame events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        return False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          return False

    poses = np.array(vo.poses) if vo.poses else np.array([])
    translations = np.array(vo.translations).reshape(-1, 3) if vo.translations else np.array([])
    loop_closures = vo.loop_closures

    # Update camera to follow current position
    if len(poses) >= 1:
      current_pose = poses[-1]
      self.current_pose = current_pose
      pos = current_pose[:3, 3]

      self.camera_x = pos[0] - 10
      self.camera_y = pos[1] + 20
      self.camera_z = pos[2] - 30

    # Clear and set up view
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
    gl.glClearColor(0.1, 0.1, 0.1, 1.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    # Look at origin, following the trajectory
    target_x = pos[0] if len(poses) >= 1 else 0
    target_y = pos[1] if len(poses) >= 1 else 0
    target_z = pos[2] if len(poses) >= 1 else 0

    glu.gluLookAt(
      self.camera_x,
      self.camera_y,
      self.camera_z,
      target_x,
      target_y,
      target_z,
      0,
      -1,
      0,
    )

    # Draw previous poses (green)
    if len(poses) >= 2:
      gl.glColor3f(0.0, 1.0, 0.0)
      for pose in poses[:-1]:
        self._draw_camera_frustum(pose)

    # Draw current pose (red)
    if len(poses) >= 1:
      gl.glColor3f(1.0, 0.0, 0.0)
      self._draw_camera_frustum(poses[-1])

    # Draw trajectory (blue)
    if len(translations) > 0:
      gl.glColor3f(0.0, 0.5, 1.0)
      gl.glLineWidth(2.0)
      self._draw_trajectory(translations)

    # Draw loop closures (yellow)
    if loop_closures and len(translations) > 0:
      gl.glColor3f(1.0, 1.0, 0.0)
      gl.glLineWidth(3.0)

      for query_frame, match_frame in loop_closures:
        query_idx = query_frame - 1
        match_idx = match_frame - 1

        if 0 <= query_idx < len(translations) and 0 <= match_idx < len(translations):
          self._draw_loop_closure(translations[query_idx], translations[match_idx])

    pygame.display.flip()
    return True

  def close(self):
    """Clean up pygame."""
    pygame.quit()
