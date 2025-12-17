from multiprocessing import Queue, Process

import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu


class VisualOdometryState:
  def __init__(self):
    self.poses = np.array([])
    self.translations = np.array([])
    self.loop_closures: list[tuple[int, int]] = []


class Display:
  def __init__(self):
    self.state = None
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    width, height = 1024, 550
    self.viewer_init(width, height)

    clock = pygame.time.Clock()
    running = True

    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            running = False

      self.viewer_refresh(q)
      pygame.display.flip()
      clock.tick(60)

    pygame.quit()
    print("Quitting viewer...")

  def viewer_init(self, w, h):
    pygame.init()
    pygame.display.set_mode((w, h), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Map Viewer")

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45, w / h, 0.1, 5000.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    self.camera_x = 0
    self.camera_y = 50
    self.camera_z = -80

    self.target_x = 0
    self.target_y = 0
    self.target_z = 0

    self.up_x = 0
    self.up_y = -1
    self.up_z = 0

    self.pointSize = 3
    self.current_pose = np.eye(4)

  def draw_camera_frustum(self, pose, scale=0.1):
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

  def draw_line_strip(self, points):
    if len(points) < 2:
      return

    gl.glBegin(gl.GL_LINE_STRIP)
    for point in points:
      gl.glVertex3f(point[0], point[1], point[2])
    gl.glEnd()

  def draw_loop_closure(self, pos1, pos2):
    """Draw a line connecting two loop closure positions."""
    gl.glBegin(gl.GL_LINES)
    gl.glVertex3f(pos1[0], pos1[1], pos1[2])
    gl.glVertex3f(pos2[0], pos2[1], pos2[2])
    gl.glEnd()

  def viewer_refresh(self, q):
    while not q.empty():
      self.state = q.get()

    if self.state is not None and self.state.poses.shape[0] >= 1:
      current_pose = self.state.poses[-1]
      self.current_pose = current_pose

      pos = current_pose[:3, 3]

      self.camera_x = pos[0] - 10
      self.camera_y = pos[1] + 20
      self.camera_z = pos[2] - 30

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()

    glu.gluLookAt(
      self.camera_x,
      self.camera_y,
      self.camera_z,
      self.target_x,
      self.target_y,
      self.target_z,
      self.up_x,
      self.up_y,
      self.up_z,
    )

    if self.state is not None:
      if self.state.poses.shape[0] >= 2:
        # Draw previous poses
        gl.glColor3f(0.0, 1.0, 0.0)
        for pose in self.state.poses[:-1]:
          self.draw_camera_frustum(pose)

      if self.state.poses.shape[0] >= 1:
        # Draw current pose
        gl.glColor3f(1.0, 0.0, 0.0)
        current_pose = self.state.poses[-1]
        self.draw_camera_frustum(current_pose)

      if self.state.translations.shape[0] != 0:
        gl.glPointSize(self.pointSize)
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glLineWidth(2.0)
        self.draw_line_strip(self.state.translations)

      # Draw loop closures as yellow lines
      if self.state.loop_closures and self.state.translations.shape[0] > 0:
        gl.glColor3f(1.0, 1.0, 0.0)  # Yellow for loop closures
        gl.glLineWidth(3.0)

        for query_frame, match_frame in self.state.loop_closures:
          # Frame indices are offset by 1 from translations array (frame 0 has no translation)
          query_idx = query_frame - 1
          match_idx = match_frame - 1

          if 0 <= query_idx < len(self.state.translations) and 0 <= match_idx < len(self.state.translations):
            pos1 = self.state.translations[query_idx]
            pos2 = self.state.translations[match_idx]
            self.draw_loop_closure(pos1, pos2)

  def update(self, vo):
    if self.q is None:
      return

    state = VisualOdometryState()
    state.poses = np.array(vo.poses)
    state.translations = np.array(vo.translations).reshape(-1, 3)
    state.loop_closures = vo.loop_closures
    self.q.put(state)
