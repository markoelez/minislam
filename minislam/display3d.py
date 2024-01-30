#!/usr/bin/env python3
import numpy as np
import OpenGL.GL as gl
import pypangolin as pangolin
from multiprocessing import Process, Queue
import sys
sys.path.append('lib')


class VisualOdometryState:
    def __init__(self):
        self.poses = []
        self.translations = []


class Display():
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        width, height = 1024, 550
        self.viewer_init(width, height)
        while not pangolin.ShouldQuit():
            self.viewer_refresh(q)
        print('Quitting viewer...')

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        viewpoint_x = 0
        viewpoint_y = 50
        viewpoint_z = -80
        viewpoint_f = 1000

        self.proj = pangolin.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w // 2, h // 2, 0.1, 5000)
        self.look_view = pangolin.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 180 / w, 1.0, -w / h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.pointSize = 3

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        self.scam.Follow(self.Twc, True)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.dcam.Activate(self.scam)

        if self.state is not None:
            if self.state.poses.shape[0] >= 2:
                # draw previous poses
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawCameras(self.state.poses[:-1])

            if self.state.poses.shape[0] >= 1:
                # draw current pose
                gl.glColor3f(1.0, 0.0, 0.0)
                current_pose = self.state.poses[-1:]
                pangolin.DrawCameras(current_pose)
                self.update_Twc(current_pose[0])

            if self.state.translations.shape[0] != 0:
                # draw estimated translations
                gl.glPointSize(self.pointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                pangolin.DrawLine(self.state.translations)

        pangolin.FinishFrame()

    def update_Twc(self, pose):
        self.Twc.m = pose

    def update(self, vo):
        if self.q is None:
            return

        state = VisualOdometryState()
        state.poses = np.array(vo.poses)
        state.translations = np.array(vo.translations).reshape(-1, 3)
        self.q.put(state)
