"""Stand-in for the Technic/Clover ROS API so board/agent.py can be tested
without a drone: navigate (nav_from_sp interpolation), get_telemetry, land,
camera frames and a fake thermal driver."""
import io
import math
import os
import sys
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'board'))
import numpy as np  # noqa: E402
import thermal_drivers  # noqa: E402
from server.thermal import project  # noqa: E402

NAN = float('nan')


class Res(types.SimpleNamespace):
    pass


class FakeRos:
    def __init__(self, start, all_fake=None):
        self.lock = threading.Lock()
        self.pos = np.array([start[0], start[1], 0.0])
        self.sp = self.pos.copy()
        self.nav = None
        self.yaw = 0.05
        self.armed, self.mode = False, 'POSCTL'
        self.map_seen = False
        self.map_block_until = 0.0
        self.t_map = 0.0
        self.markers, self.t_markers = 0, 0.0
        self.img, self.t_img, self.img_fmt = None, 0.0, 'jpeg'
        self.calls = []
        self.others = all_fake if all_fake is not None else []
        self.others.append(self)
        threading.Thread(target=self._physics, daemon=True).start()
        threading.Thread(target=self._camera, daemon=True).start()

    # --- services ---
    def navigate(self, x, y, z, yaw, speed, frame_id, auto_arm=False):
        with self.lock:
            self.calls.append(('navigate', frame_id, round(x, 2), round(y, 2), round(z, 2)))
            if auto_arm and not self.armed:
                self.armed, self.mode = True, 'OFFBOARD'
                self.sp = self.pos.copy()
            if not self.armed or self.mode != 'OFFBOARD':
                return Res(success=False, message='Not in OFFBOARD mode')
            if frame_id == 'body':
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                tgt = self.pos + np.array([c * x - s * y, s * x + c * y, z])
            elif frame_id == 'aruco_map':
                if not self.map_seen:
                    return Res(success=False, message='Can\'t transform from aruco_map')
                tgt = np.array([x, y, z])
            else:
                return Res(success=False, message='bad frame')
            if not math.isnan(yaw):
                self.yaw = yaw
            self.nav = (self.sp.copy(), tgt, max(speed, 0.05), time.time())
            return Res(success=True, message='')

    def get_telemetry(self, frame_id=''):
        with self.lock:
            ok = frame_id != 'aruco_map' or self.map_seen
            p = self.pos + np.random.normal(0, 0.005, 3)
            return Res(frame_id=frame_id, connected=True, armed=self.armed, mode=self.mode,
                       x=p[0] if ok else NAN, y=p[1] if ok else NAN, z=p[2] if ok else NAN,
                       roll=0.0, pitch=0.0, yaw=self.yaw, vx=0.0, vy=0.0, vz=0.0,
                       voltage=24.6, cell_voltage=4.1)

    def land(self):
        with self.lock:
            self.calls.append(('land',))
            if self.armed:
                self.mode = 'AUTO.LAND'
            return Res(success=True, message='')

    def rc_takeover(self):
        with self.lock:
            self.mode = 'POSCTL'
            self.nav = None

    def drop_map(self, seconds):
        """Markers out of view for N seconds: no aruco_map pose, navigate in aruco_map fails."""
        with self.lock:
            self.map_seen = False
            self.map_block_until = time.time() + seconds

    # --- simulation ---
    def _physics(self):
        last = time.time()
        while True:
            time.sleep(0.02)
            now = time.time()
            dt, last = now - last, now
            with self.lock:
                if not self.armed:
                    continue
                if self.mode == 'AUTO.LAND':
                    self.pos[2] = max(0.0, self.pos[2] - 0.5 * dt)
                    if self.pos[2] == 0.0:
                        self.armed, self.mode, self.nav = False, 'POSCTL', None
                    continue
                if self.mode != 'OFFBOARD':      # pilot: hover in place
                    continue
                if self.nav:
                    a, b, v, t0 = self.nav
                    L = float(np.linalg.norm(b - a))
                    self.sp = b if L < 1e-6 else a + (b - a) * min(1.0, v * (now - t0) / L)
                self.pos += (self.sp - self.pos) * min(1.0, dt * 3.0)
                if self.pos[2] > 0.35 and now > self.map_block_until:
                    self.map_seen = True
                    self.t_map = now
                    self.markers, self.t_markers = 4, now

    def _camera(self):
        from PIL import Image, ImageDraw
        while True:
            time.sleep(0.2)
            im = Image.new('RGB', (320, 240), (90, 100, 90))
            ImageDraw.Draw(im).text((10, 10), 'FAKE ROS z=%.2f' % self.pos[2], fill=(255, 255, 255))
            buf = io.BytesIO()
            im.save(buf, 'JPEG')
            with self.lock:
                self.img, self.t_img = buf.getvalue(), time.time()


class FakeThermal:
    """Registered as thermal driver 'fake'; looks at hot spots through the fake drone pose."""
    name = 'fake'
    ros = None
    spots = [(-1.9, 0.9, 9.0, 0.18), (1.4, 1.2, 7.0, 0.15), (2.6, -0.4, 6.0, 0.12), (-0.4, -0.6, 5.0, 0.15)]

    def __init__(self, c):
        pass

    def read(self):
        time.sleep(0.12)
        r = self.ros
        with r.lock:
            pose = {'x': float(r.pos[0]), 'y': float(r.pos[1]), 'z': float(r.pos[2]), 'roll': 0.0, 'pitch': 0.0, 'yaw': r.yaw}
        if pose['z'] < 0.1:
            return np.full((24, 32), 20.0, np.float32), 'C'
        pts, ok, _ = project(32, 24, 55, 35, pose, 0.0)
        T = 20.0 + np.random.normal(0, 0.15, (24, 32))
        for x, y, dT, rad in self.spots:
            T += dT * np.exp(-((pts[..., 0] - x) ** 2 + (pts[..., 1] - y) ** 2) / (2 * rad ** 2))
        return T.astype(np.float32), 'C'

    def close(self):
        pass


thermal_drivers.DRIVERS['fake'] = FakeThermal
