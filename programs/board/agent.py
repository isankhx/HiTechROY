#!/usr/bin/env python3
"""ThermoSwarm onboard agent for SKYRIS Technic (Clover-compatible ROS1).

Connects OUT to the ground server, streams telemetry / RGB / thermal and
executes navigate / land commands. Python 3.6+, stdlib + rospy (+cv2/numpy
only for UVC thermal cameras).

  python3 agent.py --check                    # 8 s self-test, never arms
  python3 agent.py --probe-thermal            # test the thermal camera only (no ROS)
  python3 agent.py                            # run (settings from agent.json)

Safety on board (independent of the server):
  * no heartbeat from server > hold_after_s -> hover; > land_after_s -> land
  * a neighbour closer than the guard distance and we fly toward it -> hover
  * pilot switched mode on the RC -> agent stops sending any command until disarm
"""
import argparse
import json
import math
import os
import queue
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from protocol import pack, recv_blocking, DEFAULT_PORT  # noqa: E402
import thermal_drivers  # noqa: E402

VERSION = 'agent-1.0'
NAN = float('nan')

DEFAULTS = {
    'id': 'T1',
    'server': '192.168.1.100',
    'port': DEFAULT_PORT,
    'camera_topic': '/main_camera/image_raw/compressed',
    'map_pose_topic': '/aruco_map/pose',
    'markers_topic': '/aruco_detect/markers',
    'map_frame': 'aruco_map',
    'takeoff_frame': 'body',
    'srv_packages': ['technic.srv', 'clover.srv'],
    'rgb_fps': 5,
    'telem_hz': 10,
    'hold_after_s': 1.5,
    'land_after_s': 6.0,
    'max_speed': 1.0,
    'thermal': {'driver': 'none', 'device': '/dev/ttyUSB0', 'baud': 115200, 'fps': 8},
}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fin(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# --------------------------------------------------------------------- ROS
class Ros:
    def __init__(self, c):
        import rospy
        self.rospy = rospy
        rospy.init_node('thermoswarm_agent', anonymous=False, disable_signals=True)
        srv = None
        for pkg in c['srv_packages']:
            try:
                srv = __import__(pkg, fromlist=['Navigate', 'GetTelemetry'])
                break
            except ImportError:
                continue
        if srv is None:
            raise ImportError('no service package among %s' % c['srv_packages'])
        from std_srvs.srv import Trigger
        from sensor_msgs.msg import CompressedImage
        log('ROS services from', srv.__name__)
        for name in ('navigate', 'get_telemetry', 'land'):
            rospy.wait_for_service(name, timeout=15)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.land = rospy.ServiceProxy('land', Trigger)
        self.lock = threading.Lock()
        self.img, self.t_img, self.img_fmt = None, 0.0, ''
        self.t_map = 0.0
        self.markers, self.t_markers = 0, 0.0
        rospy.Subscriber(c['camera_topic'], CompressedImage, self._img, queue_size=1, buff_size=2 ** 22)
        rospy.Subscriber(c['map_pose_topic'], rospy.AnyMsg, self._map, queue_size=1)
        try:
            from aruco_pose.msg import MarkerArray
            rospy.Subscriber(c['markers_topic'], MarkerArray, self._markers, queue_size=1)
        except ImportError:
            log('aruco_pose.msg not found: marker count unavailable')

    def _img(self, m):
        with self.lock:
            self.img, self.t_img, self.img_fmt = bytes(m.data), time.time(), m.format

    def _map(self, _m):
        self.t_map = time.time()

    def _markers(self, m):
        self.markers, self.t_markers = len(m.markers), time.time()


# --------------------------------------------------------------------- agent
class Agent:
    def __init__(self, c, ros):
        self.c, self.ros = c, ros
        self.sock = None
        self.send_lock = threading.Lock()
        self.ctrl = queue.Queue()
        self.slots = {}           # latest-only: telem / rgb / thermal
        self.cv = threading.Condition()
        self.cmds = queue.Queue()
        self.goto_slot = None
        self.last_hb = 0.0
        self.neighbors, self.guard_dist, self.guard_landed, self.t_nb = [], 1.06, 0.66, 0.0
        self.telem = {}
        self.t_telem = 0.0
        self.flight = False       # we armed / took off this flight
        self.seen_armed = False
        self.manual = False
        self.wd = None            # watchdog state: None / 'hold' / 'land'
        self.target = None        # last goto target (x, y, z)
        self.guard_active = False
        self.thermal_ok = False
        self.thermal_name = c['thermal'].get('driver', 'none')

    # ---------- network ----------
    def emit(self, msg, payload=b'', slot=None):
        with self.cv:
            if slot:
                self.slots[slot] = (msg, payload)
            else:
                self.ctrl.put((msg, payload))
            self.cv.notify()

    def net_loop(self):
        while True:
            try:
                s = socket.create_connection((self.c['server'], int(self.c['port'])), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(None)
                self.sock = s
                s.sendall(pack({'type': 'hello', 'id': self.c['id'], 'agent': VERSION,
                                'caps': {'thermal': self.thermal_name != 'none', 'thermal_driver': self.thermal_name}}))
                log('connected to server %s:%s' % (self.c['server'], self.c['port']))
                t = threading.Thread(target=self.send_loop, args=(s,), daemon=True)
                t.start()
                while True:
                    msg, _ = recv_blocking(s)
                    self.on_msg(msg)
            except Exception as e:
                log('server link: %s' % e)
            finally:
                if self.sock is not None:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                self.sock = None
                with self.cv:
                    self.cv.notify_all()
            time.sleep(1)

    def send_loop(self, s):
        try:
            while self.sock is s:
                with self.cv:
                    while self.ctrl.empty() and not self.slots and self.sock is s:
                        self.cv.wait(0.5)
                    items = []
                    while not self.ctrl.empty():
                        items.append(self.ctrl.get_nowait())
                    for k in ('telem', 'thermal', 'rgb'):
                        if k in self.slots:
                            items.append(self.slots.pop(k))
                for msg, payload in items:
                    s.settimeout(3)
                    s.sendall(pack(msg, payload))
        except Exception as e:
            log('send: %s' % e)
            try:
                s.close()
            except Exception:
                pass

    def on_msg(self, msg):
        typ = msg.get('type')
        if typ == 'hb':
            self.last_hb = time.time()
            self.neighbors, self.t_nb = msg.get('neighbors', []), time.time()
            self.guard_dist = float(msg.get('guard', self.guard_dist))
            self.guard_landed = float(msg.get('guard_landed', self.guard_landed))
            if self.wd == 'hold':
                self.wd = None
                self.event('info', 'связь восстановлена')
        elif typ == 'cmd':
            if msg['op'] == 'goto':
                self.goto_slot = msg
                self.cmds.put(None)
            else:
                self.cmds.put(msg)
        elif typ == 'photo':
            with self.ros.lock:
                img, t = self.ros.img, self.ros.t_img
            if img is None or time.time() - t > 2:
                self.emit({'type': 'ack', 'req': msg.get('req'), 'op': 'photo', 'ok': False, 'error': 'no fresh camera frame'})
            else:
                self.emit({'type': 'photo', 'req': msg.get('req'), 'pose': self.pose(), 't': t}, img)

    def event(self, level, text):
        log(level.upper(), text)
        self.emit({'type': 'log', 'level': level, 'msg': text})

    # ---------- telemetry ----------
    def pose(self):
        t = self.telem
        if time.time() - self.t_telem > 0.3 or not fin(t.get('x')):
            return None
        return {k: t[k] for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}

    def telem_loop(self):
        period = 1.0 / self.c['telem_hz']
        while True:
            t0 = time.time()
            try:
                r = self.ros.get_telemetry(frame_id=self.c['map_frame'])
                armed, mode = bool(r.armed), r.mode
                if armed:
                    self.seen_armed = True
                if self.flight and armed and mode not in ('OFFBOARD', 'AUTO.LAND') and not self.manual:
                    self.manual = True
                    self.event('err', 'режим %s с пульта — агент больше не отправляет команды до посадки' % mode)
                if not armed and self.seen_armed:          # landed / disarmed after a flight
                    if self.manual:
                        self.event('info', 'разоружён: управление агентом снова разрешено')
                    self.manual = self.flight = self.seen_armed = False
                now = time.time()
                xyz = [v if fin(v) else None for v in (r.x, r.y, r.z)]
                if None in xyz:
                    xyz = [None, None, None]
                tm = {'type': 'telem', 't': now, 'x': xyz[0], 'y': xyz[1], 'z': xyz[2],
                      'roll': r.roll if fin(r.roll) else 0.0, 'pitch': r.pitch if fin(r.pitch) else 0.0,
                      'yaw': r.yaw if fin(r.yaw) else None,
                      'vx': r.vx if fin(r.vx) else None, 'vy': r.vy if fin(r.vy) else None,
                      'armed': armed, 'mode': mode, 'connected': bool(r.connected),
                      'voltage': r.voltage if fin(r.voltage) else None,
                      'cell_voltage': r.cell_voltage if fin(r.cell_voltage) else None,
                      'map_age': round(now - self.ros.t_map, 2) if self.ros.t_map else 99.0,
                      'markers': self.ros.markers if now - self.ros.t_markers < 1 else 0,
                      'manual': self.manual, 'guard': self.guard_active, 'wd': self.wd}
                self.telem, self.t_telem = tm, now
                self.emit(tm, slot='telem')
            except Exception as e:
                log('telemetry: %s' % e)
                time.sleep(0.5)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def rgb_loop(self):
        last = 0.0
        while True:
            time.sleep(1.0 / self.c['rgb_fps'])
            with self.ros.lock:
                img, t = self.ros.img, self.ros.t_img
            if img is not None and t != last and 'jpeg' in (self.ros.img_fmt or 'jpeg').lower():
                last = t
                self.emit({'type': 'rgb', 't': t}, img, slot='rgb')

    def thermal_loop(self):
        tc = self.c['thermal']
        if tc.get('driver', 'none') == 'none':
            return
        period = 1.0 / float(tc.get('fps', 8))
        while True:
            try:
                drv = thermal_drivers.open_driver(tc)
                self.event('ok', 'тепловизор %s открыт' % drv.name)
                last = 0.0
                while True:
                    rows, units = drv.read()
                    if rows is None:
                        continue
                    now = time.time()
                    if now - last < period:
                        continue
                    last = now
                    h, w, data = thermal_drivers.to_f32(rows)
                    self.thermal_ok = True
                    self.emit({'type': 'thermal', 't': now, 'w': w, 'h': h, 'units': units, 'pose': self.pose()},
                              data, slot='thermal')
            except Exception as e:
                self.thermal_ok = False
                self.event('err', 'тепловизор: %s (повтор через 2 с)' % e)
                try:
                    drv.close()
                except Exception:
                    pass
                time.sleep(2)

    # ---------- commands ----------
    def ack(self, msg, ok, err=None):
        self.emit({'type': 'ack', 'req': msg.get('req'), 'op': msg.get('op'), 'ok': ok, 'error': err})

    def nav(self, x, y, z, yaw, speed, frame, auto_arm=False):
        r = self.ros.navigate(x=x, y=y, z=z, yaw=yaw, speed=min(float(speed), self.c['max_speed']),
                              frame_id=frame, auto_arm=auto_arm)
        if not r.success:
            raise RuntimeError(r.message)

    def hold(self):
        self.target = None
        self.nav(0.0, 0.0, 0.0, NAN, 0.5, 'body')

    def cmd_loop(self):
        while True:
            msg = self.cmds.get()
            if msg is None:
                msg, self.goto_slot = self.goto_slot, None
                if msg is None:
                    continue
            op = msg.get('op')
            try:
                if self.manual:
                    raise RuntimeError('pilot has control (RC)')
                if op == 'takeoff':
                    if self.telem.get('armed'):
                        raise RuntimeError('already armed')
                    self.wd = None
                    self.flight = True
                    self.nav(0.0, 0.0, float(msg['z']), NAN, msg.get('speed', 0.5), self.c['takeoff_frame'], auto_arm=True)
                elif op == 'goto':
                    if not self.telem.get('armed'):
                        raise RuntimeError('not armed')
                    tgt = (float(msg['x']), float(msg['y']), float(msg['z']))
                    if self.guard_blocks(tgt):
                        self.hold()
                        raise RuntimeError('neighbour guard: target is toward a close drone')
                    yaw = msg.get('yaw')
                    self.nav(tgt[0], tgt[1], tgt[2], NAN if yaw is None else float(yaw), msg['speed'], self.c['map_frame'])
                    self.target = tgt
                elif op == 'hold':
                    self.hold()
                elif op == 'land':
                    self.target = None
                    r = self.ros.land()
                    if not r.success:
                        raise RuntimeError(r.message)
                else:
                    raise RuntimeError('unknown op')
                if op != 'goto':
                    self.ack(msg, True)
            except Exception as e:
                self.ack(msg, False, str(e))

    def guard_blocks(self, tgt):
        """True if a close neighbour gets even closer by moving to tgt."""
        p = self.pose()
        if p is None or time.time() - self.t_nb > 1.0:
            return False
        for n in self.neighbors:
            dn = math.hypot(n['x'] - p['x'], n['y'] - p['y'])
            need = self.guard_dist if n.get('flying', True) else self.guard_landed
            if dn < need:
                if math.hypot(n['x'] - tgt[0], n['y'] - tgt[1]) < dn - 0.02:
                    return True
        return False

    def watchdog_loop(self):
        while True:
            time.sleep(0.1)
            t = self.telem
            if not t.get('armed') or self.manual or t.get('mode') != 'OFFBOARD':
                self.guard_active = False
                continue
            age = time.time() - self.last_hb
            try:
                if age > self.c['land_after_s'] and self.wd != 'land':
                    self.wd = 'land'
                    self.event('err', 'нет связи с сервером %.1f с — ПОСАДКА' % age)
                    self.ros.land()
                elif age > self.c['hold_after_s'] and self.wd is None:
                    self.wd = 'hold'
                    self.event('warn', 'нет связи с сервером %.1f с — зависаю' % age)
                    self.hold()
                if self.target is not None and self.guard_blocks(self.target):
                    self.guard_active = True
                    self.event('warn', 'сосед ближе %.2f м — стоп' % self.guard_dist)
                    self.hold()            # clears target; flag drops on the next accepted goto
                elif self.target is not None:
                    self.guard_active = False
            except Exception as e:
                log('watchdog: %s' % e)

    def run(self):
        for fn in (self.net_loop, self.telem_loop, self.rgb_loop, self.thermal_loop, self.cmd_loop, self.watchdog_loop):
            threading.Thread(target=fn, daemon=True, name=fn.__name__).start()
        while True:
            time.sleep(1)


# --------------------------------------------------------------------- tools
def probe_thermal(c, seconds=10):
    tc = c['thermal']
    print('driver:', tc)
    drv = thermal_drivers.open_driver(tc)
    n, t0, last = 0, time.time(), None
    while time.time() - t0 < seconds:
        rows, units = drv.read()
        if rows is None:
            continue
        n += 1
        last = rows
        flat = sorted(v for r in rows for v in r)
        print('frame %3d  %dx%d  min %.2f  median %.2f  max %.2f %s' % (
            n, len(rows[0]), len(rows), flat[0], flat[len(flat) // 2], flat[-1], units), flush=True)
    print('fps: %.1f' % (n / (time.time() - t0)))
    if last is not None:
        path = os.path.join(HERE, 'thermal_probe.csv')
        with open(path, 'w') as f:
            for r in last:
                f.write(','.join('%.2f' % v for v in r) + '\n')
        print('last frame saved to', path, '(look at it: hot object must be where you expect, else set rotate/flip in config.json)')
    drv.close()


def check(c, ros):
    print('Self-test 8 s, the drone is NOT armed.')
    time.sleep(8)
    r = ros.get_telemetry(frame_id=c['map_frame'])
    print('telemetry %s: connected=%s armed=%s mode=%s x=%.2f y=%.2f z=%.2f yaw=%.2f V=%.2f' % (
        c['map_frame'], r.connected, r.armed, r.mode, r.x, r.y, r.z, r.yaw, r.voltage))
    print('map pose age: %s' % ('%.2f s' % (time.time() - ros.t_map) if ros.t_map else 'NEVER (markers not seen)'))
    print('markers visible: %d' % ros.markers)
    print('camera: %s' % ('%s, %d bytes, %.2f s old' % (ros.img_fmt, len(ros.img), time.time() - ros.t_img) if ros.img else 'NO FRAMES on ' + c['camera_topic']))
    try:
        s = socket.create_connection((c['server'], int(c['port'])), timeout=3)
        s.close()
        print('server %s:%s reachable' % (c['server'], c['port']))
    except Exception as e:
        print('server %s:%s NOT reachable: %s' % (c['server'], c['port'], e))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(HERE, 'agent.json'))
    p.add_argument('--id')
    p.add_argument('--server')
    p.add_argument('--port', type=int)
    p.add_argument('--check', action='store_true')
    p.add_argument('--probe-thermal', action='store_true')
    a = p.parse_args()
    c = json.loads(json.dumps(DEFAULTS))
    if os.path.exists(a.config):
        user = json.load(open(a.config))
        th = dict(c['thermal'], **user.pop('thermal', {}))
        c.update(user)
        c['thermal'] = th
    for k in ('id', 'server', 'port'):
        if getattr(a, k):
            c[k] = getattr(a, k)
    if a.probe_thermal:
        return probe_thermal(c)
    import fcntl
    lock = open('/tmp/thermoswarm_agent.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit('another agent is already running on this drone')
    ros = Ros(c)
    if a.check:
        return check(c, ros)
    log('ThermoSwarm agent %s, id=%s, server=%s:%s, thermal=%s' % (VERSION, c['id'], c['server'], c['port'], c['thermal'].get('driver')))
    Agent(c, ros).run()


if __name__ == '__main__':
    main()
