"""Simulated drones. They connect to the hub over TCP with the same protocol
as board/agent.py, so the server code path is identical to real flights.
Only used with --sim; the hub then listens on 127.0.0.1 only."""
import asyncio
import io
import math
import random
import time
import numpy as np
from board.protocol import pack, recv_async
from . import clock
from .thermal import project
from .vision import SIM_COLORS

RGB_HFOV = 70.0
FAULT_NAMES = {'link': 'потеря связи', 'map': 'потеря меток', 'reject_goto': 'борт отклоняет команды', 'wind': 'снос ветром',
               'jump': 'скачок позиции', 'latency': 'задержка связи', 'thermal': 'отказ тепловизора', 'camera': 'отказ камеры',
               'rc': 'перехват пультом', 'rc_release': 'пилот сажает', 'battery': 'низкая батарея', 'takeoff_refuse': 'отказ взлёта',
               'crash': 'отказ мотора', 'disarm': 'внезапная посадка', 'clear': 'снять отказы'}


class World:
    def __init__(self, cfg, field):
        self.cfg = cfg
        self.markers = field['markers']
        s = cfg['sim']
        self.ambient = s['ambient_c']
        self.objects = list(s['objects'])
        # trees: dark spiky crowns in RGB; an infected one carries heated white marks
        self.trees = s.get('trees', [])
        for tr in self.trees:
            if tr.get('infected'):
                for k, (mx, my) in enumerate(self.tree_marks(tr)):
                    self.objects.append({'x': mx, 'y': my, 'z': tr.get('h', 1.6) * 0.6, 'dT': tr.get('dT', 8.0),
                                         'r': 0.07, 'cls': None})
        self.drones = {}  # id -> SimDrone

    @staticmethod
    def tree_marks(tr):
        r = tr['r'] * 0.55
        return [(tr['x'] + r * math.cos(a), tr['y'] + r * math.sin(a)) for a in (0.6, 3.4)]

    def inject(self, drone_id, fault, seconds=None, value=None):
        """Fault injection for tests and the UI panel. Returns a description."""
        d = self.drones[drone_id]
        now = clock.now()
        timed = {'link': 8.0, 'map': 4.0, 'reject_goto': 3.0, 'wind': 6.0, 'jump': 0.3, 'latency': 10.0, 'thermal': 1e9, 'camera': 1e9}
        if fault in timed:
            dur = float(seconds) if seconds is not None else timed[fault]
            entry = {'until': now + dur}
            if fault == 'wind':
                entry['value'] = value if value else [0.35, 0.1]
            elif fault == 'jump':
                entry['value'] = float(value) if value else 1.5
            elif fault == 'latency':
                entry['value'] = float(value) if value else 1.2
            elif fault == 'map' and value == 'drift':
                entry['value'] = 'drift'       # like the real Technic: height estimate runs away without markers
            d.f[fault] = entry
            return '%s: %s на %.1f с' % (drone_id, FAULT_NAMES[fault], dur)
        if fault == 'rc':
            d.manual, d.mode, d.nav = True, 'POSCTL', None
            return '%s: пилот взял управление' % drone_id
        if fault == 'rc_release':
            d.manual = False
            if d.armed:
                d.mode = 'AUTO.LAND'
            return '%s: пилот сажает вручную' % drone_id
        if fault == 'battery':
            d.voltage = float(value) if value else 3.33 * 6
            return '%s: батарея %.2f В' % (drone_id, d.voltage)
        if fault == 'takeoff_refuse':
            n = int(value) if value else 99            # value=1: refuse once (PX4 still settling)
            d.f['takeoff_refuse'] = n
            return '%s: взлёт будет отклонён %s' % (drone_id, '%d раз' % n if n < 99 else 'всегда')
        if fault == 'crash':
            d.f['crash'] = True
            return '%s: отказ мотора' % drone_id
        if fault == 'disarm':
            if d.armed:
                d.mode = 'AUTO.LAND'
            return '%s: внезапная посадка' % drone_id
        if fault == 'clear':
            d.f.clear()
            d.manual = False
            d.voltage = 25.0
            return '%s: отказы сняты' % drone_id
        raise ValueError('unknown fault ' + fault)


    def temperature(self, gx, gy):
        T = self.ambient + 0.15 * gx + np.random.normal(0, 0.15, gx.shape)
        for o in self.objects:
            r2 = (gx - o['x']) ** 2 + (gy - o['y']) ** 2
            T += o['dT'] * np.exp(-r2 / (2 * o['r'] ** 2))
        return T

    def temperature_rays(self, pose, rays):
        """Each hot object at its own height (floor objects z=0, marks on crowns higher):
        the ray of every pixel is intersected with that object's plane."""
        sh = rays.shape[:2]
        gx0 = pose['x'] + (0 - pose['z']) / np.where(rays[..., 2] < -1e-3, rays[..., 2], -1) * rays[..., 0]
        T = self.ambient + 0.15 * gx0 + np.random.normal(0, 0.15, sh)
        dz = np.where(rays[..., 2] < -1e-3, rays[..., 2], -1)
        for o in self.objects:
            zo = o.get('z', 0.0)
            if zo >= pose['z'] - 0.1:
                continue
            s = (zo - pose['z']) / dz
            gx, gy = pose['x'] + s * rays[..., 0], pose['y'] + s * rays[..., 1]
            T += o['dT'] * np.exp(-((gx - o['x']) ** 2 + (gy - o['y']) ** 2) / (2 * o['r'] ** 2))
        return T


class SimDrone:
    def __init__(self, cfg, dcfg, start, world, port):
        self.cfg, self.id, self.role = cfg, dcfg['id'], dcfg['role']
        self.color = dcfg.get('color', '#888')
        self.world, self.port = world, port
        self.R = cfg['separation']['drone_radius_m']
        self.pos = np.array([start[0], start[1], 0.0])
        self.yaw = math.radians(random.uniform(-10, 10))
        self.armed, self.mode = False, 'POSCTL'
        self.sp = self.pos.copy()
        self.nav = None
        self.yaw_sp = self.yaw
        self.map_seen, self.t_map = False, -1e9
        self.gdrift = np.zeros(2)
        self.zdrift = 0.0
        self.voltage = 25.0
        self.last_hb = clock.now()
        self.hb_ts = None
        self.neighbors, self.guard, self.guard_landed = [], 1.0, 0.66
        self.guard_active = False
        self.wd_state = None
        self.writer = None
        self.manual = False
        self.f = {}               # injected faults, see World.inject
        world.drones[self.id] = self

    def fault(self, name):
        """Active fault value or None (timed faults expire automatically)."""
        v = self.f.get(name)
        if v is None:
            return None
        if isinstance(v, dict) and 'until' in v:
            if clock.now() > v['until']:
                self.f.pop(name, None)
                return None
        return v

    # ----------------------------------------------------------- comms
    def send(self, msg, payload=b''):
        if self.writer is None or self.fault('link'):
            return
        lat = self.fault('latency')
        if lat and msg.get('type') in ('telem', 'thermal'):
            asyncio.get_event_loop().call_later(lat['value'] / clock.SCALE, self._raw_send, pack(msg, payload))
        else:
            self._raw_send(pack(msg, payload))

    def _raw_send(self, data):
        if self.writer is not None and not self.fault('link'):
            self.writer.write(data)

    async def run(self):
        while True:
            try:
                reader, self.writer = await asyncio.open_connection('127.0.0.1', self.port)
                self.send({'type': 'hello', 'id': self.id, 'agent': 'SIM',
                           'caps': {'thermal': self.role == 'thermal', 'thermal_driver': 'sim',
                                    'rgb_geom': self.rgb_geom(320, 240)}})
                self.last_hb = clock.now()
                tasks = [asyncio.ensure_future(x) for x in (self._telem(), self._rgb(), self._thermal())]
                try:
                    while True:
                        msg, _ = await recv_async(reader)
                        self._on(msg)
                finally:
                    for t in tasks:
                        t.cancel()
            except (ConnectionError, OSError, asyncio.IncompleteReadError):
                self.writer = None
                await asyncio.sleep(0.5)

    def _on(self, msg):
        typ = msg.get('type')
        if self.fault('link'):
            return
        if typ == 'hb':
            self.last_hb = clock.now()
            self.hb_ts = msg.get('ts')
            self.neighbors, self.guard = msg.get('neighbors', []), msg.get('guard', 1.0)
            self.guard_landed = msg.get('guard_landed', 0.66)
            self.wd_state = None
        elif typ == 'cmd':
            ok, err = True, None
            op = msg['op']
            if self.manual:
                ok, err = False, 'pilot has control (RC)'
            elif op == 'takeoff':
                if self.f.get('takeoff_refuse'):
                    self.f['takeoff_refuse'] -= 1
                    if self.f['takeoff_refuse'] <= 0:
                        self.f.pop('takeoff_refuse')
                    ok, err = False, 'Arming denied: preflight check failed (simulated)'
                elif not self.armed:
                    self.armed, self.mode = True, 'OFFBOARD'
                    self.sp = self.pos.copy()
                    self._navigate(self.pos + np.array([0, 0, msg['z']]), msg.get('speed', 0.5))
            elif op == 'goto':
                tgt = np.array([msg['x'], msg['y'], msg['z']])
                if self.mode != 'OFFBOARD':
                    ok, err = False, 'not OFFBOARD'
                elif not self.map_seen:
                    ok, err = False, "Can't transform from aruco_map"
                elif self.fault('reject_goto'):
                    ok, err = False, 'navigate rejected (simulated)'
                elif self._guard_blocks(tgt):
                    ok, err = False, 'neighbor guard'
                    self._navigate(self.pos.copy(), 0.5)
                else:
                    if msg.get('yaw') is not None:
                        self.yaw_sp = msg['yaw']
                    self._navigate(tgt, msg['speed'])
            elif op == 'climb':
                if self.mode == 'OFFBOARD':
                    self._navigate(self.pos + np.array([0, 0, msg['dz']]), 0.5)
                else:
                    ok, err = False, 'not OFFBOARD'
            elif op == 'hold':
                if self.mode == 'OFFBOARD':
                    self._navigate(self.pos.copy(), 0.5)
            elif op == 'land':
                if self.armed:
                    self.mode = 'AUTO.LAND'
            self.send({'type': 'ack', 'req': msg.get('req'), 'op': op, 'ok': ok, 'error': err})
        elif typ == 'photo':
            if self.fault('camera'):
                self.send({'type': 'ack', 'req': msg['req'], 'op': 'photo', 'ok': False, 'error': 'no fresh camera frame'})
            else:
                asyncio.ensure_future(self._photo(msg['req']))

    def _guard_blocks(self, tgt):
        self.guard_active = False
        for n in self.neighbors:
            dn = math.hypot(n['x'] - self.pos[0], n['y'] - self.pos[1])
            need = self.guard if n.get('flying', True) else self.guard_landed
            if dn < need * 0.97:
                after = math.hypot(n['x'] - tgt[0], n['y'] - tgt[1])
                if after < dn - 0.02:
                    self.guard_active = True
                    return True
        return False

    def _navigate(self, tgt, speed):
        start = self.sp.copy()
        self.nav = (start, np.array(tgt, float), max(speed, 0.05), clock.now())

    # ----------------------------------------------------------- physics
    def step(self, dt):
        now = clock.now()
        if self.armed:
            age = now - self.last_hb
            lk = self.cfg['link']
            if age > lk['land_after_s'] and self.wd_state != 'land' and self.mode == 'OFFBOARD':
                self.wd_state = 'land'
                self.mode = 'AUTO.LAND'
            elif age > lk['hold_after_s'] and self.wd_state is None and self.mode == 'OFFBOARD':
                self.wd_state = 'hold'
                self._navigate(self.pos.copy(), 0.5)
        if not self.armed:
            # on the ground the camera sees no markers and optical flow drifts the estimate
            if self.map_seen:
                self.gdrift += np.array([0.03, 0.02]) * dt
            return
        if self.fault('crash'):
            self.f.pop('crash')
            self.armed, self.mode, self.nav = False, 'POSCTL', None
            self.pos[2] = 0.0
            return
        if self.mode == 'AUTO.LAND':
            self.sp = np.array([self.pos[0], self.pos[1], 0.0])
            self.pos[2] = max(0.0, self.pos[2] - 0.45 * dt)
            if self.pos[2] <= 0.0:
                self.armed, self.mode, self.nav = False, 'POSCTL', None
            return
        if self.nav:
            a, b, v, t0 = self.nav
            L = float(np.linalg.norm(b - a))
            k = 1.0 if L < 1e-6 else min(1.0, v * (now - t0) / L)
            self.sp = a + (b - a) * k
        if self.manual:                      # pilot holds position (no setpoint following)
            self.sp = self.pos.copy()
        err = self.sp - self.pos
        self.pos += err * min(1.0, dt * 3.0) + np.random.normal(0, 0.002, 3) * (self.pos[2] > 0.05)
        wind = self.fault('wind')
        if wind:
            self.pos[0] += wind['value'][0] * dt
            self.pos[1] += wind['value'][1] * dt
        dy = (self.yaw_sp - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += max(-0.8 * dt, min(0.8 * dt, dy))
        self.voltage -= 0.0025 * dt
        # on the ground the camera sees no markers and optical flow drifts the estimate
        # (as on the real Technic): the reported position wanders until markers are seen again
        mf = self.fault('map')
        if mf and isinstance(mf, dict) and mf.get('value') == 'drift':
            self.zdrift += 0.5 * dt            # estimate says "rising"; holding it the drone really sinks
            self.pos[2] = max(0.0, self.pos[2] - 0.5 * dt)
        elif not mf:
            self.zdrift = 0.0
        if self.pos[2] > 0.35 and not self.fault('map'):
            self.map_seen, self.t_map = True, now
            self.gdrift[:] = 0.0
        elif self.fault('map'):
            self.map_seen = False

    def pose(self):
        return {'x': float(self.pos[0]), 'y': float(self.pos[1]), 'z': float(self.pos[2]),
                'roll': random.gauss(0, 0.008), 'pitch': random.gauss(0, 0.008), 'yaw': self.yaw}

    async def _telem(self):
        while True:
            await clock.sleep(0.1)
            now = clock.now()
            p = self.pose()
            pub = {'type': 'telem', 'armed': self.armed, 'mode': self.mode, 'connected': True,
                   'voltage': round(self.voltage, 2), 'cell_voltage': round(self.voltage / 6, 3),
                   'map_age': round(now - self.t_map, 2) if self.t_map > -1e8 else 99.0,
                   'markers': self._visible_markers(), 'manual': self.manual, 'guard': self.guard_active, 'hb_ts': self.hb_ts,
                   'roll': p['roll'], 'pitch': p['pitch'], 'yaw': p['yaw']}
            jump = self.fault('jump')
            if self.map_seen or self.zdrift:
                pub.update(x=p['x'] + self.gdrift[0] + random.gauss(0, 0.01) + (jump['value'] if jump else 0.0),
                           y=p['y'] + self.gdrift[1] + random.gauss(0, 0.01), z=p['z'] + self.zdrift)
            else:
                pub.update(x=None, y=None, z=None)
            self.send(pub)

    def _visible_markers(self):
        z = self.pos[2]
        if z < 0.35:
            return 0
        ex = z * math.tan(math.radians(RGB_HFOV / 2))
        return sum(1 for m in self.world.markers if abs(m['x'] - self.pos[0]) < ex and abs(m['y'] - self.pos[1]) < ex * 0.75)

    async def _thermal(self):
        if self.role != 'thermal':
            return
        t = self.cfg['thermal']
        k = int(round(t['rotate_deg'] / 90.0)) % 4
        swapped = k % 2 == 1
        sw, sh = t.get('sim_size', [32, 24])        # simulated camera after server-side downsampling
        w, h = (sh, sw) if swapped else (sw, sh)
        hfov, vfov = (t['vfov_deg'], t['hfov_deg']) if swapped else (t['hfov_deg'], t['vfov_deg'])
        while True:
            await clock.sleep(1 / 6)
            if self.fault('thermal'):
                continue
            pose = self.pose()
            if pose['z'] < 0.05:
                frame = np.full((h, w), self.world.ambient) + np.random.normal(0, 0.15, (h, w))
            else:
                pts, ok, rays = project(w, h, hfov, vfov, pose, 0.0)
                frame = self.world.temperature_rays(pose, rays)
                for o in self.world.drones.values():
                    if o is self:
                        continue
                    oz = o.pos[2]
                    if oz < pose['z'] - 0.1:
                        dz = rays[..., 2]
                        s = (oz - pose['z']) / np.where(dz < -1e-3, dz, -1)
                        px, py = pose['x'] + s * rays[..., 0], pose['y'] + s * rays[..., 1]
                        frame = np.where(np.hypot(px - o.pos[0], py - o.pos[1]) < self.R, 42.0, frame)
            # undo the server-side orientation so the configured rotate/flip is exercised
            f = frame
            if t['flip_v']:
                f = f[::-1, :]
            if t['flip_h']:
                f = f[:, ::-1]
            f = np.rot90(f, k)
            f = np.ascontiguousarray(f, dtype='<f4')
            self.send({'type': 'thermal', 'w': f.shape[1], 'h': f.shape[0], 'units': 'C', 't': time.time(),
                       'pose': pose}, f.tobytes())

    @staticmethod
    def rgb_geom(W, H):
        """Camera model of render(): pinhole, image top = forward (as a board agent sends it)."""
        f = (W / 2) / math.tan(math.radians(RGB_HFOV / 2))
        return {'w': W, 'h': H, 'K': [f, 0, W / 2, 0, f, H / 2, 0, 0, 1], 'D': [], 'model': 'plumb_bob',
                'R_bc': [0, -1, 0, -1, 0, 0, 0, 0, -1], 'frame': 'sim_camera'}

    async def _rgb(self):
        loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(0.25)  # wall clock: 4 fps
            if self.fault('camera'):
                continue
            pose = self.pose()
            jpg = await loop.run_in_executor(None, self.render, 320, 240)
            self.send({'type': 'rgb', 'w': 320, 'h': 240, 't': time.time(), 'pose': pose}, jpg)

    async def _photo(self, req):
        pose = self.pose()
        jpg = await asyncio.get_event_loop().run_in_executor(None, self.render, 640, 480)
        self.send({'type': 'photo', 'req': req, 'pose': pose, 'w': 640, 'h': 480}, jpg)

    # ----------------------------------------------------------- RGB render
    def render(self, W, H):
        from PIL import Image, ImageDraw
        x, y, z, yaw = float(self.pos[0]), float(self.pos[1]), max(float(self.pos[2]), 0.12), self.yaw
        ex = z * math.tan(math.radians(RGB_HFOV / 2))
        ey = ex * H / W
        c, s = math.cos(yaw), math.sin(yaw)

        def px(wx, wy, h=0.0):
            dx, dy = wx - x, wy - y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            k = z / max(z - h, 0.05)
            return (W / 2 - by * k / ex * W / 2, H / 2 - bx * k / ey * H / 2)

        img = Image.new('RGB', (W, H), (96, 104, 92))
        d = ImageDraw.Draw(img)
        sx0, sx1, sy0, sy1 = self.cfg['survey_bounds']
        d.polygon([px(sx0, sy0), px(sx1, sy0), px(sx1, sy1), px(sx0, sy1)], fill=(188, 190, 182))
        reach = math.hypot(ex, ey) + 0.3
        for m in self.world.markers:
            if abs(m['x'] - x) > reach or abs(m['y'] - y) > reach:
                continue
            L = m['length']
            cell = L / 6
            ox, oy = m['x'] - L / 2, m['y'] - L / 2
            d.polygon([px(ox, oy), px(ox + L, oy), px(ox + L, oy + L), px(ox, oy + L)], fill=(15, 15, 15))
            rnd = random.Random(m['id'])
            for i in range(4):
                for j in range(4):
                    if rnd.random() < 0.5:
                        cx0, cy0 = ox + (1 + i) * cell, oy + (1 + j) * cell
                        d.polygon([px(cx0, cy0), px(cx0 + cell, cy0), px(cx0 + cell, cy0 + cell), px(cx0, cy0 + cell)],
                                  fill=(240, 240, 240))
        for o in self.world.objects:
            if not o.get('cls') or abs(o['x'] - x) > reach or abs(o['y'] - y) > reach:
                continue
            col, r = SIM_COLORS[o['cls']], o['r']
            if o['cls'] == 'battery':
                d.polygon([px(o['x'] - r, o['y'] - r * 0.6), px(o['x'] + r, o['y'] - r * 0.6),
                           px(o['x'] + r, o['y'] + r * 0.6), px(o['x'] - r, o['y'] + r * 0.6)], fill=col)
            elif o['cls'] == 'person':
                d.polygon([px(o['x'] + r * math.cos(a) * 0.5, o['y'] + r * math.sin(a) * 1.2)
                           for a in np.linspace(0, 2 * math.pi, 20)], fill=col)
            else:
                d.polygon([px(o['x'] + r * math.cos(a), o['y'] + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24)], fill=col)
                d.polygon([px(o['x'] + r * 0.4 * math.cos(a), o['y'] + r * 0.4 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 16)],
                          fill=(255, 220, 60))
        for tr in self.world.trees:
            if abs(tr['x'] - x) > reach + 1 or abs(tr['y'] - y) > reach + 1:
                continue
            th = tr.get('h', 1.6)
            if th >= z - 0.2:
                continue
            spikes = []
            for i in range(36):                          # spiky crown seen from above
                a = 2 * math.pi * i / 36
                rr = tr['r'] * (1.0 if i % 2 == 0 else 0.62)
                spikes.append(px(tr['x'] + rr * math.cos(a), tr['y'] + rr * math.sin(a), th * 0.45))
            d.polygon(spikes, fill=SIM_COLORS['tree'])
            if tr.get('infected'):
                for mx, my in self.world.tree_marks(tr):
                    d.polygon([px(mx + 0.06 * math.cos(a), my + 0.06 * math.sin(a), th * 0.6)
                               for a in np.linspace(0, 2 * math.pi, 12)], fill=(235, 235, 235))
        for o in self.world.drones.values():
            if o is self or o.pos[2] >= z - 0.05:
                continue
            ox, oy, oz = float(o.pos[0]), float(o.pos[1]), float(o.pos[2])
            pts = [px(ox + self.R * math.cos(a), oy + self.R * math.sin(a), oz) for a in np.linspace(0, 2 * math.pi, 20)]
            d.polygon(pts, fill=(40, 40, 46), outline=o.color)
        d.text((6, 4), 'SIM %s  z=%.2f' % (self.id, z), fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=70)
        return buf.getvalue()


async def run_sim(cfg, field, port, only=None):
    world = World(cfg, field)
    starts = cfg['sim']['starts']
    drones = [SimDrone(cfg, dc, starts[i % len(starts)], world, port) for i, dc in enumerate(cfg['drones'])
              if not only or dc['id'] in only]

    async def physics():
        last = clock.now()
        while True:
            await clock.sleep(0.05)
            now = clock.now()
            dt = min(now - last, 0.2)
            last = now
            for dr in drones:
                dr.step(dt)

    asyncio.ensure_future(physics())
    for dr in drones:
        asyncio.ensure_future(dr.run())
    return world
