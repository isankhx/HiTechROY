"""Swarm mission: takeoff -> map lock -> thermal survey + RGB inspections -> home -> land."""
import asyncio
import collections
import concurrent.futures
import math
import os
import time
import numpy as np
from . import clock
from .geom import d2, clamp_in, point_poly
from .record import Recorder
from .trees import CamGeom, TreeMapper
from .thermal import ThermalMapper, public
from .thermal_nn import ThermalNN
from .traffic import Traffic, Motion

PHASES = {'idle': 'Ожидание', 'takeoff': 'Взлёт и поиск меток', 'survey': 'Обследование',
          'return': 'Возврат и посадка', 'done': 'Завершено', 'aborted': 'Аварийная посадка', 'manual': 'Ручное управление'}


def plan_survey(cfg, start):
    t, f = cfg['thermal'], cfg['flight']
    alt = t['alt_m'] - t['target_height_m']
    swapped = int(round(t['rotate_deg'] / 90.0)) % 2 == 1
    hfov, vfov = (t['vfov_deg'], t['hfov_deg']) if swapped else (t['hfov_deg'], t['vfov_deg'])
    along_x = abs(math.sin(math.radians(f.get('yaw_deg', 0)))) < 0.5
    across, along = (hfov, vfov)
    swath = 2 * alt * math.tan(math.radians(across) / 2)
    half_along = alt * math.tan(math.radians(along) / 2)
    spacing = max(0.2, swath * (1 - t['overlap']))
    R = cfg['separation']['drone_radius_m']
    sx0, sx1, sy0, sy1 = cfg['survey_bounds']
    fx0, fx1, fy0, fy1 = cfg['flight_bounds']
    if not along_x:  # lanes along Y: swap axes
        sx0, sx1, sy0, sy1 = sy0, sy1, sx0, sx1
        fx0, fx1, fy0, fy1 = fy0, fy1, fx0, fx1
    first, last = sy0 + swath / 2, sy1 - swath / 2
    if first >= last:
        ys = [(sy0 + sy1) / 2]
    else:
        n = int(math.ceil((last - first) / spacing - 1e-6)) + 1
        ys = [first + (last - first) * i / (n - 1) for i in range(n)] if n > 1 else [first]
    ys = [min(max(y, fy0 + R), fy1 - R) for y in ys]
    xa, xb = max(sx0 + half_along, fx0 + R), min(sx1 - half_along, fx1 - R)
    if xa > xb:
        xa = xb = (sx0 + sx1) / 2
    best = None
    for flip_y in (False, True):
        for flip_x in (False, True):
            lanes = ys[::-1] if flip_y else ys
            pts, fwd = [], not flip_x
            for y in lanes:
                a, b = (xa, xb) if fwd else (xb, xa)
                pts += [(a, y), (b, y)]
                fwd = not fwd
            if not along_x:
                pts = [(p[1], p[0]) for p in pts]
            dist = math.hypot(pts[0][0] - start[0], pts[0][1] - start[1])
            if best is None or dist < best[0]:
                best = (dist, pts)
    return [(x, y, t['alt_m']) for x, y in best[1]], {'swath_m': round(swath, 2), 'lanes': len(ys), 'spacing_m': round(spacing, 2)}


class Mission:
    def __init__(self, cfg, hub, detector, mode, runs_dir, broadcast):
        self.cfg, self.hub, self.detector, self.mode = cfg, hub, detector, mode
        self.runs_dir = runs_dir
        self.broadcast = broadcast
        self.events = collections.deque(maxlen=400)
        self.traffic = Traffic(cfg, self.log)
        self.mapper = ThermalMapper(cfg)
        self.trees = TreeMapper(cfg)
        self._tree_pool = concurrent.futures.ThreadPoolExecutor(1)
        self._tree_busy = {}
        self._t_tree = {}
        self._geom_warned = set()
        try:
            from .field import load_map
            self.markers = load_map(cfg['map_file'])['markers']
        except Exception:
            self.markers = []
        self.phase, self.paused = 'idle', False
        self.rec = None
        self.members = []
        self.survey_path, self.survey_info = [], {}
        self.pending = collections.deque()
        self.t_phase = self.t_pause = 0.0
        self._t_state = self._t_grid = self._t_live = self._t_trail = self._t_tel = self._t_keep = 0.0
        self.survey_done = False
        self.thermal_dead = False
        self.t_start = 0.0
        self.t_done = None
        t = cfg['thermal']
        self.nn = ThermalNN(t.get('nn_weights'), t.get('nn_conf', 0.4), t.get('nn_imgsz', 128), t.get('nn_range', (-2, 12))) \
            if t.get('detector', 'threshold') != 'threshold' else None
        self._nn_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._nn_busy = False
        hub.on_thermal = self.on_thermal
        hub.on_rgb = self.on_rgb
        hub.log = self.log

    # ------------------------------------------------------------- logging
    def log(self, level, msg):
        ev = {'t': time.strftime('%H:%M:%S'), 'level': level, 'msg': msg}
        self.events.append(ev)
        if self.rec:
            self.rec.event(dict(ev, wall=time.time()))
        print('[%s] %-4s %s' % (ev['t'], level, msg), flush=True)
        self.broadcast('log', ev)

    @property
    def drones(self):
        return list(self.hub.drones.values())

    def by_role(self, role):
        return [d for d in self.members if d.role == role]

    # ------------------------------------------------------------- thermal
    def on_thermal(self, drone, msg, payload):
        try:
            w, h = int(msg['w']), int(msg['h'])
            frame = np.frombuffer(payload, '<f4').reshape(h, w)
        except Exception as e:
            self.log('err', '%s: неверный тепловой кадр: %s' % (drone.id, e))
            return
        pose = msg.get('pose')
        if pose is None and drone.pos is not None:
            t = drone.telem
            pose = {'x': t['x'], 'y': t['y'], 'z': t['z'], 'roll': t.get('roll', 0.0),
                    'pitch': t.get('pitch', 0.0), 'yaw': t.get('yaw', 0.0)}
        active = self.phase == 'survey' and drone.motion.tag == 'survey' and drone.pos is not None \
            and abs(drone.pos[2] - self.cfg['thermal']['alt_m']) < 0.4
        idx = None
        if self.rec and self.phase != 'idle':
            idx = self.rec.thermal({'t': time.time(), 'drone': drone.id, 'w': w, 'h': h,
                                    'units': msg.get('units'), 'pose': pose, 'used': active}, payload)
        others = [(d.pos[0], d.pos[1], d.pos[2], d.flying, d.locked) for d in self.drones if d is not drone and d.pos]
        prep = self.mapper.prepare(frame)
        if active and self.nn is not None and self.nn.ok and not self._nn_busy:
            # neural detection in a worker thread; the result is merged back in the event loop
            self._nn_busy = True
            fut = asyncio.get_event_loop().run_in_executor(self._nn_pool, self.nn.detect, prep['dT'])
            t_wall = time.time()
            fut.add_done_callback(lambda f: self._after_nn(f, prep, pose, others, t_wall, idx, drone.id))
        else:
            self.mapper.process_prepared(prep, pose if active else None, others, time.time(), idx, None)
        now = time.monotonic()
        if now - self._t_live > 0.2:
            self._t_live = now
            live = self.mapper.live_payload()
            if live:
                live['drone'] = drone.id
                live['units'] = msg.get('units', '')
                self.broadcast('thermal', live)

    def _after_nn(self, fut, prep, pose, others, t_wall, idx, did):
        self._nn_busy = False
        try:
            boxes = fut.result()
        except Exception as e:
            self.log('err', 'ИК-нейросеть: %s — дальше только пороговая детекция' % e)
            self.nn.ok = False
            boxes = None
        if self.rec and boxes is not None:
            self.rec.nn({'t': t_wall, 'drone': did, 'index': idx, 'w': prep['dT'].shape[1], 'h': prep['dT'].shape[0],
                         'boxes': boxes})
        if self.phase == 'survey':
            self.mapper.process_prepared(prep, pose, others, t_wall, idx, boxes)

    # ------------------------------------------------------------- operator page (/op)
    def _op_find(self, oid):
        for lst in (self.mapper.anomalies, self.trees.trees):
            for o in lst:
                if o['id'] == oid:
                    return o, lst
        raise RuntimeError('нет объекта %s' % oid)

    def _queue_operator(self):
        """Operator objects become inspection targets like detected ones (once per mission)."""
        if self.survey_done and not self.inspectors():
            return
        for o in self.mapper.anomalies + self.trees.trees:
            if o.get('source') != 'operator' or o.get('_queued') or o['status'] != 'confirmed' or o['drone']:
                continue
            if o.get('kind') == 'tree' and (o.get('anomaly') or (not o['infected'] and not self.trees.p['inspect_healthy'])):
                continue                            # covered by its anomaly / healthy trees not photographed
            o['_queued'] = True
            if o['id'] not in self.pending:
                self.pending.append(o['id'])

    def op_add(self, kind, x, y, infected=False, label=None):
        x, y = float(x), float(y)
        x0, x1, y0, y1 = self.cfg['survey_bounds']
        if not (x0 - 0.5 <= x <= x1 + 0.5 and y0 - 0.5 <= y <= y1 + 0.5):
            raise RuntimeError('точка вне поля')
        t = time.time()
        if kind == 'tree':
            mp = self.trees
            o = mp._new(x, y, mp.p['tree_height_m'] / 2, 0.05, 0.0, t)
            o.update(crown_m=0.5, infected=bool(infected), infected_by='operator' if infected else None,
                     infected_lock=True)
        elif kind == 'anomaly':
            mp = self.mapper
            o = mp._new(x, y, 0.0, 0.05, 0.0, t)
            o['op_label'] = label or None
            o['label'] = label or None
        else:
            raise RuntimeError('неизвестный тип ' + str(kind))
        o.update(source='operator', status='confirmed', observations=0,
                 _A=np.zeros((3, 3)), _b=np.zeros(3), _c=0.0, _wsum=0.0)
        mp.next_id += 1
        mp.anomalies.append(o)
        self.log('info', 'Оператор: %s %s (%.2f; %.2f)%s' % (
            'ёлка' if kind == 'tree' else 'аномалия', o['id'], x, y,
            (' — заражена' if infected else ' — здорова') if kind == 'tree' else (' — ' + label if label else '')))
        self._link_trees()
        self.save_anomalies()
        return o['id']

    def op_update(self, oid, x=None, y=None, infected=None, label=None):
        o, _ = self._op_find(oid)
        if o['status'] == 'assigned':
            raise RuntimeError('%s сейчас осматривает %s' % (oid, o['drone']))
        what = []
        if x is not None and y is not None:
            o['x'], o['y'] = float(x), float(y)
            what.append('→ (%.2f; %.2f)' % (o['x'], o['y']))
        if infected is not None and o.get('kind') == 'tree':
            o['infected'], o['infected_lock'] = bool(infected), True
            o['infected_by'] = 'operator' if infected else None
            if not infected:
                o['anomaly'] = None
            what.append('заражена' if infected else 'здорова')
        if label is not None and o.get('kind') != 'tree':
            o['label'] = o['op_label'] = label or None
            what.append('класс «%s»' % label if label else 'класс снят')
        o['source_edit'] = 'operator'
        self.log('info', 'Оператор: %s %s' % (oid, ', '.join(what) or 'без изменений'))
        self._link_trees()
        self.save_anomalies()

    def op_delete(self, oid):
        o, lst = self._op_find(oid)
        if o['status'] == 'assigned':
            raise RuntimeError('%s сейчас осматривает %s' % (oid, o['drone']))
        lst.remove(o)
        if oid in self.pending:
            self.pending.remove(oid)
        for e in self.trees.trees:
            if e.get('anomaly') == oid:
                e['infected'], e['anomaly'] = False, None
        self.log('info', 'Оператор: %s удалён' % oid)
        self.save_anomalies()

    # ------------------------------------------------------------- trees (RGB)
    def on_rgb(self, drone, msg, jpeg):
        """RGB frame from a flying drone during the survey -> tree map."""
        P = self.trees.p
        if self.phase != 'survey' or not P['enabled'] or P['detector'] == 'off' or not drone.flying:
            return
        if drone.lock_lost or self._tree_busy.get(drone.id):
            return
        now = time.monotonic()
        if now - self._t_tree.get(drone.id, 0.0) < 1.0 / max(P['fps'], 0.1):
            return
        pose = msg.get('pose')
        if pose is None:
            if drone.pos is None or 'roll' not in drone.telem:
                return
            t = drone.telem
            pose = {'x': t['x'], 'y': t['y'], 'z': t['z'], 'roll': t.get('roll', 0.0),
                    'pitch': t.get('pitch', 0.0), 'yaw': t.get('yaw', 0.0)}
        if not self.trees.usable(pose):
            return
        geom = drone.caps.get('rgb_geom')
        if not geom and drone.id not in self._geom_warned:
            self._geom_warned.add(drone.id)
            self.log('warn', '%s: борт не прислал геометрию RGB-камеры (старый агент?) — ёлки по грубой модели, hfov %.0f°' % (
                drone.id, self.trees.hfov))
        self._t_tree[drone.id] = now
        self._tree_busy[drone.id] = True
        others = [(o.pos[0], o.pos[1], o.pos[2], o.flying) for o in self.drones if o is not drone and o.pos]
        fut = asyncio.get_event_loop().run_in_executor(self._tree_pool, self.trees.detect, jpeg)
        t_wall = time.time()

        def done(f, did=drone.id):
            self._tree_busy[did] = False
            try:
                det = f.result()
            except ImportError as e:
                if self.trees.p['enabled']:
                    self.trees.p['enabled'] = False
                    self.log('err', 'Детектор ёлок выключен: %s — на ноутбуке: .venv/bin/pip install opencv-python-headless '
                                    '(или перезапустите ./start_server.sh)' % e)
                return
            except Exception as e:
                now_m = time.monotonic()
                if now_m - getattr(self, '_t_tree_err', 0.0) > 10:
                    self._t_tree_err = now_m
                    self.log('err', 'Детектор ёлок: %s' % e)
                return
            if self.phase == 'survey':
                self.trees.process(det, pose, CamGeom(geom, self.trees.hfov), others, t_wall)
        fut.add_done_callback(done)

    def _link_trees(self):
        """Infected tree = a thermal anomaly on it. A healthy tree waiting for a photo
        that turns out to be infected leaves the queue: the anomaly inspection covers it."""
        R = self.trees.p['link_radius_m']
        not_tree = set(c.lower() for c in self.trees.p.get('not_tree_classes', ['heater']))
        # only settled positions: a 'new' anomaly may still be far from where it really is;
        # a hot spot YOLO called a heater (other teams' ceramic heaters) never infects a tree
        hot = [a for a in self.mapper.anomalies if a['status'] not in ('new', 'ignored')
               and (a.get('label') or '').lower() not in not_tree]
        for e in self.trees.trees:
            if e.get('infected_lock'):
                continue                            # the operator decided for this tree
            if e['infected'] and e.get('anomaly') and not any(a['id'] == e['anomaly'] for a in hot):
                a = self.anomaly(e['anomaly'])
                e['infected'], e['anomaly'] = False, None
                if a:
                    a.pop('tree', None)
                if e['status'] == 'infected':
                    e['status'] = 'confirmed'
                    if self.trees.p['inspect_healthy'] and e['id'] not in self.pending:
                        self.pending.append(e['id'])
                self.log('info', 'Ёлка %s: тёплое пятно рядом — %s, не заражение; ёлка здорова' % (
                    e['id'], (a or {}).get('label') or 'не ёлка'))
                self.save_anomalies()
                continue
            best = min(hot, key=lambda a: d2((a['x'], a['y']), (e['x'], e['y'])), default=None)
            inf = best is not None and d2((best['x'], best['y']), (e['x'], e['y'])) < R
            if inf and not e['infected']:
                e['infected'], e['anomaly'] = True, best['id']
                best['tree'] = e['id']
                if e['id'] in self.pending:
                    self.pending.remove(e['id'])
                if e['status'] == 'confirmed':
                    e['status'] = 'infected'
                self.log('warn', 'Ёлка %s заражена: на ней тепловая аномалия %s' % (e['id'], best['id']))
                self.save_anomalies()

    def anomaly(self, aid):
        for a in self.mapper.anomalies:
            if a['id'] == aid:
                return a
        for a in self.trees.trees:
            if a['id'] == aid:
                return a
        return None

    def save_anomalies(self):
        if self.rec:
            self.rec.save_json('anomalies.json', [dict(public(a), thermal=a.get('thermal')) for a in self.mapper.anomalies])
            self.rec.save_json('trees.json', [public(e) for e in self.trees.trees])

    # ------------------------------------------------------------- operator
    def start(self, only=None):
        if self.phase in ('takeoff', 'survey', 'return', 'manual'):
            raise RuntimeError('Миссия уже выполняется' if self.phase != 'manual' else 'Идёт ручное управление: сначала посадите дроны')
        members = [d for d in self.drones if d.connected and d.fresh and (not only or d.id in only)]
        if not any(d.role == 'thermal' for d in members):
            raise RuntimeError('Нет связи с тепловым дроном')
        flying = [d.id for d in self.drones if d.flying and (d.fresh or d.connected)]
        if flying:
            raise RuntimeError('Уже в воздухе: ' + ', '.join(flying))
        known = [d for d in members if d.pos is not None]
        for i, a in enumerate(known):
            for b in known[i + 1:]:
                if d2(a.pos, b.pos) < self.traffic.guard:
                    raise RuntimeError('%s и %s стоят слишком близко (%.2f м, нужно ≥ %.2f м между центрами)' % (
                        a.id, b.id, d2(a.pos, b.pos), self.traffic.guard))
        skipped = [d.id for d in self.drones if d not in members]
        if skipped:
            self.log('warn', 'Без связи/телеметрии, не участвуют: ' + ', '.join(skipped))
        th = [d for d in members if d.role == 'thermal'][0]
        if not th.thermal_ok:
            self.log('warn', '%s: тепловизор не отвечает — карта нагрева будет пустой' % th.id)
        if self.rec:
            self.rec.close()
        self.rec = Recorder(self.runs_dir, 'sim' if self.mode == 'sim' else '')
        self.rec.save_json('config.json', {k: v for k, v in self.cfg.items() if k != 'operator'})
        keep_a = [a for a in self.mapper.anomalies if a.get('source') == 'operator']
        keep_t = [e for e in self.trees.trees if e.get('source') == 'operator']
        self.mapper.reset()
        self.trees.reset()
        for lst, keep, mp in ((self.mapper.anomalies, keep_a, self.mapper), (self.trees.anomalies, keep_t, self.trees)):
            for o in keep:                      # operator marks stay; inspection results start over
                o.update(status='confirmed', drone=None, photos=[], detections=[], label=o.get('op_label'),
                         photo=None, photo_annotated=None, observations=0, _queued=False)
                o.pop('reach_fails', None)
                lst.append(o)
                mp.next_id = max(mp.next_id, int(o['id'][1:]) + 1)
        self.pending.clear()
        self.survey_done = False
        self.members = sorted(members, key=lambda d: -d.prio)
        stagger = self.cfg['flight']['takeoff_stagger_s']
        t0 = clock.now()
        for d in self.drones:
            d.task, d.excluded, d.lock_lost, d.last_fail, d.link_slow = None, False, False, None, False
        for i, d in enumerate(self.members):
            d.motion, d.home, d.busy, d.trail = Motion(), None, False, []
            d.task = {'kind': 'takeoff', 'stage': 'wait', 't': t0 + i * stagger, 'label': 'взлёт'}
        self.paused = False
        self.thermal_dead = False
        self.t_start, self.t_done = clock.now(), None
        self.set_phase('takeoff')
        self.log('ok', 'СТАРТ роя: %s. Запись: %s' % (', '.join(d.id for d in self.members), self.rec.dir))

    def set_phase(self, p):
        self.phase, self.t_phase = p, clock.now()
        self.log('info', 'Этап: ' + PHASES[p])

    def pause(self, on):
        if self.phase not in ('takeoff', 'survey', 'return', 'manual'):
            return
        if on == self.paused:
            return
        self.paused = on
        if on:
            self.t_pause = clock.now()
            for d in self.members:
                # a drone still climbing to the takeoff height finishes the climb (hold would strand it at 0.3 m)
                if d.controllable and not (d.task and d.task['kind'] == 'takeoff'):
                    self.hub.command(d, 'hold')
                    d.motion.reset_setpoint()
            self.log('warn', 'ПАУЗА: все зависли на месте')
        else:
            dt = clock.now() - self.t_pause
            for d in self.members:
                t = d.task or {}
                if t.get('kind') == 'takeoff':
                    t['t'] += dt
                elif t.get('kind') == 'inspect':
                    t['t0'] += dt
                    t['tv'] = t.get('tv', t['t0']) + dt
            self.log('info', 'Продолжение')

    def return_home(self):
        if self.phase in ('takeoff', 'survey', 'manual'):
            self.paused = False
            self.survey_done = True
            self.set_phase('return')

    def land_all(self):
        for d in self.drones:
            if d.connected and (d.flying or d.telem.get('armed')):
                self.hub.command(d, 'land')
                self._landing(d)
        if self.phase in ('takeoff', 'survey', 'return', 'manual'):
            self.set_phase('aborted')
        self.log('err', 'ПОСАДКА ВСЕХ на месте')

    def land_one(self, did):
        d = self.hub.drones[did]
        if not d.connected:
            raise RuntimeError('%s: нет связи' % d.id)
        if not (d.flying or d.telem.get('armed')):
            raise RuntimeError('%s уже на земле' % d.id)
        self.hub.command(d, 'land')
        self._release(d)
        self._landing(d)
        self.log('warn', '%s: посадка по команде оператора' % d.id)

    # ------------------------------------------------------------- manual control (map clicks)
    def manual(self, op, did, x=None, y=None, z=None):
        """Operator flies one drone: takeoff / goto (x, y, z) / hold / land. Moves go through the same
        step-by-step traffic manager, so separation, detours and all fault reactions stay active."""
        if did not in self.hub.drones:
            raise RuntimeError('нет дрона %s' % did)
        d = self.hub.drones[did]
        if self.phase in ('takeoff', 'survey', 'return', 'aborted'):
            raise RuntimeError('Идёт миссия — ручное управление недоступно')
        if op == 'land':
            return self.land_one(did)
        if op == 'takeoff':
            if not (d.connected and d.fresh):
                raise RuntimeError('%s: нет связи' % did)
            if d.flying:
                raise RuntimeError('%s уже в воздухе' % did)
            for o in self.drones:
                if o is not d and o.pos is not None and d.pos is not None and o.fresh and d2(o.pos, d.pos) < self.traffic.guard:
                    raise RuntimeError('%s стоит ближе %.2f м к %s' % (did, self.traffic.guard, o.id))
            if self.phase != 'manual':
                if self.rec:
                    self.rec.close()
                self.rec = Recorder(self.runs_dir, 'manual')
                self.members = []
                self.paused = False
                self.set_phase('manual')
            if d not in self.members:
                self.members.append(d)
            d.motion, d.home, d.busy, d.excluded, d.lock_lost, d.last_fail, d.link_slow = Motion(), None, False, False, False, None, False
            d.task = {'kind': 'takeoff', 'stage': 'wait', 't': clock.now(), 'label': 'взлёт',
                      'alt': float(z) if z is not None else self.cfg['flight']['takeoff_alt_m']}
            self.log('info', '%s: ручной взлёт' % did)
            return
        if self.phase != 'manual' or d not in self.members or d.excluded:
            raise RuntimeError('%s не в ручном полёте: сначала «Взлёт»' % did)
        if not d.controllable:
            raise RuntimeError('%s сейчас не управляется (%s)' % (did, d.state))
        if op == 'hold':
            self.hub.command(d, 'hold')
            d.motion.stop()
            d.motion.reset_setpoint()
            d.task = {'kind': 'manual', 'label': 'ручное · висит'}
            return
        if op == 'goto':
            R = self.traffic.R
            p = clamp_in((float(x), float(y)), self.cfg['flight_bounds'], R)
            z = min(max(float(z if z is not None else d.pos[2]), 0.6), 4.5)
            for o in self.drones:   # target must not be next to another drone (in the air or landed)
                if o is not d and o.pos is not None and d2(o.pos, p) < self.traffic.required(d, o):
                    raise RuntimeError('точка ближе %.2f м к %s' % (self.traffic.required(d, o), o.id))
            cur = d.pos
            route = [(cur[0], cur[1], z), (p[0], p[1], z)] if z > cur[2] + 0.05 else [(p[0], p[1], cur[2]), (p[0], p[1], z)]
            d.motion.set_route(route, self.cfg['flight']['speed_mps'], tag='manual')
            d.task = {'kind': 'manual', 'label': 'ручное → (%.2f; %.2f; %.1f)' % (p[0], p[1], z), 'goal': [p[0], p[1], z]}
            self.log('info', '%s → (%.2f; %.2f), высота %.1f м' % (did, p[0], p[1], z))
            return
        raise RuntimeError('неизвестная команда ' + op)

    def _manual(self, now):
        f = self.cfg['flight']
        for d in self.members:
            t = d.task or {}
            if t.get('kind') == 'takeoff':
                if t['stage'] == 'wait':
                    self.hub.command(d, 'takeoff', z=f['takeoff_alt_m'], speed=0.5)
                    t['stage'], t['t'] = 'climb', now
                elif t['stage'] in ('climb', 'retry', 'search'):
                    if self._climb(d, t, now) == 'ready':
                        d.home = self._home_point(d)
                        d.motion.reset_setpoint()
                        d.motion.set_route([(d.home[0], d.home[1], t['alt'])], f['speed_mps'], tag='manual')
                        d.task = {'kind': 'manual', 'label': 'ручное · подъём на %.1f м' % t['alt']}
                        self.log('ok', '%s: нашёл себя по меткам (%.2f; %.2f)' % (d.id, d.pos[0], d.pos[1]))
            elif t.get('kind') == 'manual' and not d.motion.moving and d.task.get('label', '').startswith('ручное →'):
                d.task['label'] = 'ручное · висит в (%.2f; %.2f)' % (d.pos[0], d.pos[1])
            elif t.get('kind') == 'home' and not d.excluded and self._home_ready(d):
                self._home_step(d, now)
        if self.members and all(d.excluded for d in self.members) and \
                all(not d.flying for d in self.members if d.fresh or d.connected):
            self.log('ok', 'Ручной полёт завершён: все на земле')
            self.phase = 'idle'

    def update_settings(self, changes, path):
        from . import settings
        if self.phase in ('takeoff', 'survey', 'return', 'aborted', 'manual'):
            raise RuntimeError('Параметры меняются только между миссиями')
        done = settings.apply(self.cfg, changes)
        if done:
            self.traffic = Traffic(self.cfg, self.log)           # separation distances
            sep = self.cfg['separation']
            self.hub.guard = 2 * sep['drone_radius_m'] + sep['min_gap_m']
            settings.save(self.cfg, path)
            self.log('ok', 'Параметры изменены: ' + ', '.join(done))
        return done

    def reset(self):
        if self.phase in ('idle', 'done', 'aborted'):
            self.mapper.reset()
            self.trees.reset()
            self.pending.clear()
            for d in self.drones:
                d.trail = []
            self.phase = 'idle'
            self.log('info', 'Карта и аномалии очищены')

    async def snapshot(self, did):
        d = self.hub.drones[did]
        msg, jpeg = await self.hub.request(d, {'type': 'photo'}, timeout=5)
        os.makedirs('dataset/raw', exist_ok=True)
        name = 'dataset/raw/%s_%s.jpg' % (d.id, time.strftime('%Y%m%d-%H%M%S'))
        with open(name, 'wb') as f:
            f.write(jpeg)
        self.log('ok', 'Кадр для датасета: ' + name)
        return name

    # ------------------------------------------------------------- precise landing
    def _home_point(self, d):
        """Home = where the drone was placed by hand. Drones stand on markers: snap the
        first fix in the air (already a bit off) to that marker's centre - the explicit
        drones[].home_marker if set, else the nearest marker within landing.snap_m."""
        L = self.cfg.get('landing', {})
        x, y = d.pos[0], d.pos[1]
        want = next((c.get('home_marker') for c in self.cfg['drones'] if c['id'] == d.id), None)
        m = None
        if want is not None:
            m = next((k for k in self.markers if k['id'] == int(want)), None)
            if m is None:
                self.log('err', '%s: home_marker %s нет на карте — дом по позиции взлёта' % (d.id, want))
        elif self.markers:
            near = min(self.markers, key=lambda k: math.hypot(k['x'] - x, k['y'] - y))
            if math.hypot(near['x'] - x, near['y'] - y) <= L.get('snap_m', 0.35):
                m = near
        if m is None:
            return (round(x, 3), round(y, 3))
        self.log('info', '%s: дом — метка %d (%.2f; %.2f), отклонение при взлёте %.0f см' % (
            d.id, m['id'], m['x'], m['y'], 100 * math.hypot(m['x'] - x, m['y'] - y)))
        return (float(m['x']), float(m['y']))

    def _home_ready(self, d):
        """May the landing sequence act now? Route legs need a controllable drone;
        the alignment over the marker continues even if markers flicker at low height."""
        t = d.task or {}
        if t.get('stage') in ('align', 'final'):
            return d.connected and d.fresh and d.flying and not d.manual and d.telem.get('mode') == 'OFFBOARD'
        return d.controllable and not d.motion.moving

    def _home_step(self, d, now):
        """travel (route home at cruise height) -> descend to landing.approach_alt_m over
        the marker -> align (hold until XY error < tol for hold_s) -> final (down to
        final_alt_m, align again: markers are still seen) -> land. A short last drop
        without markers lands within a few cm instead of drifting from 2 m."""
        L = self.cfg.get('landing', {})
        t = d.task
        st = t.get('stage', 'travel')
        hx, hy = d.home
        za, zf = L.get('approach_alt_m', 0.8), L.get('final_alt_m', 0.45)
        tol, hold = L.get('tolerance_m', 0.06), L.get('hold_s', 1.0)
        if st == 'travel':
            d.motion.set_route([(hx, hy, za)], L.get('speed_mps', 0.3), tag='home', homing=True)
            t.update(stage='descend', label='снижение над домом')
            return
        if st == 'descend':
            t.update(stage='align', t=now, ok=None, t_cmd=0.0, z=za, label='выравнивание над меткой')
            return
        p = d.pos
        err = math.hypot(p[0] - hx, p[1] - hy) if p else 9.0
        if not d.locked:
            if st == 'final' or now - t['t'] > L.get('align_timeout_s', 10.0):
                return self._touchdown(d, err, 'метки не видны')
            return
        if err < tol:
            t['ok'] = t['ok'] or now
            if now - t['ok'] >= hold:
                if st == 'align' and zf < za - 0.05:
                    t.update(stage='final', t=now, ok=None, t_cmd=0.0, z=zf, label='точная посадка')
                    self.goto(d, (hx, hy, zf), 0.2, self.traffic.yaw)
                    return
                return self._touchdown(d, err, None)
        else:
            t['ok'] = None
        if now - t['t_cmd'] > 1.0:                  # keep pushing the setpoint onto the marker
            self.goto(d, (hx, hy, t['z']), 0.2, self.traffic.yaw)
            t['t_cmd'] = now
        if now - t['t'] > L.get('align_timeout_s', 10.0):
            self._touchdown(d, err, 'не выровнялся за %.0f с' % L.get('align_timeout_s', 10.0))

    def _touchdown(self, d, err, why):
        self.hub.command(d, 'land')
        self._landing(d)
        self.log('ok' if not why else 'warn', '%s: посадка на точку старта (%.2f; %.2f), отклонение %.0f см%s' % (
            d.id, d.home[0], d.home[1], 100 * err, ' — ' + why if why else ''))

    # ------------------------------------------------------------- helpers
    def _landing(self, d):
        d.motion.stop()
        d.excluded = True
        d.busy = False
        d.task = {'kind': 'landing', 'label': 'посадка'}

    def _release(self, d):
        """Give an unfinished inspection back to the queue."""
        if d.task and d.task.get('kind') == 'inspect' and d.task.get('stage') != 'done':
            a = self.anomaly(d.task['aid'])
            if a and a['status'] == 'assigned' and a['photos']:
                self._fuse(a)                       # some views were shot: keep what we have
            elif a and a['status'] == 'assigned':
                a['status'], a['drone'] = 'confirmed', None
                self.pending.appendleft(a['id'])
        d.busy = False

    def goto(self, d, p, speed, yaw):
        self.hub.command(d, 'goto', x=round(p[0], 3), y=round(p[1], 3), z=round(p[2], 3),
                         yaw=yaw, speed=speed)

    def request_yield(self, d, spot, why):
        if d.busy or (d.task and d.task.get('kind') in ('inspect', 'home', 'landing')):
            return
        d.motion.set_route([spot], self.cfg['flight']['speed_mps'], tag='park')
        d.task = {'kind': 'park', 'label': why}

    def send_home(self, d):
        z = d.pos[2] if d.pos else self.cfg['inspect']['cruise_alt_m']
        hx, hy = d.home
        d.motion.set_route([(hx, hy, z)], self.cfg['flight']['speed_mps'], tag='home', homing=True)
        d.task = {'kind': 'home', 'label': 'домой'}
        d.busy = True

    def inspectors(self):
        """Drones that do RGB inspections: T2/T3; if none of them is (still) available,
        the thermal drone does them itself after its survey (single-drone mode / fallback)."""
        rgb = [d for d in self.by_role('rgb') if not d.excluded]
        if rgb or not self.survey_done or not self.cfg['inspect'].get('thermal_inspects_if_alone', True):
            return rgb
        return [d for d in self.by_role('thermal') if not d.excluded and
                not (d.task and d.task.get('kind') in ('home', 'landing', 'landed', 'manual'))]

    def thermal_path(self):
        th = [d for d in self.by_role('thermal') if d.controllable and d.motion.tag == 'survey']
        return self.traffic.path_ahead(th[0], self.traffic.lookahead + 0.5) if th else []

    # ------------------------------------------------------------- loop
    async def run(self):
        while True:
            try:
                self.tick()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log('err', 'Ошибка цикла миссии: %s' % e)
            await clock.sleep(0.1)

    def tick(self):
        now = clock.now()
        self._watch()
        if self.phase == 'survey':
            for keep, gone in self.mapper.consolidate():
                if gone['id'] in self.pending:
                    self.pending.remove(gone['id'])
                    if keep['status'] == 'confirmed' and keep['id'] not in self.pending and not keep['drone']:
                        self.pending.append(keep['id'])
                self.log('info', '%s и %s — один источник (по триангуляции): объединены в %s (%.2f; %.2f), высота %.2f м' % (
                    keep['id'], gone['id'], keep['id'], keep['x'], keep['y'], keep['z']))
                self.save_anomalies()
        if self.phase == 'survey':
            self._queue_operator()
            self.trees.consolidate()
            for e in self.trees.promote(time.time(), 1.0 / clock.SCALE):
                self.log('ok', 'Ёлка %s: (%.2f; %.2f), крона ≈%.2f м, наблюдений %d' % (
                    e['id'], e['x'], e['y'], e['crown_m'] or 0, e['observations']))
                self._link_trees()
                if e['infected']:
                    e['status'] = 'infected'
                elif self.trees.p['inspect_healthy']:
                    self.pending.append(e['id'])
                self.save_anomalies()
            self._link_trees()
            for a in self.mapper.promote(time.time(), 1.0 / clock.SCALE):
                # a drone whose position is stale (lost markers/link, landed somewhere nearby) may be this hot spot
                # a drone whose position is unreliable (lost markers or link in flight) may be this hot spot;
                # landed drones with a good last fix are already masked in the frame (R + 0.2 m)
                near = [o.id for o in self.drones if o.pos and (
                    (o.pos_uncertain and d2(o.pos, (a['x'], a['y'])) < self.traffic.R + o.pos_uncertain) or
                    (o.flying and not o.locked and d2(o.pos, (a['x'], a['y'])) < self.traffic.R + 0.4))]
                x0, x1, y0, y1 = self.cfg['survey_bounds']
                if not (x0 - 0.15 <= a['x'] <= x1 + 0.15 and y0 - 0.15 <= a['y'] <= y1 + 0.15):
                    a['status'] = 'outside'
                    self.log('info', 'Тепло за краем поля: %s (%.2f; %.2f), ΔT=%.0f — не осматривается (соседи, нагреватели)' % (
                        a['id'], a['x'], a['y'], a['dT_max']))
                    self.save_anomalies()
                    continue
                if near:
                    a['status'], a['label'] = 'ignored', 'дрон?'
                    self.log('warn', 'Аномалия %s (%.2f; %.2f): рядом %s с неизвестной точной позицией — возможно, это его моторы; осмотр не назначаю' % (
                        a['id'], a['x'], a['y'], ','.join(near)))
                    self.save_anomalies()
                    continue
                self.pending.append(a['id'])
                self.log('warn', 'Тепловая аномалия %s: ΔT=%.1f, (%.2f; %.2f) ±%.2f м, высота источника %.2f м, наблюдений %d' % (
                    a['id'], a['dT_max'], a['x'], a['y'], a['err'], a['z'], a['observations']))
                self.save_anomalies()
        if not self.paused:
            if self.phase == 'takeoff':
                self._takeoff(now)
            elif self.phase == 'survey':
                self._survey(now)
            elif self.phase == 'manual':
                self._manual(now)
            if self.phase in ('return', 'aborted'):
                self._return(now)
            if self.phase in ('takeoff', 'survey', 'return', 'manual'):
                self.traffic.tick(self.drones, self.goto, self.request_yield)
                if self.phase != 'manual' and now - self._t_keep > 1.0:
                    self._t_keep = now
                    idle = [d for d in self.members if d.controllable and not d.busy and not d.motion.moving
                            and (d.task is None or d.task.get('kind') == 'park')]
                    busy = [d for d in self.members if d.controllable and d.motion.moving]
                    self.traffic.keep_clear(idle, busy, self.drones, self.request_yield)
        self._labels()
        self._periodic()

    def _watch(self):
        loc = self.cfg.get('localization', {'hold_after_s': 2.0, 'land_after_s': 10.0})
        bat = self.cfg.get('battery', {'cell_warn_v': 3.5, 'cell_land_v': 3.4})
        now = clock.now()
        if self.phase in ('takeoff', 'survey') and now - self.t_start > self.cfg['flight'].get('max_mission_s', 900):
            self.log('err', 'Лимит времени миссии %.0f с — возврат и посадка' % self.cfg['flight'].get('max_mission_s', 900))
            self.return_home()
        if self.phase == 'survey' and not self.survey_done:
            th = self.by_role('thermal')
            if th and not th[0].excluded and th[0].thermal_age > 5.0 and not self.thermal_dead:
                self.thermal_dead = True
                self.log('err', '%s: нет тепловых кадров %.0f с — облёт продолжается без тепловой карты, проверьте тепловизор' % (th[0].id, th[0].thermal_age))
            elif self.thermal_dead and th and th[0].thermal_age < 1.0:
                self.thermal_dead = False
                self.log('ok', '%s: тепловые кадры снова идут' % th[0].id)
        for d in self.members:
            if d.excluded:
                if d.task and d.task.get('kind') == 'landing' and d.fresh and not d.flying:
                    d.task = {'kind': 'landed', 'label': 'на земле'}
                continue
            lost = now - d.t_telem
            cell = d.telem.get('cell_voltage') if d.fresh else None
            if cell is not None and d.flying and d.task and d.task.get('kind') != 'takeoff':
                if cell < bat['cell_land_v']:
                    if d.task.get('kind') != 'home':
                        self.log('err', '%s: батарея %.2f В/элемент — возврат домой и посадка' % (d.id, cell))
                        self._release(d)
                        if d.home is None:
                            d.home = (d.pos[0], d.pos[1])
                        self.send_home(d)
                        d.task['low_battery'] = True
                        if d.role == 'thermal':
                            self.survey_done = True
                    d.task['bat_warned'] = True
                elif cell < bat['cell_warn_v'] and not d.task.get('bat_warned'):
                    d.task['bat_warned'] = True
                    self.log('warn', '%s: батарея %.2f В/элемент — скоро посадка' % (d.id, cell))
            if lost > self.cfg['link']['land_after_s'] + 1:
                d.pos_uncertain = max(d.pos_uncertain, 0.4)   # board hovered then landed: small drift
                self.log('err', '%s: нет связи %.0f с — борт сам садится, исключён из миссии' % (d.id, lost))
                self._release(d)
                self._landing(d)
                continue
            if d.task and d.task.get('kind') == 'takeoff':
                continue
            if d.flying and d.manual or (d.fresh and d.flying and d.telem.get('mode') not in ('OFFBOARD', 'AUTO.LAND')):
                self.log('err', '%s: ручное управление с пульта (%s) — программа его больше не трогает' % (d.id, d.telem.get('mode')))
                self._release(d)
                self._landing(d)
                d.task = {'kind': 'manual', 'label': 'пульт'}
            elif d.fresh and not d.flying and self.phase in ('survey', 'return', 'manual'):
                self.log('warn', '%s: на земле (разоружён)' % d.id)
                self._release(d)
                self._landing(d)
            elif d.fresh and d.flying:
                lk = self.cfg['link']
                if d.rtt is not None and d.rtt > lk.get('slow_hold_s', 1.0) and not d.link_slow:
                    d.link_slow = True
                    self.hub.command(d, 'hold')
                    d.motion.reset_setpoint()
                    self.log('warn', '%s: задержка связи %.1f с — новые перемещения остановлены, зависаю' % (d.id, d.rtt))
                elif d.link_slow and d.rtt is not None and d.rtt < lk.get('slow_resume_s', 0.6):
                    d.link_slow = False
                    self.log('ok', '%s: задержка связи %.1f с — продолжаю' % (d.id, d.rtt))
                if (d.task or {}).get('kind') == 'home' and d.task.get('stage') in ('align', 'final'):
                    continue        # last metres over the home marker: _home_step decides itself
                age = d.telem.get('map_age', 99.0)
                if now - d.jump_t < 1.5:
                    age = max(age, loc['hold_after_s'] + 0.1)   # position jumped: treat as unreliable
                # the landing decision uses time measured here, not the age the board reports
                if d.lock_lost and now - d.lock_lost_since > loc['land_after_s'] - loc['hold_after_s']:
                    self.log('err', '%s: метки не видны %.0f с — посадка' % (d.id, now - d.lock_lost_since + loc['hold_after_s']))
                    self.hub.command(d, 'land')
                    self._release(d)
                    self._landing(d)
                elif d.lock_lost and d.pos and getattr(d, 'z_at_loss', None) is not None and \
                        abs(d.pos[2] - d.z_at_loss) > loc.get('max_blind_dz_m', 0.5):
                    # without markers PX4's height estimate runs away (height source = camera):
                    # holding "altitude" then really means sinking; land under control now
                    self.log('err', '%s: без меток оценка высоты ушла на %+.1f м за %.0f с — высота PX4 без камеры ненадёжна, '
                                    'посадка сейчас (проверьте EKF2_HGT_REF / дальномер)' % (
                                        d.id, d.pos[2] - d.z_at_loss, now - d.lock_lost_since))
                    self.hub.command(d, 'land')
                    self._release(d)
                    self._landing(d)
                elif age > loc['hold_after_s'] and not d.lock_lost:
                    d.z_at_loss = d.pos[2] if d.pos else None
                    d.lock_lost, d.lock_lost_since = True, now
                    d.pos_uncertain = max(d.pos_uncertain, 0.9)   # no markers: PX4 may drift
                    self.hub.command(d, 'hold')
                    d.motion.reset_setpoint()
                    why = 'скачок позиции' if now - d.jump_t < 1.5 else 'метки потеряны (%.1f с)' % age
                    self.log('warn', '%s: %s — зависаю, жду' % (d.id, why))
                elif d.lock_lost and age < 1.0:
                    d.lock_lost = False
                    d.pos_uncertain = 0.0
                    self.log('ok', '%s: метки снова видны, продолжаю' % d.id)

    def _climb(self, d, t, now):
        """Takeoff after the command was sent: 'ready' (markers found in the air),
        'failed' (gave up, landing) or None (still trying).
        * board refused (PX4 not ready yet: EKF still settling, arming check) ->
          up to flight.takeoff_retries more attempts, takeoff_retry_s apart;
        * in the air but no markers for lock_timeout_s -> one climb to
          flight.search_alt_m (more markers in view) and the same wait again,
          then land."""
        f = self.cfg['flight']
        fail = d.last_fail
        if t['stage'] == 'retry':
            if now >= t['t']:
                self.hub.command(d, 'takeoff', z=f['takeoff_alt_m'], speed=0.5)
                t['stage'], t['t'] = 'climb', now
                self.log('info', '%s: повторный взлёт (попытка %d)' % (d.id, t['tries'] + 1))
            return None
        if fail and fail['op'] == 'takeoff' and fail['t'] >= t['t']:
            t['tries'] = t.get('tries', 0) + 1
            if t['tries'] <= f.get('takeoff_retries', 2) and not d.telem.get('armed'):
                self.log('warn', '%s: борт отклонил взлёт: «%s» — повтор через %.0f с' % (
                    d.id, fail['error'], f.get('takeoff_retry_s', 4.0)))
                t['stage'], t['t'] = 'retry', now + f.get('takeoff_retry_s', 4.0)
                return None
            self.log('err', '%s: взлёт отклонён бортом (%s) — не участвует' % (d.id, fail['error']))
            if d.telem.get('armed'):
                self.hub.command(d, 'land')
            self._landing(d)
            return 'failed'
        if d.locked and d.flying and d.pos and d.pos[2] > 0.4:
            return 'ready'
        if now - t['t'] > f['lock_timeout_s']:
            sa = f.get('search_alt_m', 1.6)
            if t['stage'] == 'climb' and d.flying and d.telem.get('mode') == 'OFFBOARD':
                self.log('warn', '%s: метки не видны %.0f с на %.1f м (маркеров в кадре: %s) — поднимаюсь до %.1f м' % (
                    d.id, f['lock_timeout_s'], f['takeoff_alt_m'], d.telem.get('markers', '?'), sa))
                self.hub.command(d, 'climb', dz=sa - f['takeoff_alt_m'])
                t['stage'], t['t'] = 'search', now
                return None
            self.log('err', '%s: метки не найдены (взлёт %.1f м%s, маркеров в кадре: %s, режим %s, armed=%s) — посадка' % (
                d.id, f['takeoff_alt_m'], ', поиск %.1f м' % sa if t['stage'] == 'search' else '',
                d.telem.get('markers', '?'), d.telem.get('mode'), d.telem.get('armed')))
            if d.telem.get('armed'):
                self.hub.command(d, 'land')
            self._landing(d)
            return 'failed'
        return None

    def _takeoff(self, now):
        f = self.cfg['flight']
        for d in self.members:
            t = d.task
            if not t or t['kind'] != 'takeoff':
                continue
            if t['stage'] == 'wait' and now >= t['t']:
                self.hub.command(d, 'takeoff', z=f['takeoff_alt_m'], speed=0.5)
                t['stage'], t['t'] = 'climb', now
                self.log('info', '%s: взлёт на %.1f м' % (d.id, f['takeoff_alt_m']))
            elif t['stage'] in ('climb', 'retry', 'search'):
                st = self._climb(d, t, now)
                if st == 'failed':
                    continue
                if st == 'ready':
                    d.pos_uncertain = 0.0
                    d.home = self._home_point(d)
                    d.motion.reset_setpoint()
                    d.task = {'kind': 'ready', 'label': 'карта найдена'}
                    self.log('ok', '%s: нашёл себя по меткам: (%.2f; %.2f), высота %.2f' % (d.id, d.pos[0], d.pos[1], d.pos[2]))
        if any(d.task and d.task['kind'] == 'takeoff' for d in self.members):
            return
        th = [d for d in self.by_role('thermal') if not d.excluded]
        if not th:
            self.log('err', 'Тепловой дрон не готов — возврат и посадка всех')
            self.survey_done = True
            self.set_phase('return')
            return
        th = th[0]
        self.survey_path, self.survey_info = plan_survey(self.cfg, th.home)
        self.log('info', 'Маршрут облёта: %d галсов, полоса %.2f м, шаг %.2f м' % (
            self.survey_info['lanes'], self.survey_info['swath_m'], self.survey_info['spacing_m']))
        ta = self.cfg['thermal']['alt_m']
        th.motion.set_route([(th.home[0], th.home[1], ta)] + self.survey_path, self.cfg['thermal']['speed_mps'], tag='survey')
        th.task, th.busy = {'kind': 'survey', 'label': 'тепловой облёт'}, True
        ca = self.cfg['inspect']['cruise_alt_m']
        for d in self.by_role('rgb'):
            if not d.excluded:
                d.motion.set_route([(d.home[0], d.home[1], ca)], self.cfg['flight']['speed_mps'], tag='park')
                d.task = {'kind': 'park', 'label': 'ожидание'}
        self.set_phase('survey')

    def _survey(self, now):
        th = self.by_role('thermal')[0] if self.by_role('thermal') else None
        if not self.survey_done:
            if th is None or th.excluded:
                self.survey_done = True
                self.log('err', 'Тепловой облёт прерван')
            elif th.motion.tag == 'survey' and not th.motion.moving:
                self.survey_done = True
                cov = float((self.mapper.cnt > 0).mean() * 100)
                self.log('ok', 'Тепловой облёт завершён: покрытие %.0f%%, кадров %d, аномалий %d' % (
                    cov, self.mapper.frames, len(self.mapper.anomalies)))
                if [d for d in self.by_role('rgb') if not d.excluded] or not self.cfg['inspect'].get('thermal_inspects_if_alone', True):
                    self.send_home(th)
                else:
                    th.task, th.busy = {'kind': 'park', 'label': 'осматривает сам'}, False
                    if self.pending:
                        self.log('info', '%s: фотодронов нет — осмотры выполняет сам' % th.id)
        for d in self.members:
            if d.task and d.task['kind'] == 'home' and not d.excluded and self._home_ready(d):
                self._home_step(d, now)
        self._inspections(now)
        insp = self.inspectors()
        busy = any(d.task and d.task['kind'] == 'inspect' for d in self.members)
        if self.survey_done and not busy and (not self.pending or not insp):
            for aid in self.pending:
                a = self.anomaly(aid)
                if a and a['status'] == 'confirmed':
                    a['status'] = 'not_inspected'
            self.pending.clear()
            self.set_phase('return')

    def _views(self, p, statics, a=None):
        """Photo points for one anomaly: above it (nadir), then N points on a circle of
        orbit_radius_m around it - the same object from other angles (К3-05, К3-10).
        A healthy tree gets trees.healthy_views views (default: nadir only)."""
        ic = self.cfg['inspect']
        R = self.traffic.R
        views = [(p[0], p[1], 'над целью')]
        n, r = int(ic.get('orbit_points', 0)), float(ic.get('orbit_radius_m', 0.5))
        if a is not None and a.get('kind') == 'tree':
            n = max(0, int(self.trees.p['healthy_views']) - 1)
        names = {0: 'восток', 1: 'север', 2: 'запад', 3: 'юг'}
        for k in range(n):
            ang = 2 * math.pi * k / n
            q = (p[0] + r * math.cos(ang), p[1] + r * math.sin(ang))
            q2 = clamp_in(q, self.cfg['flight_bounds'], R)
            if d2(q, q2) > 0.05 or any(d2(o.pos, q2) < self.traffic.need_static(o) for o in statics):
                continue                            # outside the field or next to a landed drone
            views.append((q2[0], q2[1], names.get(k, '%d°' % round(math.degrees(ang))) if n == 4 else '%d°' % round(math.degrees(ang))))
        return views

    def _start_view(self, d, t, now):
        ic = self.cfg['inspect']
        x, y, _ = t['views'][t['vi']]
        route = []
        if t['vi'] == 0 and abs(d.pos[2] - ic['cruise_alt_m']) > 0.15:
            route.append((d.pos[0], d.pos[1], ic['cruise_alt_m']))
        if t['vi'] == 0:
            route += [(x, y, ic['cruise_alt_m']), (x, y, ic['photo_alt_m'])]
        else:
            route += [(x, y, ic['photo_alt_m'])]   # orbit at photo height
        d.motion.set_route(route, self.cfg['flight']['speed_mps'], tag='inspect')
        t['stage'], t['tv'] = 'fly', now

    def _inspections(self, now):
        ic = self.cfg['inspect']
        for d in self.inspectors():
            t = d.task
            if d.excluded or not d.controllable:
                continue
            if t and t['kind'] == 'inspect':
                a = self.anomaly(t['aid'])
                limit = ic['task_timeout_s'] if t['vi'] == 0 else ic['task_timeout_s'] / 3
                if now - t['tv'] > limit and t['stage'] in ('fly', 'settle'):
                    if t['vi'] == 0:
                        a['reach_fails'] = a.get('reach_fails', 0) + 1
                        if a['reach_fails'] == 1:
                            # usually a traffic jam (drones yielding to each other): retry at the end of the queue
                            self.log('warn', '%s: %s недоступна за %.0f с — попробую позже' % (d.id, a['id'], limit))
                            a['status'], a['drone'] = 'confirmed', None
                            self.pending.append(a['id'])
                        else:
                            self.log('err', '%s: %s недоступна за %.0f с повторно — пропуск' % (d.id, a['id'], limit))
                            a['status'] = 'unreachable'
                        t['xy'] = (d.pos[0], d.pos[1])     # wait where it is, not at the unreachable point
                        self._after_inspect(d, t)
                    else:
                        self.log('warn', '%s: ракурс «%s» для %s недоступен — следующий' % (d.id, t['views'][t['vi']][2], a['id']))
                        t['stage'] = 'next'
                elif t['stage'] == 'fly' and t['vi'] == 0 and d2(t['axy'], (a['x'], a['y'])) > 0.15 and \
                        d2(d.pos, (a['x'], a['y'])) > 0.3:
                    # the thermal estimate was refined while we were flying: retarget
                    statics = [o for o in self.drones if o.pos is not None and not o.controllable]
                    p = self.photo_point(a, statics) or t['xy']
                    t['xy'], t['axy'] = p, (a['x'], a['y'])
                    t['views'] = self._views(p, statics, a)
                    t['zone'] = [round(p[0], 2), round(p[1], 2), ic.get('orbit_radius_m', 0.5)]
                    self._start_view(d, t, now)
                    self.log('info', '%s: цель %s уточнена → (%.2f; %.2f)' % (d.id, a['id'], p[0], p[1]))
                elif t['stage'] == 'fly' and not d.motion.moving:
                    vx, vy, _ = t['views'][t['vi']]
                    if d2(d.pos, (vx, vy)) > 0.35:
                        if t['vi'] == 0:
                            self.log('err', '%s: не добрался до %s — пропуск' % (d.id, a['id']))
                            a['status'] = 'unreachable'
                            self._after_inspect(d, t)
                        else:
                            t['stage'] = 'next'
                    else:
                        t['stage'], t['t'] = 'settle', now
                elif t['stage'] == 'settle' and now - t['t'] >= (ic['settle_s'] if t['vi'] == 0 else ic.get('orbit_settle_s', 1.0)):
                    t['stage'] = 'photo'
                    asyncio.ensure_future(self._photo(d, a, t))
                elif t['stage'] == 'next':
                    t['vi'] += 1
                    if t['vi'] < len(t['views']):
                        self._start_view(d, t, now)
                    else:
                        t['stage'] = 'done'
                if t['stage'] == 'done':
                    self._fuse(a)
                    self._after_inspect(d, t)
                continue
            if self.paused or not self.pending or d.motion.moving:
                continue
            if d.rgb_age is None or d.rgb_age > 3.0:
                if not d.cam_warned:
                    d.cam_warned = True
                    self.log('err', '%s: нет кадров с камеры — осмотры ему не назначаются' % d.id)
                if self.survey_done and not d.busy and not (d.task and d.task.get('kind') in ('home', 'landing')):
                    # nothing left for it to do; hovering idle it only gets in the others' way
                    self.log('info', '%s: без камеры, облёт закончен — домой' % d.id)
                    self.send_home(d)
                continue
            if d.cam_warned:
                d.cam_warned = False
                self.log('ok', '%s: камера снова даёт кадры' % d.id)
            # free drone: take the nearest pending anomaly that is away from the thermal path
            tp = self.thermal_path()
            statics = [o for o in self.drones if o.pos is not None and not o.controllable]
            best = None
            for aid in list(self.pending):
                a = self.anomaly(aid)
                p = self.photo_point(a, statics)
                if p is None:
                    self.pending.remove(aid)
                    a['status'] = 'unreachable'
                    q = clamp_in((a['x'], a['y']), self.cfg['flight_bounds'], self.traffic.R)
                    why = 'за границей зоны полёта' if d2(q, (a['x'], a['y'])) > 0.5 else 'рядом стоит другой дрон'
                    self.log('warn', '%s: %s — осмотр невозможен' % (aid, why))
                    continue
                if tp and point_poly(p, tp) < self.traffic.D + 0.2:
                    continue
                blockers = [o for o in self.members if o is not d and o.motion.final and d2(o.motion.final, p) < self.traffic.D]
                if blockers:
                    for o in blockers:
                        # an idle drone parked next to the target would block it forever: move it away
                        if o.controllable and not o.busy and not o.motion.moving and \
                                not (o.task and o.task.get('kind') in ('inspect', 'home', 'landing', 'survey', 'manual')):
                            spot = self.traffic.free_spot(o, self.members, [[(p[0], p[1]), (p[0] + 0.01, p[1])]])
                            if spot:
                                self.request_yield(o, spot, 'освобождает место для осмотра ' + aid)
                    continue
                dist = d2(d.pos, p)
                if best is None or dist < best[0]:
                    best = (dist, a, p)
            if best is None:
                continue
            _, a, p = best
            self.pending.remove(a['id'])
            a['status'], a['drone'] = 'assigned', d.id
            views = self._views(p, statics, a)
            d.task = {'kind': 'inspect', 'aid': a['id'], 'stage': 'fly', 't0': now, 'tv': now, 'xy': p,
                      'axy': (a['x'], a['y']), 'views': views, 'vi': 0,
                      'zone': [round(p[0], 2), round(p[1], 2), ic.get('orbit_radius_m', 0.5)],
                      'label': 'осмотр ' + a['id']}
            self._start_view(d, d.task, now)
            d.busy = True
            self.log('info', '%s → %s (%.2f; %.2f): снимок сверху + %d ракурса по кругу r=%.1f м' % (
                d.id, a['id'], p[0], p[1], len(views) - 1, ic.get('orbit_radius_m', 0.5)))

    def photo_point(self, a, statics):
        """Where to hover for the photo: above the anomaly, pushed away from landed/manual
        drones by the separation distance (the camera still sees it: ±0.7 m at 1 m height)."""
        R = self.traffic.R
        p = clamp_in((a['x'], a['y']), self.cfg['flight_bounds'], R)
        for o in statics:
            D = self.traffic.need_static(o)
            dist = d2(o.pos, p)
            if dist < D:
                if dist < 1e-6:
                    return None
                k = (D + 0.05) / dist
                p = clamp_in((o.pos[0] + (p[0] - o.pos[0]) * k, o.pos[1] + (p[1] - o.pos[1]) * k), self.cfg['flight_bounds'], R)
        if any(d2(o.pos, p) < self.traffic.need_static(o) - 0.02 for o in statics) or d2(p, (a['x'], a['y'])) > 0.5:
            return None
        return p

    def _after_inspect(self, d, t):
        p = t['xy']
        d.motion.set_route([(p[0], p[1], self.cfg['inspect']['cruise_alt_m'])], self.cfg['flight']['speed_mps'], tag='park')
        d.task = {'kind': 'park', 'label': 'ожидание'}
        d.busy = False
        self.save_anomalies()

    async def _photo(self, d, a, t):
        view = t['views'][t['vi']][2]
        try:
            msg, jpeg = await self.hub.request(d, {'type': 'photo'}, timeout=6)
            if msg.get('type') != 'photo' or not jpeg:
                raise RuntimeError(msg.get('error') or 'борт не прислал фото')
            n = len(a['photos'])
            ph = {'file': self.rec.photo('%s_%s_%d.jpg' % (a['id'], d.id, n), jpeg), 'view': view,
                  'drone': d.id, 'pose': msg.get('pose'), 't': time.time(), 'detections': [], 'annotated': None}
            dets, ann = await asyncio.get_event_loop().run_in_executor(None, self.detector.detect, jpeg)
            self._weigh(ph, dets, a, d, msg, jpeg)
            ph['detections'] = dets
            if ann:
                ph['annotated'] = self.rec.photo('%s_%s_%d_det.jpg' % (a['id'], d.id, n), ann)
            a['photos'].append(ph)
            self.log('ok', '%s: %s, ракурс «%s» — %s' % (
                d.id, a['id'], view, ', '.join('%s %.0f%%' % (x['name'], x['conf'] * 100) for x in dets) if dets
                else ('объект не распознан' if self.detector.ok else 'YOLO не загружен')))
            t['stage'] = 'next'
        except Exception as e:
            if t['vi'] > 0 or a['photos']:
                self.log('warn', '%s: фото %s, ракурс «%s» не получено (%s) — следующий' % (d.id, a['id'], view, e))
                t['stage'] = 'next'
                return
            a['retries'] = a.get('retries', 0) + 1
            if a['retries'] <= 1:
                a['status'], a['drone'] = 'confirmed', None
                self.pending.appendleft(a['id'])
                self.log('err', '%s: фото %s не получено (%s) — попробует другой дрон' % (d.id, a['id'], e))
            else:
                a['status'] = 'photo_failed'
                self.log('err', '%s: фото %s не получено повторно: %s' % (d.id, a['id'], e))
            t['stage'] = 'abort'
            self._after_inspect(d, t)

    def _weigh(self, ph, dets, a, d, msg, jpeg):
        """Weight of every detection = how close its box is to where the target itself
        projects in this photo (camera model + pose). A neighbour object in the corner of
        the frame must not become the answer for this target."""
        W, H = msg.get('w'), msg.get('h')
        if not W or not H:
            try:
                from PIL import Image
                import io
                W, H = Image.open(io.BytesIO(jpeg)).size
            except Exception:
                W = H = None
        pose = ph.get('pose')
        uv = None
        if W and pose and all(k in pose for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
            try:
                g = CamGeom(d.caps.get('rgb_geom'), self.trees.hfov)
                uv = g.project((a['x'], a['y'], a.get('z') or 0.0), pose, W, H)
            except Exception:
                uv = None
        ph['target_px'] = [round(uv[0], 1), round(uv[1], 1)] if uv else None
        for x in dets:
            if uv is None or not x.get('box'):
                x['w'] = 1.0
                continue
            x0, y0, x1, y1 = x['box']
            dx = max(x0 - uv[0], 0.0, uv[0] - x1)
            dy = max(y0 - uv[1], 0.0, uv[1] - y1)
            x['w'] = round(math.exp(-(math.hypot(dx, dy) / (0.12 * W)) ** 2), 3)

    def _fuse(self, a):
        """Result over all views: class with the largest summed confidence; how many views saw it."""
        if not a['photos']:
            if a['status'] == 'assigned':
                a['status'] = 'photo_failed'
            return
        ignore = set(self.cfg['yolo'].get('ignore_classes', ['drone']))
        score, views = {}, {}
        for ph in a['photos']:
            seen = set()
            for x in ph['detections']:
                wgt = x.get('w', 1.0)
                if wgt < 0.3:
                    continue                    # something else in the frame, not this target
                score[x['name']] = score.get(x['name'], 0.0) + x['conf'] * wgt
                if x['name'] not in seen:
                    views[x['name']] = views.get(x['name'], 0) + 1
                    seen.add(x['name'])
        real = {k: v for k, v in score.items() if k not in ignore}
        prio = [c for c in self.cfg['yolo'].get('priority_classes', ['tree_infected']) if c in real]
        if prio:
            # white marks are visible only from some sides: one confident view is enough
            label, status = max(prio, key=real.get), 'inspected'
        elif real:
            label, status = max(real, key=real.get), 'inspected'
        elif score:
            label, status = max(score, key=score.get), 'ignored'
        else:
            label, status = None, 'inspected'
        if label is None and a.get('op_label'):
            label = a['op_label']                   # nothing recognised: keep the operator's class
        best = max(a['photos'], key=lambda ph: max([x['conf'] for x in ph['detections'] if x['name'] == label] or [0]))
        a.update(label=label, status=status, label_views=views.get(label, 0), views_total=len(a['photos']),
                 detections=best['detections'], photo=best['file'], photo_annotated=best['annotated'],
                 photo_pose=best['pose'])
        self.log('ok', '%s: итог по %d ракурсам — %s' % (
            a['id'], len(a['photos']), '%s (%d из %d)' % (label, views.get(label, 0), len(a['photos'])) if label else 'объект не распознан'))
        if status == 'ignored':
            self.log('info', '%s: это другой дрон (горячие моторы), не считается находкой' % a['id'])
        if a.get('kind') == 'tree' and label and label in self.cfg['yolo'].get('priority_classes', ['tree_infected']) \
                and not a['infected']:
            a['infected'], a['infected_by'] = True, 'YOLO'
            self.log('warn', 'Ёлка %s заражена: YOLO видит отметины (%s), тепла рядом не найдено' % (a['id'], label))
        self._link_trees()

    def _return(self, now):
        for d in self.members:
            t = d.task or {}
            if not d.excluded and t.get('kind') == 'home' and t.get('stage') in ('align', 'final') and self._home_ready(d):
                self._home_step(d, now)
                continue
            if d.excluded or not d.controllable:
                continue
            if t.get('kind') == 'inspect' and t.get('stage') == 'photo':
                continue  # let the photo finish
            if t.get('kind') != 'home':
                self._release(d)
                if d.home is None:
                    d.home = (d.pos[0], d.pos[1])
                self.send_home(d)
            elif not d.motion.moving:
                self._home_step(d, now)
        auto = [d for d in self.members if not (d.task and d.task.get('kind') == 'manual')]
        if all(d.excluded for d in self.members) and all(not d.flying for d in auto if d.fresh or d.connected):
            pilots = [d.id for d in self.members if d not in auto and d.flying]
            if pilots:
                self.log('warn', 'Под управлением пилота, сажает вручную: ' + ', '.join(pilots))
            self.save_anomalies()
            n = len(self.mapper.anomalies)
            self.log('ok', 'Обследование завершено. Аномалий: %d, осмотрено: %d' % (
                n, sum(1 for a in self.mapper.anomalies if a['status'] == 'inspected')))
            tr = self.trees.trees
            if tr:
                self.log('ok', 'Ёлок: %d — заражённых %d (%s), здоровых %d' % (
                    len(tr), sum(e['infected'] for e in tr), ', '.join(e['id'] for e in tr if e['infected']) or '—',
                    sum(not e['infected'] for e in tr)))
            self.phase = 'done'
            self.t_done = clock.now()
            self.log('info', 'Этап: ' + PHASES['done'] + ' за %.0f с' % (self.t_done - self.t_start))

    def _labels(self):
        for d in self.drones:
            if not d.connected:
                d.state = 'нет связи'
            elif d.manual:
                d.state = 'пульт'
            elif d.lock_lost:
                d.state = 'нет меток · зависание'
            elif d.link_slow:
                d.state = 'задержка связи · зависание'
            elif d.task:
                d.state = d.task.get('label', d.task['kind'])
                if d.task['kind'] == 'park' and not d.motion.moving:
                    d.state = 'ожидание'
                if d.motion.blocked_by:
                    d.state += ' · ждёт ' + ','.join(d.motion.blocked_by)
            else:
                d.state = 'в воздухе' if d.flying else 'на земле'

    # ------------------------------------------------------------- publish
    def _periodic(self):
        now = time.monotonic()
        if now - self._t_state > 0.2:
            self._t_state = now
            self.broadcast('state', self.state())
        if now - self._t_grid > 1.0:
            self._t_grid = now
            self.broadcast('heat', self.mapper.grid_payload())
        if now - self._t_trail > 2.0:
            self._t_trail = now
            self.broadcast('trails', {d.id: d.trail[-1500:] for d in self.drones})
        if self.rec and now - self._t_tel > 0.5:
            self._t_tel = now
            for d in self.members:
                if d.fresh:
                    t = d.telem
                    self.rec.telemetry({'t': time.time(), 'id': d.id, 'x': t.get('x'), 'y': t.get('y'), 'z': t.get('z'),
                                        'yaw': t.get('yaw'), 'v': t.get('voltage'), 'mode': t.get('mode'),
                                        'armed': t.get('armed'), 'map_age': t.get('map_age'), 'state': d.state})

    def state(self):
        cov = float((self.mapper.cnt > 0).mean() * 100)
        return {
            'mode': self.mode, 'phase': self.phase, 'phase_name': PHASES[self.phase], 'paused': self.paused,
            'detector': self.detector.name, 'drones': [d.public() for d in self.drones],
            'thermal_detector': self.cfg['thermal'].get('detector', 'threshold') + (
                ' + ' + self.nn.name if self.nn is not None and self.nn.ok else ''),
            'mission_s': round(((self.t_done or clock.now()) - self.t_start), 1) if self.t_start else None,
            'anomalies': [public(a) for a in self.mapper.anomalies],
            'trees': [public(e) for e in self.trees.trees],
            'tree_detector': self.trees.p['detector'] if self.trees.p['enabled'] else 'off',
            'survey_path': [[round(p[0], 2), round(p[1], 2)] for p in self.survey_path],
            'survey_info': self.survey_info, 'coverage': round(cov, 1), 'frames': self.mapper.frames,
            'pending': list(self.pending), 'run_dir': self.rec.dir if self.rec else None,
            'D': round(self.traffic.D, 2), 'guard': round(self.traffic.guard, 2), 'R': self.traffic.R,
        }
