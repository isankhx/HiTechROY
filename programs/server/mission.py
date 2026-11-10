"""Swarm mission: takeoff -> map lock -> thermal survey + RGB inspections -> home -> land."""
import asyncio
import collections
import math
import os
import time
import numpy as np
from . import clock
from .geom import d2, clamp_in, point_poly
from .record import Recorder
from .thermal import ThermalMapper, public
from .traffic import Traffic, Motion

PHASES = {'idle': 'Ожидание', 'takeoff': 'Взлёт и поиск меток', 'survey': 'Обследование',
          'return': 'Возврат и посадка', 'done': 'Завершено', 'aborted': 'Аварийная посадка'}


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
        hub.on_thermal = self.on_thermal
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
        self.mapper.process(frame, pose if active else None, others, time.time(), idx)
        now = time.monotonic()
        if now - self._t_live > 0.2:
            self._t_live = now
            live = self.mapper.live_payload()
            if live:
                live['drone'] = drone.id
                live['units'] = msg.get('units', '')
                self.broadcast('thermal', live)

    def anomaly(self, aid):
        for a in self.mapper.anomalies:
            if a['id'] == aid:
                return a
        return None

    def save_anomalies(self):
        if self.rec:
            self.rec.save_json('anomalies.json', [dict(public(a), thermal=a.get('thermal')) for a in self.mapper.anomalies])

    # ------------------------------------------------------------- operator
    def start(self):
        if self.phase in ('takeoff', 'survey', 'return'):
            raise RuntimeError('Миссия уже выполняется')
        members = [d for d in self.drones if d.connected and d.fresh]
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
        self.rec.save_json('config.json', self.cfg)
        self.mapper.reset()
        self.pending.clear()
        self.survey_done = False
        self.members = sorted(members, key=lambda d: -d.prio)
        stagger = self.cfg['flight']['takeoff_stagger_s']
        t0 = clock.now()
        for d in self.drones:
            d.task, d.excluded, d.lock_lost, d.last_fail = None, False, False, None
        for i, d in enumerate(self.members):
            d.motion, d.home, d.busy, d.trail = Motion(), None, False, []
            d.task = {'kind': 'takeoff', 'stage': 'wait', 't': t0 + i * stagger, 'label': 'взлёт'}
        self.paused = False
        self.thermal_dead = False
        self.t_start = clock.now()
        self.set_phase('takeoff')
        self.log('ok', 'СТАРТ роя: %s. Запись: %s' % (', '.join(d.id for d in self.members), self.rec.dir))

    def set_phase(self, p):
        self.phase, self.t_phase = p, clock.now()
        self.log('info', 'Этап: ' + PHASES[p])

    def pause(self, on):
        if self.phase not in ('takeoff', 'survey', 'return'):
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
            self.log('info', 'Продолжение')

    def return_home(self):
        if self.phase in ('takeoff', 'survey'):
            self.paused = False
            self.survey_done = True
            self.set_phase('return')

    def land_all(self):
        for d in self.drones:
            if d.connected and (d.flying or d.telem.get('armed')):
                self.hub.command(d, 'land')
                self._landing(d)
        if self.phase in ('takeoff', 'survey', 'return'):
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

    def reset(self):
        if self.phase in ('idle', 'done', 'aborted'):
            self.mapper.reset()
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
            if a and a['status'] == 'assigned':
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
            for a in self.mapper.promote(time.time(), 1.0 / clock.SCALE):
                # a drone whose position is stale (lost markers/link, landed somewhere nearby) may be this hot spot
                near = [o.id for o in self.drones if o.pos and not o.locked and
                        d2(o.pos, (a['x'], a['y'])) < self.traffic.R + (0.9 if (not o.flying or not o.connected) else 0.4)]
                if near:
                    a['status'], a['label'] = 'ignored', 'дрон?'
                    self.log('warn', 'Аномалия %s (%.2f; %.2f): рядом %s с неизвестной точной позицией — возможно, это его моторы; осмотр не назначаю' % (
                        a['id'], a['x'], a['y'], ','.join(near)))
                    self.save_anomalies()
                    continue
                self.pending.append(a['id'])
                self.log('warn', 'Тепловая аномалия %s: ΔT=%.1f, (%.2f; %.2f) ±%.2f м, наблюдений %d' % (
                    a['id'], a['dT_max'], a['x'], a['y'], a['err'], a['observations']))
                self.save_anomalies()
        if not self.paused:
            if self.phase == 'takeoff':
                self._takeoff(now)
            elif self.phase == 'survey':
                self._survey(now)
            if self.phase in ('return', 'aborted'):
                self._return(now)
            if self.phase in ('takeoff', 'survey', 'return'):
                self.traffic.tick(self.drones, self.goto, self.request_yield)
                if now - self._t_keep > 1.0:
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
            elif d.fresh and not d.flying and self.phase in ('survey', 'return'):
                self.log('warn', '%s: на земле (разоружён)' % d.id)
                self._release(d)
                self._landing(d)
            elif d.fresh and d.flying:
                age = d.telem.get('map_age', 99.0)
                if now - d.jump_t < 1.5:
                    age = max(age, loc['hold_after_s'] + 0.1)   # position jumped: treat as unreliable
                # the landing decision uses time measured here, not the age the board reports
                if d.lock_lost and now - d.lock_lost_since > loc['land_after_s'] - loc['hold_after_s']:
                    self.log('err', '%s: метки не видны %.0f с — посадка' % (d.id, now - d.lock_lost_since + loc['hold_after_s']))
                    self.hub.command(d, 'land')
                    self._release(d)
                    self._landing(d)
                elif age > loc['hold_after_s'] and not d.lock_lost:
                    d.lock_lost, d.lock_lost_since = True, now
                    self.hub.command(d, 'hold')
                    d.motion.reset_setpoint()
                    why = 'скачок позиции' if now - d.jump_t < 1.5 else 'метки потеряны (%.1f с)' % age
                    self.log('warn', '%s: %s — зависаю, жду' % (d.id, why))
                elif d.lock_lost and age < 1.0:
                    d.lock_lost = False
                    self.log('ok', '%s: метки снова видны, продолжаю' % d.id)

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
            elif t['stage'] == 'climb':
                fail = d.last_fail
                if fail and fail['op'] == 'takeoff' and fail['t'] >= t['t']:
                    self.log('err', '%s: взлёт отклонён бортом (%s) — не участвует' % (d.id, fail['error']))
                    if d.telem.get('armed'):
                        self.hub.command(d, 'land')
                    self._landing(d)
                elif d.locked and d.flying and d.pos and d.pos[2] > 0.4:
                    d.home = (round(d.pos[0], 3), round(d.pos[1], 3))
                    d.motion.reset_setpoint()
                    d.task = {'kind': 'ready', 'label': 'карта найдена'}
                    self.log('ok', '%s: нашёл себя по меткам: (%.2f; %.2f), высота %.2f' % (d.id, d.pos[0], d.pos[1], d.pos[2]))
                elif now - t['t'] > f['lock_timeout_s']:
                    self.log('err', '%s: метки не найдены за %.0f с — посадка' % (d.id, f['lock_timeout_s']))
                    self.hub.command(d, 'land')
                    self._landing(d)
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
                self.send_home(th)
        for d in self.members:
            if d.task and d.task['kind'] == 'home' and not d.excluded and d.controllable and not d.motion.moving:
                self.hub.command(d, 'land')
                self._landing(d)
                self.log('info', '%s: посадка в точке старта' % d.id)
        self._inspections(now)
        rgb_busy = any(d.task and d.task['kind'] == 'inspect' for d in self.by_role('rgb'))
        rgb_alive = [d for d in self.by_role('rgb') if not d.excluded]
        if self.survey_done and not rgb_busy and (not self.pending or not rgb_alive):
            for aid in self.pending:
                a = self.anomaly(aid)
                if a and a['status'] == 'confirmed':
                    a['status'] = 'not_inspected'
            self.pending.clear()
            self.set_phase('return')

    def _inspections(self, now):
        ic = self.cfg['inspect']
        for d in self.by_role('rgb'):
            t = d.task
            if d.excluded or not d.controllable:
                continue
            if t and t['kind'] == 'inspect':
                a = self.anomaly(t['aid'])
                if now - t['t0'] > ic['task_timeout_s'] and t['stage'] in ('fly', 'settle'):
                    self.log('err', '%s: %s недоступна за %.0f с — пропуск' % (d.id, a['id'], ic['task_timeout_s']))
                    a['status'] = 'unreachable'
                    self._after_inspect(d, t)
                elif t['stage'] == 'fly' and d2(t['axy'], (a['x'], a['y'])) > 0.15 and \
                        d2(d.pos, (a['x'], a['y'])) > 0.3:
                    # the thermal estimate was refined while we were flying: retarget
                    p = self.photo_point(a, [o for o in self.drones if o.pos is not None and not o.controllable]) or t['xy']
                    t['xy'], t['axy'] = p, (a['x'], a['y'])
                    d.motion.set_route([(p[0], p[1], ic['cruise_alt_m']), (p[0], p[1], ic['photo_alt_m'])],
                                       self.cfg['flight']['speed_mps'], tag='inspect')
                    self.log('info', '%s: цель %s уточнена → (%.2f; %.2f)' % (d.id, a['id'], p[0], p[1]))
                elif t['stage'] == 'fly' and not d.motion.moving:
                    if d2(d.pos, t['xy']) > 0.35:
                        self.log('err', '%s: не добрался до %s — пропуск' % (d.id, a['id']))
                        a['status'] = 'unreachable'
                        self._after_inspect(d, t)
                    else:
                        t['stage'], t['t'] = 'settle', now
                elif t['stage'] == 'settle' and now - t['t'] >= ic['settle_s']:
                    t['stage'] = 'photo'
                    asyncio.ensure_future(self._photo(d, a, t))
                elif t['stage'] == 'done':
                    self._after_inspect(d, t)
                continue
            if self.paused or not self.pending or d.motion.moving:
                continue
            if d.rgb_age is None or d.rgb_age > 3.0:
                if not d.cam_warned:
                    d.cam_warned = True
                    self.log('err', '%s: нет кадров с камеры — осмотры ему не назначаются' % d.id)
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
                    self.log('warn', '%s: рядом стоит другой дрон — осмотр невозможен' % aid)
                    continue
                if tp and point_poly(p, tp) < self.traffic.D + 0.2:
                    continue
                if any(o is not d and o.motion.final and d2(o.motion.final, p) < self.traffic.D for o in self.members):
                    continue
                dist = d2(d.pos, p)
                if best is None or dist < best[0]:
                    best = (dist, a, p)
            if best is None:
                continue
            _, a, p = best
            self.pending.remove(a['id'])
            a['status'], a['drone'] = 'assigned', d.id
            route = []
            if abs(d.pos[2] - ic['cruise_alt_m']) > 0.15:
                route.append((d.pos[0], d.pos[1], ic['cruise_alt_m']))
            route += [(p[0], p[1], ic['cruise_alt_m']), (p[0], p[1], ic['photo_alt_m'])]
            d.motion.set_route(route, self.cfg['flight']['speed_mps'], tag='inspect')
            d.task = {'kind': 'inspect', 'aid': a['id'], 'stage': 'fly', 't0': now, 'xy': p, 'axy': (a['x'], a['y']),
                      'label': 'осмотр ' + a['id']}
            d.busy = True
            self.log('info', '%s → %s (%.2f; %.2f)' % (d.id, a['id'], p[0], p[1]))

    def photo_point(self, a, statics):
        """Where to hover for the photo: above the anomaly, pushed away from landed/manual
        drones by the separation distance (the camera still sees it: ±0.7 m at 1 m height)."""
        R, D = self.traffic.R, self.traffic.D
        p = clamp_in((a['x'], a['y']), self.cfg['flight_bounds'], R)
        for o in statics:
            dist = d2(o.pos, p)
            if dist < D:
                if dist < 1e-6:
                    return None
                k = (D + 0.05) / dist
                p = clamp_in((o.pos[0] + (p[0] - o.pos[0]) * k, o.pos[1] + (p[1] - o.pos[1]) * k), self.cfg['flight_bounds'], R)
        if any(d2(o.pos, p) < D - 0.02 for o in statics) or d2(p, (a['x'], a['y'])) > 0.5:
            return None
        return p

    def _after_inspect(self, d, t):
        p = t['xy']
        d.motion.set_route([(p[0], p[1], self.cfg['inspect']['cruise_alt_m'])], self.cfg['flight']['speed_mps'], tag='park')
        d.task = {'kind': 'park', 'label': 'ожидание'}
        d.busy = False
        self.save_anomalies()

    async def _photo(self, d, a, t):
        try:
            msg, jpeg = await self.hub.request(d, {'type': 'photo'}, timeout=6)
            if msg.get('type') != 'photo' or not jpeg:
                raise RuntimeError(msg.get('error') or 'борт не прислал фото')
            a['photo'] = self.rec.photo('%s_%s.jpg' % (a['id'], d.id), jpeg)
            a['photo_pose'] = msg.get('pose')
            dets, ann = await asyncio.get_event_loop().run_in_executor(None, self.detector.detect, jpeg)
            if ann:
                a['photo_annotated'] = self.rec.photo('%s_%s_det.jpg' % (a['id'], d.id), ann)
            a['detections'] = dets
            ignore = set(self.cfg['yolo'].get('ignore_classes', ['drone']))
            real = [x for x in dets if x['name'] not in ignore]
            a['label'] = real[0]['name'] if real else (dets[0]['name'] if dets else None)
            a['status'] = 'ignored' if dets and not real else 'inspected'
            self.log('ok', '%s: %s сфотографирована — %s' % (
                d.id, a['id'], ', '.join('%s %.0f%%' % (x['name'], x['conf'] * 100) for x in dets) if dets
                else ('объект не распознан' if self.detector.ok else 'YOLO не загружен')))
            if a['status'] == 'ignored':
                self.log('info', '%s: это другой дрон (горячие моторы), не считается находкой' % a['id'])
        except Exception as e:
            a['retries'] = a.get('retries', 0) + 1
            if a['retries'] <= 1:
                a['status'], a['drone'] = 'confirmed', None
                self.pending.appendleft(a['id'])
                self.log('err', '%s: фото %s не получено (%s) — попробует другой дрон' % (d.id, a['id'], e))
            else:
                a['status'] = 'photo_failed'
                self.log('err', '%s: фото %s не получено повторно: %s' % (d.id, a['id'], e))
        t['stage'] = 'done'

    def _return(self, now):
        for d in self.members:
            if d.excluded or not d.controllable:
                continue
            t = d.task or {}
            if t.get('kind') == 'inspect' and t.get('stage') == 'photo':
                continue  # let the photo finish
            if t.get('kind') != 'home':
                self._release(d)
                if d.home is None:
                    d.home = (d.pos[0], d.pos[1])
                self.send_home(d)
            elif not d.motion.moving:
                self.hub.command(d, 'land')
                self._landing(d)
                self.log('info', '%s: посадка в точке старта' % d.id)
        auto = [d for d in self.members if not (d.task and d.task.get('kind') == 'manual')]
        if all(d.excluded for d in self.members) and all(not d.flying for d in auto if d.fresh or d.connected):
            pilots = [d.id for d in self.members if d not in auto and d.flying]
            if pilots:
                self.log('warn', 'Под управлением пилота, сажает вручную: ' + ', '.join(pilots))
            self.save_anomalies()
            n = len(self.mapper.anomalies)
            self.log('ok', 'Обследование завершено. Аномалий: %d, осмотрено: %d' % (
                n, sum(1 for a in self.mapper.anomalies if a['status'] == 'inspected')))
            self.phase = 'done'
            self.log('info', 'Этап: ' + PHASES['done'])

    def _labels(self):
        for d in self.drones:
            if not d.connected:
                d.state = 'нет связи'
            elif d.manual:
                d.state = 'пульт'
            elif d.lock_lost:
                d.state = 'нет меток · зависание'
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
            'anomalies': [public(a) for a in self.mapper.anomalies],
            'survey_path': [[round(p[0], 2), round(p[1], 2)] for p in self.survey_path],
            'survey_info': self.survey_info, 'coverage': round(cov, 1), 'frames': self.mapper.frames,
            'pending': list(self.pending), 'run_dir': self.rec.dir if self.rec else None,
            'D': round(self.traffic.D, 2), 'guard': round(self.traffic.guard, 2), 'R': self.traffic.R,
        }
