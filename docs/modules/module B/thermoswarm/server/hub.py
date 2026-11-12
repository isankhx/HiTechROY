"""TCP hub: onboard agents connect here. One connection per drone id = one
command source per vehicle (a newer connection replaces the old one)."""
import asyncio
import math
from board.protocol import pack, recv_async
from . import clock
from .traffic import Motion

FRESH_S = 1.0


class Drone:
    def __init__(self, c):
        self.id, self.name, self.role = c['id'], c.get('name', c['id']), c['role']
        self.prio, self.color = c.get('priority', 1), c.get('color', '#888')
        self.writer = None
        self.peer = None
        self.last_rx = -1e9
        self.telem = {}
        self.t_telem = -1e9
        self.last_pos = None
        self.t_last_pos = -1e9
        # last position confirmed by markers. On the ground the camera sees no markers
        # (too close) and PX4 keeps integrating optical flow, so the reported position
        # wanders; a disarmed drone cannot move, so it is shown/avoided at this fix.
        self.fix = None
        self.was_armed = None
        self.prev_locked = False   # previous telemetry sample had fresh markers
        self.home = None
        self.motion = Motion()
        self.state = 'нет связи'
        self.task = None          # dict set by mission
        self.busy = False
        self.excluded = False     # failed / landed by operator: mission no longer commands it
        self.lock_lost = False    # markers not seen: hovering, no new moves
        self.lock_lost_since = 0.0
        self.cam_warned = False
        self.pos_uncertain = 0.0    # extra radius (m) of doubt about the last position: link lost 0.4, markers lost 0.9
        self.rtt = None             # round trip server -> board -> server (s), from the echoed hb timestamp
        self.link_slow = False      # rtt too high: no new moves
        self.last_fail = None     # last rejected command {op, error, t}
        self.rgb = None
        self.rgb_t = 0.0
        self.rgb_seq = 0
        self.rgb_cond = asyncio.Condition()
        self.thermal_ok = False
        self.t_thermal = -1e9
        self.jump_t = -1e9
        self.trail = []
        self.caps = {}
        self.stats = {'rx_bytes': 0, 'rgb_frames': 0, 'thermal_frames': 0}

    @property
    def connected(self):
        return self.writer is not None and clock.now() - self.last_rx < 3.0

    @property
    def fresh(self):
        return clock.now() - self.t_telem < FRESH_S

    @property
    def pos(self):
        t = self.telem
        if self.fresh and all(isinstance(t.get(k), (int, float)) and math.isfinite(t[k]) for k in ('x', 'y', 'z')):
            if not t.get('armed') and self.fix is not None:
                return (self.fix[0], self.fix[1], 0.0)
            return (t['x'], t['y'], t['z'])
        return self.last_pos

    @property
    def rgb_age(self):
        return clock.now() - self.rgb_t if self.rgb else None

    @property
    def thermal_age(self):
        return clock.now() - self.t_thermal

    @property
    def locked(self):
        t = self.telem
        return self.fresh and t.get('x') is not None and t.get('map_age', 99) < 1.0

    @property
    def flying(self):
        return bool(self.telem.get('armed')) if self.fresh else self.last_pos is not None and self.last_pos[2] > 0.3

    @property
    def manual(self):
        return bool(self.telem.get('manual'))

    @property
    def controllable(self):
        return self.connected and self.fresh and self.flying and not self.manual and not self.excluded \
            and not self.lock_lost and not self.link_slow and self.telem.get('mode') == 'OFFBOARD'

    def public(self):
        t = self.telem
        p = self.pos
        m = self.motion
        return {
            'id': self.id, 'name': self.name, 'role': self.role, 'color': self.color, 'prio': self.prio,
            'connected': self.connected, 'link_age': round(clock.now() - self.last_rx, 2) if self.writer else None,
            'pos': [round(v, 3) for v in p] if p else None, 'yaw': t.get('yaw'),
            'armed': t.get('armed'), 'mode': t.get('mode'), 'manual': self.manual,
            'voltage': t.get('voltage'), 'cell': t.get('cell_voltage'),
            'map_age': t.get('map_age'), 'markers': t.get('markers'), 'locked': self.locked,
            'guard': t.get('guard'), 'state': self.state, 'lock_lost': self.lock_lost,
            'rtt': round(self.rtt, 2) if self.rtt is not None else None, 'link_slow': self.link_slow,
            'task': self.task.get('label') if self.task else None,
            'zone': self.task.get('zone') if self.task else None,
            'target': [round(v, 3) for v in m.sp] if m.sp and m.moving else None,
            'route': [[round(v, 2) for v in q] for q in m.route[:40]],
            'blocked_by': m.blocked_by, 'home': self.home, 'thermal': self.thermal_ok,
            'rgb_age': round(self.rgb_age, 1) if self.rgb else None,
            'thermal_age': round(self.thermal_age, 1) if self.thermal_ok else None,
        }


class Hub:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.drones = {c['id']: Drone(c) for c in cfg['drones']}
        self.on_thermal = None
        self.on_rgb = None
        self.req = 0
        self.pending = {}
        sep = cfg['separation']
        self.guard = 2 * sep['drone_radius_m'] + sep['min_gap_m']
        self.guard_landed = 2 * sep['drone_radius_m'] + 0.1

    async def start(self, host, port):
        self.server = await asyncio.start_server(self._client, host, port)
        self.log('info', 'Порт для бортов: %s:%d' % (host, port))
        asyncio.ensure_future(self._heartbeat())

    async def _client(self, reader, writer):
        peer = writer.get_extra_info('peername')
        drone = None
        try:
            msg, _ = await asyncio.wait_for(recv_async(reader), 10)
            if msg.get('type') != 'hello' or msg.get('id') not in self.drones:
                self.log('warn', 'Отклонено подключение %s: неизвестный id %r' % (peer, msg.get('id')))
                writer.close()
                return
            drone = self.drones[msg['id']]
            if drone.writer is not None:
                self.log('warn', '%s: новое подключение заменяет старое' % drone.id)
                try:
                    drone.writer.close()
                except Exception:
                    pass
            drone.writer, drone.peer, drone.last_rx = writer, peer, clock.now()
            drone.caps = msg.get('caps', {})
            drone.thermal_ok = bool(drone.caps.get('thermal'))
            drone.motion.reset_setpoint()
            self.log('ok', '%s подключён (%s, %s)%s' % (drone.id, peer[0] if peer else '?', msg.get('agent', '?'),
                                                         ', тепловизор: ' + drone.caps.get('thermal_driver', '') if drone.thermal_ok else ''))
            while True:
                msg, payload = await recv_async(reader)
                drone.last_rx = clock.now()
                drone.stats['rx_bytes'] += len(payload)
                self._dispatch(drone, msg, payload)
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.TimeoutError, OSError) as e:
            if drone:
                self.log('err', '%s: связь потеряна (%s)' % (drone.id, type(e).__name__))
        except Exception as e:
            self.log('err', 'Ошибка протокола %s: %s' % (peer, e))
        finally:
            if drone and drone.writer is writer:
                drone.writer = None
            try:
                writer.close()
            except Exception:
                pass

    def _dispatch(self, drone, msg, payload):
        typ = msg.get('type')
        if typ == 'telem':
            drone.telem = msg
            drone.t_telem = clock.now()
            if isinstance(msg.get('hb_ts'), (int, float)):
                drone.rtt = max(0.0, clock.now() - msg['hb_ts'])
            ok_xy = all(isinstance(msg.get(k), (int, float)) and math.isfinite(msg[k]) for k in ('x', 'y'))
            if ok_xy and msg.get('map_age', 99.0) < 1.0:
                drone.fix = (msg['x'], msg['y'])
            armed = bool(msg.get('armed'))
            armed_changed = drone.was_armed is not None and armed != drone.was_armed
            drone.was_armed = armed
            # a jump right after markers come back is the flow drift being corrected, not a false marker
            locked_now = msg.get('map_age', 99.0) < 1.0
            reacquired = locked_now and not drone.prev_locked
            drone.prev_locked = locked_now
            p = drone.pos
            if p and drone.fresh and msg.get('x') is not None:
                if not armed_changed and not reacquired and drone.last_pos is not None and clock.now() - drone.t_last_pos < 0.6 and \
                        math.hypot(p[0] - drone.last_pos[0], p[1] - drone.last_pos[1]) > 0.8:
                    drone.jump_t = clock.now()
                    self.log('warn', '%s: скачок позиции %.1f м за %.1f с — ложная метка?' % (
                        drone.id, math.hypot(p[0] - drone.last_pos[0], p[1] - drone.last_pos[1]), clock.now() - drone.t_last_pos))
                drone.last_pos, drone.t_last_pos = p, clock.now()
                if not drone.trail or abs(drone.trail[-1][0] - p[0]) + abs(drone.trail[-1][1] - p[1]) > 0.05:
                    drone.trail.append((round(p[0], 3), round(p[1], 3)))
                    if len(drone.trail) > 3000:
                        del drone.trail[:500]
        elif typ == 'rgb':
            drone.rgb, drone.rgb_t = payload, clock.now()
            drone.rgb_seq += 1
            drone.stats['rgb_frames'] += 1
            asyncio.ensure_future(self._notify(drone))
            if self.on_rgb:
                self.on_rgb(drone, msg, payload)
        elif typ == 'thermal':
            drone.stats['thermal_frames'] += 1
            drone.t_thermal = clock.now()
            if self.on_thermal:
                self.on_thermal(drone, msg, payload)
        elif typ in ('photo', 'ack'):
            fut = self.pending.pop(msg.get('req'), None)
            if fut and not fut.done():
                fut.set_result((msg, payload))
            if typ == 'ack' and not msg.get('ok', True):
                drone.last_fail = {'op': msg.get('op'), 'error': msg.get('error'), 't': clock.now()}
                if msg.get('op') == 'goto':
                    # the board did not move: plan the next step from where it really is, after a pause
                    drone.motion.reset_setpoint()
                    drone.motion.retry_after = clock.now() + 1.0
                self.log('err', '%s: команда %s не выполнена: %s' % (drone.id, msg.get('op'), msg.get('error')))
        elif typ == 'log':
            self.log(msg.get('level', 'info'), '%s: %s' % (drone.id, msg.get('msg')))

    async def _notify(self, drone):
        async with drone.rgb_cond:
            drone.rgb_cond.notify_all()

    def send(self, drone, msg):
        if drone.writer is None:
            return False
        try:
            drone.writer.write(pack(msg))
            return True
        except Exception as e:
            self.log('err', '%s: ошибка отправки: %s' % (drone.id, e))
            return False

    def command(self, drone, op, **kw):
        self.req += 1
        return self.send(drone, dict(type='cmd', op=op, req=self.req, **kw))

    async def request(self, drone, msg, timeout=8.0):
        self.req += 1
        msg = dict(msg, req=self.req)
        fut = asyncio.get_event_loop().create_future()
        self.pending[self.req] = fut
        if not self.send(drone, msg):
            self.pending.pop(self.req, None)
            raise ConnectionError('нет связи с ' + drone.id)
        try:
            return await asyncio.wait_for(fut, timeout / clock.SCALE)
        finally:
            self.pending.pop(msg['req'], None)

    async def _heartbeat(self):
        while True:
            await clock.sleep(0.2)
            nbs = [{'id': d.id, 'x': d.pos[0], 'y': d.pos[1], 'z': d.pos[2], 'flying': d.flying}
                   for d in self.drones.values() if d.pos is not None and d.fresh]
            for d in self.drones.values():
                if d.writer is not None:
                    self.send(d, {'type': 'hb', 'ts': clock.now(), 'guard': self.guard, 'guard_landed': self.guard_landed,
                                  'neighbors': [n for n in nbs if n['id'] != d.id]})
