"""Step-reservation traffic manager.

Every flying drone moves along its route in short steps (navigate calls).
A step is granted only if the swept XY segment keeps the required centre
distance D = 2R + gap + margin from every other drone's reservation
(its position -> its current step target). Steps that increase an already
too-small distance are allowed (escape). Idle drones in the way are moved to
a free parking spot, lower-priority drones step aside, static obstacles
(landed / manual / lost drones) are bypassed with a detour point.
"""
import math
from . import clock
from .geom import d2, d3, point_seg, seg_seg, point_poly, toward, truncate, inside


class Motion:
    def __init__(self):
        self.route = []          # remaining waypoints (x, y, z)
        self.sp = None           # current step target (where the setpoint goes)
        self.sp_from = None
        self.t_cmd = 0.0
        self.speed = 0.4
        self.final = None        # last waypoint of current route
        self.arrived = True
        self.t_route_end = 0.0
        self.blocked_by = []
        self.blocked_since = None
        self.last_resolve = 0.0
        self.stuck = 0
        self.retry_after = 0.0   # no new goto before this time (board rejected the last one)
        self.homing = False      # flying to own home: may approach landed drones
        self.tag = None          # what this route is for (survey / inspect / park / home ...)

    def set_route(self, points, speed, tag=None, homing=False):
        self.route = [tuple(p) for p in points]
        self.final = self.route[-1] if self.route else None
        self.speed = speed
        self.arrived = not self.route
        self.blocked_by, self.blocked_since, self.stuck = [], None, 0
        self.homing = homing
        self.tag = tag

    def stop(self):
        """Forget the route; the drone keeps its current setpoint."""
        self.route, self.final, self.arrived = [], None, True
        self.blocked_by, self.blocked_since = [], None

    def reset_setpoint(self):
        """After hold/pause/reconnect the board setpoint is the current position."""
        self.sp = None
        if not self.route and not self.arrived and self.final is not None:
            self.route = [self.final]   # finish the interrupted last step

    @property
    def moving(self):
        return bool(self.route) or not self.arrived

    def eta(self):
        if self.sp is None or self.sp_from is None:
            return 0.0
        return self.t_cmd + d3(self.sp_from, self.sp) / max(self.speed, 0.05)


class Traffic:
    def __init__(self, cfg, log):
        s = cfg['separation']
        self.R = s['drone_radius_m']
        self.D = 2 * self.R + s['min_gap_m'] + s['position_margin_m']
        self.guard = 2 * self.R + s['min_gap_m']
        self.lookahead = s['lookahead_m']
        self.step = cfg['flight']['step_m']
        self.yaw = math.radians(cfg['flight'].get('yaw_deg', 0.0))
        self.bounds = cfg['flight_bounds']
        self.log = log

    # ---- reservations -------------------------------------------------
    def reservation(self, dr):
        """XY polyline that drone dr currently occupies."""
        p = dr.pos
        if p is None:
            return None
        if dr.controllable and dr.motion.sp is not None:
            return [p, dr.motion.sp]
        return [p]

    def path_ahead(self, dr, length=None):
        if dr.pos is None:
            return []
        pts = [dr.pos]
        if dr.motion.sp is not None and dr.motion.moving:
            pts.append(dr.motion.sp)
        pts += dr.motion.route
        return truncate(pts, self.lookahead if length is None else length)

    def required(self, a, b):
        if a.motion.homing and not b.flying:
            return 2 * self.R + 0.1
        return self.D

    def conflicts(self, dr, target, drones):
        """Drones that block the step pos -> target."""
        out = []
        p = dr.pos
        for o in drones:
            if o is dr:
                continue
            res = self.reservation(o)
            if res is None:
                continue
            need = self.required(dr, o)
            if len(res) == 1:
                dist_new = point_seg(res[0], p, target)
                now = d2(p, res[0])
                after = d2(target, res[0])
            else:
                dist_new = seg_seg(p, target, res[0], res[1])
                now = point_seg(p, res[0], res[1])
                after = point_seg(target, res[0], res[1])
            if dist_new >= need:
                continue
            if now < need and after >= now - 1e-3 and dist_new >= now - 0.05:
                continue  # already too close: moving away is allowed
            out.append(o)
        return out

    # ---- free spot search ---------------------------------------------
    def free_spot(self, dr, drones, protect, z=None, radii=(0.6, 1.0, 1.4, 1.8, 2.3, 2.8, 3.5)):
        """Nearest spot clear of protected polylines and other drones."""
        p = dr.pos
        if p is None:
            return None
        z = p[2] if z is None else z
        best = None
        for r in radii:
            for k in range(16):
                a = 2 * math.pi * k / 16
                c = (p[0] + r * math.cos(a), p[1] + r * math.sin(a), z)
                if not inside(c, self.bounds, self.R):
                    continue
                if any(point_poly(c, poly) < self.D + 0.15 for poly in protect):
                    continue
                ok = True
                for o in drones:
                    if o is dr:
                        continue
                    res = self.reservation(o)
                    if res and point_poly(c, res) < self.D + 0.05:
                        ok = False
                        break
                    if o.motion.final is not None and d2(c, o.motion.final) < self.D + 0.05:
                        ok = False
                        break
                if not ok or self.conflicts(dr, c, drones):
                    continue
                score = r - 0.25 * min(min((point_poly(c, poly) for poly in protect), default=3.0), 3.0)
                if best is None or score < best[0]:
                    best = (score, c)
            if best is not None:
                return best[1]
        return None

    def detour(self, dr, target, obstacle, drones):
        """Waypoint that bypasses a static obstacle on the way to target."""
        p, c = dr.pos, obstacle.pos
        dx, dy = target[0] - p[0], target[1] - p[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return None
        nx, ny = -dy / L, dx / L
        t = max(0.0, min(1.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / (L * L)))
        q = (p[0] + dx * t, p[1] + dy * t)
        opts = []
        for s in (1, -1):
            for extra in (0.2, 0.45, 0.8):
                w = (c[0] + s * nx * (self.D + extra), c[1] + s * ny * (self.D + extra), target[2])
                if inside(w, self.bounds, self.R) and not self.conflicts(dr, w, drones):
                    opts.append((d2(w, q), w))
                    break
        return min(opts)[1] if opts else None

    # ---- main tick ----------------------------------------------------
    def tick(self, drones, send_goto, request_yield):
        now = clock.now()
        self._drones = drones
        active = [d for d in drones if d.controllable and d.pos is not None]
        active.sort(key=lambda d: -d.prio)
        for dr in active:
            m = dr.motion
            if not m.route:
                if not m.arrived:
                    ok = d2(dr.pos, m.final) < 0.18 and abs(dr.pos[2] - m.final[2]) < 0.2
                    if ok or now > m.t_route_end + 8.0:
                        m.arrived = True
                        if not ok:
                            self.log('warn', '%s: точка не достигнута точно (%.2f м), продолжаю' % (dr.id, d2(dr.pos, m.final)))
                continue
            if now < m.retry_after:
                continue
            # previous step still in progress?
            if m.sp is not None and now > m.eta() + 3.0 and d2(dr.pos, m.sp) > 0.3:
                self.log('warn', '%s: не дошёл до точки шага (%.2f м) — продолжаю от текущей позиции' % (dr.id, d2(dr.pos, m.sp)))
                m.sp = None
            if m.sp is not None and (now < m.eta() - 0.15 or d2(dr.pos, m.sp) > 0.6):
                continue
            origin = m.sp if m.sp is not None else dr.pos
            while m.route and d3(origin, m.route[0]) < 0.03:
                m.route.pop(0)
            if not m.route:
                m.t_route_end = now
                continue
            target = toward(origin, m.route[0], self.step)
            blockers = self.conflicts(dr, target, drones)
            if not blockers:
                send_goto(dr, target, m.speed, self.yaw)
                m.sp_from, m.sp, m.t_cmd = origin, target, now
                m.stuck = 0
                if d3(target, m.route[0]) < 0.03:
                    m.route.pop(0)
                    if not m.route:
                        m.t_route_end = m.eta()
                if m.blocked_since is not None:
                    m.blocked_by, m.blocked_since = [], None
                continue
            # ---- blocked ----
            if m.sp is None:
                m.sp = dr.pos  # board holds where it is
            if m.blocked_since is None:
                m.blocked_since = now
            ids = [b.id for b in blockers]
            m.blocked_by = ids
            waited = now - m.blocked_since
            if now - m.last_resolve < 1.5:
                continue
            m.last_resolve = now
            m.stuck += 1
            for b in blockers:
                if b.controllable and b.flying and not b.busy and not b.motion.moving:
                    # idle drone in the way: send it to a free parking spot
                    spot = self.free_spot(b, drones, [self.path_ahead(dr, self.lookahead + 1.0)])
                    if spot:
                        request_yield(b, spot, 'уступает дорогу ' + dr.id)
            statics = [b for b in blockers if not b.controllable]
            if statics:
                ob = statics[0]
                need = self.required(dr, ob)
                if d2(m.route[0], ob.pos) < need:
                    # waypoint itself is too close to a landed/manual drone: shorten the leg
                    q = self.pull_back(origin, m.route[0], ob.pos, need + 0.05)
                    if q is not None and d2(q, origin) > 0.1:
                        self.log('warn', '%s: укорачиваю отрезок до (%.2f; %.2f) — рядом стоит %s' % (dr.id, q[0], q[1], ob.id))
                        m.route[0] = q
                        m.blocked_by, m.blocked_since, m.stuck = [], None, 0
                    else:
                        self._skip(dr, origin, now, 'рядом неуправляемый ' + ob.id)
                elif waited > 1.0:
                    w = self.detour(dr, m.route[0], ob, drones)
                    if w:
                        m.route.insert(0, w)
                        self.log('info', '%s: объезд %s' % (dr.id, ob.id))
                    elif m.stuck > 5:
                        self._skip(dr, origin, now, 'не обойти ' + ob.id)
                continue
            if m.stuck > 20:
                self._skip(dr, origin, now, 'проход занят %s' % ','.join(ids))
                continue
            higher = [b for b in blockers if b.controllable and b.prio > dr.prio and b.motion.blocked_since]
            if (higher and waited > 2.0) or waited > 8.0:
                spot = self.free_spot(dr, drones, [self.path_ahead(b, self.lookahead + 1.0) for b in blockers])
                if spot:
                    m.route.insert(0, spot)
                    m.blocked_since = now
                    self.log('info', '%s: отхожу в сторону, пропускаю %s' % (dr.id, ','.join(ids)))

    @staticmethod
    def pull_back(a, b, c, dist):
        """Farthest point on segment a->b that is at least dist from c (None if a itself is too close)."""
        for i in range(40, -1, -1):
            k = i / 40.0
            q = (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k) + tuple(b[2:])
            if d2(q, c) >= dist:
                return q
        return None

    def _skip(self, dr, origin, now, why):
        m = dr.motion
        if m.homing and len(m.route) == 1:
            # the landing spot is blocked: land at the nearest free spot instead of "here"
            spot = self.free_spot(dr, [d for d in self._drones if d is not dr], [], z=m.route[0][2])
            if spot and d2(spot, origin) > 0.1:
                self.log('warn', '%s: место посадки занято (%s) — сяду рядом в (%.2f; %.2f)' % (dr.id, why, spot[0], spot[1]))
                m.route[0] = spot
                m.final = spot
                m.blocked_by, m.blocked_since, m.stuck = [], None, 0
                return
        q = m.route.pop(0)
        self.log('warn', '%s: пропускаю точку (%.2f; %.2f) — %s' % (dr.id, q[0], q[1], why))
        if not m.route:
            m.final, m.t_route_end = tuple(origin), now
        m.blocked_by, m.blocked_since, m.stuck = [], None, 0

    def keep_clear(self, idle, busy_drones, drones, request_yield):
        """Idle flying drones leave the path of busy ones in advance."""
        for dr in idle:
            if dr.pos is None:
                continue
            paths = [self.path_ahead(b) for b in busy_drones if b is not dr and b.motion.moving]
            if not paths:
                continue
            if min(point_poly(dr.pos, p) for p in paths) < self.D + 0.1:
                spot = self.free_spot(dr, drones, paths)
                if spot:
                    request_yield(dr, spot, 'освобождает маршрут')
