"""Thermal frame -> field map.

Camera model: looking straight down, image top = drone forward (after the
configured rotate/flip), fov given for the oriented image. Every pixel ray
is rotated by roll/pitch/yaw and intersected with plane z = target height.
Anomaly positions are NOT taken from that plane: every detection keeps its
3-D ray (drone position + direction to the blob centre) and the anomaly is the
least-squares intersection of all its rays (triangulation). The drone moves
between frames, so the rays cross at the real source - its height is estimated
too (the configured target height is only a weak prior for the first frames).
A wrong height assumption therefore no longer splits one object into several.
Background = median of the frame, so only the temperature DIFFERENCE matters
(works with any camera, even one that outputs raw counts).
"""
import base64
import math
import numpy as np

MAX_W, MAX_H = 80, 60


def rotation(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


_ray_cache = {}


def body_rays(w, h, hfov, vfov):
    key = (w, h, round(hfov, 4), round(vfov, 4))
    r = _ray_cache.get(key)
    if r is None:
        tx = ((np.arange(w) + 0.5) / w * 2 - 1) * math.tan(math.radians(hfov) / 2)
        ty = ((np.arange(h) + 0.5) / h * 2 - 1) * math.tan(math.radians(vfov) / 2)
        TX, TY = np.meshgrid(tx, ty)
        # body FLU: forward = up in the image, left = left in the image
        r = np.stack([-TY, -TX, -np.ones_like(TX)], axis=-1)
        _ray_cache[key] = r
    return r


def project(w, h, hfov, vfov, pose, plane_z):
    """Ground points (h, w, 2) and validity mask for a camera at pose."""
    R = rotation(pose['roll'], pose['pitch'], pose['yaw'])
    d = body_rays(w, h, hfov, vfov) @ R.T
    dz = d[..., 2]
    ok = dz < -1e-3
    s = np.where(ok, (plane_z - pose['z']) / np.where(ok, dz, -1), 0)
    ok &= s > 0
    pts = np.stack([pose['x'] + s * d[..., 0], pose['y'] + s * d[..., 1]], axis=-1)
    return pts, ok, d


def orient(frame, rotate_deg, flip_h, flip_v):
    k = int(round(rotate_deg / 90.0)) % 4
    if k:
        frame = np.rot90(frame, -k)  # clockwise
    if flip_h:
        frame = frame[:, ::-1]
    if flip_v:
        frame = frame[::-1, :]
    return np.ascontiguousarray(frame), k % 2 == 1


def downsample(frame):
    h, w = frame.shape
    f = max(1, math.ceil(w / MAX_W), math.ceil(h / MAX_H))
    if f == 1:
        return frame
    h2, w2 = h // f, w // f
    return frame[:h2 * f, :w2 * f].reshape(h2, f, w2, f).mean(axis=(1, 3))


def upsample(a, k):
    if k <= 1:
        return a
    h, w = a.shape
    ys = np.clip((np.arange(h * k) + 0.5) / k - 0.5, 0, h - 1)
    xs = np.clip((np.arange(w * k) + 0.5) / k - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def components(mask):
    """4-connected components of a small boolean array -> list of (ys, xs)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, ys, xs = [(y0, x0)], [], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((np.array(ys), np.array(xs)))
    return out


class ThermalMapper:
    def __init__(self, cfg):
        t = cfg['thermal']
        self.t = t
        self.R = cfg['separation']['drone_radius_m']
        self.bounds = cfg['survey_bounds']
        self.cell = t['grid_cell_m']
        x0, x1, y0, y1 = self.bounds
        self.nx = int(math.ceil((x1 - x0) / self.cell))
        self.ny = int(math.ceil((y1 - y0) / self.cell))
        self.reset()

    def reset(self):
        self.sum = np.zeros((self.ny, self.nx), np.float64)
        self.cnt = np.zeros((self.ny, self.nx), np.float32)
        self.anomalies = []
        self.frames = 0
        self.last_live = None
        self.next_id = 1

    def fov(self, swapped):
        h, v = self.t['hfov_deg'], self.t['vfov_deg']
        return (v, h) if swapped else (h, v)

    def prepare(self, frame):
        """Orientation, downsampling, background. The same dT goes to the threshold
        detector, the neural detector and the live view."""
        cfg = self.t
        frame, swapped = orient(np.asarray(frame, np.float32), cfg['rotate_deg'], cfg['flip_h'], cfg['flip_v'])
        small = downsample(frame)
        bg = float(np.median(small))
        prep = {'dT': small - bg, 'bg': bg, 'swapped': swapped}
        self.last_live = {'w': small.shape[1], 'h': small.shape[0], 'bg': bg, 'dT': prep['dT']}
        return prep

    def process(self, frame, pose, others, t, frame_index=None, nn_boxes=None):
        return self.process_prepared(self.prepare(frame), pose, others, t, frame_index, nn_boxes)

    def process_prepared(self, prep, pose, others, t, frame_index=None, nn_boxes=None):
        """others: [(x,y,z,flying,position_reliable)]; nn_boxes: neural detections on prep['dT']
        or None. Detector mode thermal.detector: threshold | nn | both.
        Returns (changed anomaly list, [])."""
        cfg = self.t
        dT, bg = prep['dT'], prep['bg']
        hfov, vfov = self.fov(prep['swapped'])
        h, w = dT.shape
        if pose is None or not all(math.isfinite(pose[k]) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
            return [], []
        tilt = math.degrees(max(abs(pose['roll']), abs(pose['pitch'])))
        if tilt > cfg['max_tilt_deg'] or pose['z'] < 0.5:
            return [], []
        zt = cfg['target_height_m']
        alt = pose['z'] - zt
        if alt < 0.3:
            return [], []
        self.frames += 1

        # --- heat map (interpolated for display) ---
        px_ground = alt * 2 * math.tan(math.radians(hfov) / 2) / w
        k = int(min(4, max(1, math.ceil(px_ground / self.cell))))
        up = upsample(dT, k)
        pts, ok, rays = project(w * k, h * k, hfov, vfov, pose, zt)
        ok &= self._mask_drones(pts, rays, pose, others)
        x0, _, y0, _ = self.bounds
        ix = np.floor((pts[..., 0] - x0) / self.cell).astype(int)
        iy = np.floor((pts[..., 1] - y0) / self.cell).astype(int)
        ok &= (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        np.add.at(self.sum, (iy[ok], ix[ok]), up[ok])
        np.add.at(self.cnt, (iy[ok], ix[ok]), 1)

        # --- detection on the (downsampled) native grid ---
        pts1, ok1, rays1 = project(w, h, hfov, vfov, pose, zt)
        ok1 &= self._mask_drones(pts1, rays1, pose, others)
        mode = cfg.get('detector', 'threshold') if nn_boxes is not None else 'threshold'
        if mode not in ('threshold', 'nn', 'both'):
            mode = 'threshold'
        blobs = []                                  # (ys, xs, source, conf)
        for b in nn_boxes or []:
            x0, y0 = max(0, int(math.floor(b['xyxy'][0]))), max(0, int(math.floor(b['xyxy'][1])))
            x1, y1 = min(w, int(math.ceil(b['xyxy'][2]))), min(h, int(math.ceil(b['xyxy'][3])))
            if x1 <= x0 or y1 <= y0:
                continue
            if b['cls'] != 'thermal_anomaly':       # e.g. 'drone': hot motors, never a find
                ok1[y0:y1, x0:x1] = False
                continue
            if mode in ('nn', 'both'):
                ys, xs = np.nonzero(ok1[y0:y1, x0:x1])
                if len(ys):
                    blobs.append((ys + y0, xs + x0, 'nn', b['conf']))
        if mode in ('threshold', 'both'):
            for ys, xs in components((dT >= cfg['threshold']) & ok1):
                if len(ys) >= cfg['min_pixels']:
                    blobs.append((ys, xs, 'thr', None))
        touched = set()
        err = 0.1 + px_ground + alt * math.tan(math.radians(2))
        origin = np.array([pose['x'], pose['y'], pose['z']])
        for ys, xs, src, conf in blobs:
            wts = np.maximum(dT[ys, xs], 0.0) + 1e-3
            ray = (rays1[ys, xs] * wts[:, None]).sum(0)
            ray = ray / np.linalg.norm(ray)
            # a blob cut by the frame edge has a biased centre: it counts less
            edge = ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1
            peak = float(dT[ys, xs].max())
            a = self._merge(origin, ray, 0.3 if edge else 1.0, max(peak, 0.1), err, t, touched)
            a['sources'][src] = a['sources'].get(src, 0) + 1
            if conf is not None:
                a['nn_conf'] = max(a.get('nn_conf') or 0.0, conf)
            if a['_best'] < peak:
                a['_best'] = peak
                a['thermal'] = {'w': w, 'h': h, 'bg': round(bg, 2),
                                'dT': [round(float(v), 2) for v in dT.ravel()],
                                'pose': {k2: round(float(pose[k2]), 3) for k2 in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
                                'frame_index': frame_index, 't': t}
        for a in self.anomalies:
            if a['id'] in touched:
                a['observations'] += 1
        return [a for a in self.anomalies if a['id'] in touched], []

    def promote(self, now, passed_s=1.0):
        """'new' -> 'confirmed' once seen enough AND the camera has moved past it
        (or it stayed in view for many frames). The position is then settled,
        so the inspection drone flies to the refined point, not the first glimpse."""
        out = []
        for a in self.anomalies:
            if a['status'] == 'new' and a['observations'] >= self.t['confirm_observations'] and \
                    (now - a['t_last'] > passed_s or a['observations'] >= 20):
                a['status'] = 'confirmed'
                out.append(a)
        return out

    def _mask_drones(self, pts, rays, pose, others):
        """Hide pixels that look at other drones (hot motors / batteries)."""
        keep = np.ones(pts.shape[:2], bool)
        for (ox, oy, oz, flying, reliable) in others:
            # ground footprint of a landed drone, or its body when flying below us;
            # a drone whose position is stale (lost markers/link) may have moved: mask wider
            rad = self.R + 0.2
            keep &= np.hypot(pts[..., 0] - ox, pts[..., 1] - oy) > rad
            if flying and oz < pose['z'] - 0.1:
                dz = rays[..., 2]
                s = np.where(dz < -1e-3, (oz - pose['z']) / np.where(dz < -1e-3, dz, -1), 0)
                px = pose['x'] + s * rays[..., 0]
                py = pose['y'] + s * rays[..., 1]
                keep &= np.hypot(px - ox, py - oy) > rad
        return keep

    @staticmethod
    def _hit(origin, ray, z):
        """Where the ray crosses height z (None if it never goes down to z)."""
        if ray[2] > -1e-3 or z >= origin[2]:
            return None
        s = (z - origin[2]) / ray[2]
        return origin[0] + s * ray[0], origin[1] + s * ray[1]

    def _merge(self, origin, ray, wt, peak, err, t, touched):
        """Attach the ray to the anomaly it points at (checked at that anomaly's
        own estimated height), or start a new one at the prior height."""
        best, bd = None, self.t['merge_radius_m']
        for a in self.anomalies:
            p = self._hit(origin, ray, a['z'])
            if p is None:
                continue
            d = math.hypot(a['x'] - p[0], a['y'] - p[1])
            if d < bd:
                best, bd = a, d
        if best is None:
            zt = self.t['target_height_m']
            p = self._hit(origin, ray, zt) or (origin[0], origin[1])
            best = self._new(p[0], p[1], zt, err, peak, t)
            best.update({'_A': np.zeros((3, 3)), '_b': np.zeros(3), '_c': 0.0, '_wsum': 0.0})
            self.next_id += 1
            self.anomalies.append(best)
        if best['id'] not in touched and best.get('source') == 'operator':
            # placed by the operator: detections confirm it but never move it
            best['dT_max'] = max(best['dT_max'], peak)
            best['t_last'] = t
            touched.add(best['id'])
        elif best['id'] not in touched:
            M = wt * (np.eye(3) - np.outer(ray, ray))
            best['_A'] += M
            best['_b'] += M @ origin
            best['_c'] += float(origin @ M @ origin)
            best['_wsum'] += wt
            best['err0'] = min(best.get('err0', err), err)
            self._solve(best)
            best['dT_max'] = max(best['dT_max'], peak)
            best['t_last'] = t
            touched.add(best['id'])
        return best

    def _new(self, x, y, z, err, peak, t):
        return {'id': 'A%d' % self.next_id, 'x': x, 'y': y, 'z': z, 'err': err,
                'dT_max': peak, 'observations': 0, 't_first': t, 't_last': t,
                'status': 'new', 'drone': None, 'detections': [], 'label': None,
                'photo': None, 'photo_annotated': None, 'photos': [], 'sources': {}, 'nn_conf': None,
                '_best': -1e9, 'thermal': None}

    def _solve(self, a):
        """Least-squares point closest to all rays + weak prior z = target height
        (decides only while the rays are still almost parallel)."""
        lam = self.t.get('height_prior_weight', 0.02)
        zt = self.t['target_height_m']
        A = a['_A'].copy()
        b = a['_b'].copy()
        A[2, 2] += lam
        b[2] += lam * zt
        try:
            P = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return
        zmax = self.t.get('max_source_height_m', 3.0)
        if not -0.2 <= P[2] <= zmax:            # nonsense geometry: keep the plane estimate
            A[2, 2] += 1e6
            b[2] += 1e6 * min(max(P[2], 0.0), zmax)
            P = np.linalg.solve(A, b)
        a['x'], a['y'], a['z'] = float(P[0]), float(P[1]), float(P[2])
        # rms distance of the rays from the point -> honest position error
        sse = max(0.0, float(P @ a['_A'] @ P - 2 * a['_b'] @ P + a['_c']))
        rms = math.sqrt(sse / max(a['_wsum'], 1e-9))
        a['err'] = round(max(0.05, min(a['err0'], 2 * rms + 0.03)) if a['_wsum'] >= 5 else a['err0'], 3)

    def _absorb(self, a, b):
        """Subclass hook: merge type-specific fields of b into a."""

    def _rms(self, A, b, c, wsum):
        lam = self.t.get('height_prior_weight', 0.5)
        A2, b2 = A.copy(), b.copy()
        A2[2, 2] += lam
        b2[2] += lam * self.t['target_height_m']
        try:
            P = np.linalg.solve(A2, b2)
        except np.linalg.LinAlgError:
            return 9.0
        return math.sqrt(max(0.0, float(P @ A @ P - 2 * b @ P + c)) / max(wsum, 1e-9))

    def _consistent(self, a, b):
        ra = self._rms(a['_A'], a['_b'], a['_c'], a['_wsum'])
        rb = self._rms(b['_A'], b['_b'], b['_c'], b['_wsum'])
        rab = self._rms(a['_A'] + b['_A'], a['_b'] + b['_b'], a['_c'] + b['_c'], a['_wsum'] + b['_wsum'])
        return rab < max(0.08, 1.5 * max(ra, rb))

    def consolidate(self):
        """Two anomalies whose triangulated points ended up closer than the merge
        radius are one object seen before its height was known: fold the younger
        (still unassigned) one into the older. Returns [(kept, removed)]."""
        out = []
        R = self.t['merge_radius_m']
        for a in list(self.anomalies):
            if a not in self.anomalies:
                continue
            manual = a.get('source') == 'operator'
            for b in list(self.anomalies):
                if b is a or b.get('source') == 'operator' or b['status'] not in ('new', 'confirmed') or b['drone']:
                    continue
                if not manual and b['t_first'] < a['t_first']:
                    continue
                if b['observations'] < 3 or math.hypot(a['x'] - b['x'], a['y'] - b['y']) >= R:
                    continue
                if manual:
                    # a detection next to an operator-placed object is that object
                    a['observations'] += b['observations']
                    a['dT_max'] = max(a['dT_max'], b['dT_max'])
                    self.anomalies.remove(b)
                    out.append((a, b))
                    continue
                # one source only if ALL rays of both still meet in one point
                if not self._consistent(a, b):
                    continue
                for k in ('_A', '_b'):
                    a[k] = a[k] + b[k]
                a['_c'] += b['_c']
                a['_wsum'] += b['_wsum']
                a['observations'] += b['observations']
                a['dT_max'] = max(a['dT_max'], b['dT_max'])
                a['t_last'] = max(a['t_last'], b['t_last'])
                a['err0'] = min(a.get('err0', a['err']), b.get('err0', b['err']))
                for src, n in b.get('sources', {}).items():
                    a['sources'][src] = a['sources'].get(src, 0) + n
                if b.get('_best', -1e9) > a.get('_best', -1e9):
                    a['_best'], a['thermal'] = b['_best'], b.get('thermal')
                self._absorb(a, b)
                self.anomalies.remove(b)
                self._solve(a)
                out.append((a, b))
        return out

    def grid_payload(self):
        lo, hi = self.t['display_range']
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = self.sum / np.maximum(self.cnt, 1)
        q = np.where(self.cnt > 0, 1 + np.clip((mean - lo) / (hi - lo), 0, 1) * 254, 0).astype(np.uint8)
        x0, x1, y0, y1 = self.bounds
        return {'nx': self.nx, 'ny': self.ny, 'cell': self.cell, 'x0': x0, 'y0': y0,
                'lo': lo, 'hi': hi, 'data': base64.b64encode(q.tobytes()).decode()}

    def live_payload(self):
        L = self.last_live
        if L is None:
            return None
        dT = L['dT']
        lo, hi = float(dT.min()), float(dT.max())
        span = max(hi - lo, 3.0)  # calm picture when nothing is hot
        q = np.clip((dT - lo) / span * 255, 0, 255).astype(np.uint8)
        return {'w': L['w'], 'h': L['h'], 'bg': round(L['bg'], 2), 'lo': round(lo, 2), 'hi': round(hi, 2),
                'data': base64.b64encode(q.tobytes()).decode()}


def public(a):
    return {k: v for k, v in a.items() if not k.startswith('_') and k != 'thermal'}
