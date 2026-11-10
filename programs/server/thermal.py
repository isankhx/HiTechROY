"""Thermal frame -> field map.

Camera model: looking straight down, image top = drone forward (after the
configured rotate/flip), fov given for the oriented image. Every pixel ray
is rotated by roll/pitch/yaw and intersected with plane z = target height.
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

    def fov(self, swapped):
        h, v = self.t['hfov_deg'], self.t['vfov_deg']
        return (v, h) if swapped else (h, v)

    def process(self, frame, pose, others, t, frame_index=None):
        """frame: 2D array (native camera orientation). others: [(x,y,z,flying,position_reliable)].
        Returns (changed anomaly list, newly confirmed list)."""
        cfg = self.t
        frame, swapped = orient(np.asarray(frame, np.float32), cfg['rotate_deg'], cfg['flip_h'], cfg['flip_v'])
        hfov, vfov = self.fov(swapped)
        small = downsample(frame)
        h, w = small.shape
        bg = float(np.median(small))
        dT = small - bg
        self.last_live = {'w': w, 'h': h, 'bg': bg, 'dT': dT}
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
        hot = (dT >= cfg['threshold']) & ok1
        touched, confirmed = set(), []
        err = 0.1 + px_ground + alt * math.tan(math.radians(2))
        for ys, xs in components(hot):
            if len(ys) < cfg['min_pixels']:
                continue
            wts = dT[ys, xs]
            gx = float(np.sum(pts1[ys, xs, 0] * wts) / np.sum(wts))
            gy = float(np.sum(pts1[ys, xs, 1] * wts) / np.sum(wts))
            peak = float(wts.max())
            a = self._merge(gx, gy, peak, err, t, touched)
            if a['_best'] < peak:
                a['_best'] = peak
                a['thermal'] = {'w': w, 'h': h, 'bg': round(bg, 2),
                                'dT': [round(float(v), 2) for v in dT.ravel()],
                                'pose': {k2: round(float(pose[k2]), 3) for k2 in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')},
                                'frame_index': frame_index, 't': t}
        for a in self.anomalies:
            if a['id'] in touched:
                a['observations'] += 1
        return [a for a in self.anomalies if a['id'] in touched], confirmed

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

    def _merge(self, x, y, peak, err, t, touched):
        best, bd = None, self.t['merge_radius_m']
        for a in self.anomalies:
            d = math.hypot(a['x'] - x, a['y'] - y)
            if d < bd:
                best, bd = a, d
        if best is None:
            best = {'id': 'A%d' % (len(self.anomalies) + 1), 'x': x, 'y': y, 'err': err,
                    'dT_max': peak, 'observations': 0, 't_first': t, 't_last': t,
                    'status': 'new', 'drone': None, 'detections': [], 'label': None,
                    'photo': None, 'photo_annotated': None, '_w': 0.0, '_best': -1e9, 'thermal': None}
            self.anomalies.append(best)
        if best['id'] not in touched:
            w0, w1 = best['_w'], max(peak, 0.1)
            best['x'] = (best['x'] * w0 + x * w1) / (w0 + w1)
            best['y'] = (best['y'] * w0 + y * w1) / (w0 + w1)
            best['_w'] = w0 + w1
            best['err'] = min(best['err'], err)
            best['dT_max'] = max(best['dT_max'], peak)
            best['t_last'] = t
            touched.add(best['id'])
        return best

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
