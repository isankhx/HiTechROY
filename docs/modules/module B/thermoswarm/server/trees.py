"""Trees on the field from RGB frames (all trees, heated or not).

Every flying drone streams its downward RGB camera. On each frame:
  1. detect tree crowns in the picture
       'dark' - classical: crowns are big, dark and spiky on the light floor;
                ArUco markers (dark squares) are rejected by shape;
       'yolo' - a trained RGB model, classes whose name contains 'tree'/'ёлк'/'el';
  2. turn the crown centre pixel into a 3-D ray with the real camera model
     (camera_info K/D + camera mounting from tf, both sent by the board agent);
  3. attach the ray to a tree whose triangulated point it passes (same
     machinery as thermal anomalies, see thermal.ThermalMapper), or start a new one.
A tree is 'infected' when a thermal anomaly lies on it (heated white marks);
the rest are healthy. Trees are listed as E1, E2...
"""
import math
import numpy as np
from .thermal import ThermalMapper, rotation

DEFAULT_TREES = {
    'enabled': True,
    'detector': 'dark',            # dark | yolo | off
    'yolo_weights': 'models/trees.pt',
    'yolo_conf': 0.35,
    'tree_height_m': 1.6,
    'min_crown_m': 0.25,           # crown diameter limits (projected at tree mid height)
    'max_crown_m': 1.6,
    'dark_ratio': 0.45,            # pixel darker than floor * ratio = "dark"
    'min_roughness': 1.45,         # perimeter / circle perimeter of same area: crowns are spiky
    'merge_radius_m': 0.45,
    'confirm_observations': 3,
    'link_radius_m': 0.5,          # thermal anomaly closer than this -> tree is infected
    'inspect_healthy': True,       # photo drones also visit healthy trees
    'healthy_views': 1,            # nadir only
    'fps': 3.0,                    # frames per second per drone used for mapping
    'min_alt_m': 1.0,
    'max_tilt_deg': 15,
}


class CamGeom:
    """Pixel -> ray in the drone body frame (FLU).
    From the agent: {'w','h','K'(9),'D','model','R_bc'(9, camera optical -> base_link)}.
    Without it: pinhole with hfov, image top = forward (like the thermal camera)."""

    FALLBACK_R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    def __init__(self, caps=None, hfov_deg=90.0):
        self.caps = caps or {}
        self.hfov = hfov_deg
        g = self.caps
        self.ok = bool(g.get('K')) and g.get('w')
        self.R = np.array(g['R_bc'], float).reshape(3, 3) if g.get('R_bc') else self.FALLBACK_R

    def rays(self, px, img_w, img_h):
        """px: (N, 2) pixel coords in the received image -> (N, 3) unit rays, body frame."""
        px = np.asarray(px, np.float64).reshape(-1, 2)
        if self.ok:
            import cv2
            s = img_w / float(self.caps['w'])
            K = np.array(self.caps['K'], np.float64).reshape(3, 3).copy()
            K[:2] *= s
            D = np.array(self.caps.get('D') or [], np.float64)
            pts = px.reshape(-1, 1, 2)
            if self.caps.get('model') in ('equidistant', 'fisheye') and len(D) >= 4:
                n = cv2.fisheye.undistortPoints(pts, K, D[:4]).reshape(-1, 2)
            elif len(D):
                n = cv2.undistortPoints(pts, K, D).reshape(-1, 2)
            else:
                n = np.column_stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1]])
        else:
            f = (img_w / 2.0) / math.tan(math.radians(self.hfov) / 2)
            n = np.column_stack([(px[:, 0] - img_w / 2.0) / f, (px[:, 1] - img_h / 2.0) / f])
        r = np.column_stack([n, np.ones(len(n))]) @ self.R.T
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def project(self, P, pose, img_w, img_h):
        """World point -> pixel in an image taken at pose, or None (behind / outside)."""
        Rw = rotation(pose['roll'], pose['pitch'], pose['yaw'])
        v = Rw.T @ (np.asarray(P, float) - np.array([pose['x'], pose['y'], pose['z']]))
        c = self.R.T @ v                                  # camera optical frame
        if c[2] <= 1e-3:
            return None
        if self.ok:
            import cv2
            s = img_w / float(self.caps['w'])
            K = np.array(self.caps['K'], np.float64).reshape(3, 3).copy()
            K[:2] *= s
            D = np.array(self.caps.get('D') or [], np.float64)
            obj = c.reshape(1, 1, 3)
            if self.caps.get('model') in ('equidistant', 'fisheye') and len(D) >= 4:
                uv, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D[:4])
            else:
                uv, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D if len(D) else np.zeros(5))
            u, w = uv.reshape(2)
        else:
            f = self.focal(img_w)
            u, w = img_w / 2.0 + f * c[0] / c[2], img_h / 2.0 + f * c[1] / c[2]
        if not (-0.1 * img_w <= u <= 1.1 * img_w and -0.1 * img_h <= w <= 1.1 * img_h):
            return None
        return float(u), float(w)

    def focal(self, img_w):
        if self.ok:
            return self.caps['K'][0] * img_w / float(self.caps['w'])
        return (img_w / 2.0) / math.tan(math.radians(self.hfov) / 2)


def detect_dark(bgr, p):
    """Classical crown detector. Returns [(cx, cy, area_px, touches_border)]."""
    import cv2
    h, w = bgr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    floor = float(np.percentile(g, 70))                 # the field is mostly light floor
    if floor < 40:                                      # too dark to tell anything
        return []
    dark = (g < floor * p['dark_ratio']).astype(np.uint8)
    k = max(3, w // 80) | 1
    m = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < w * h * 0.004:
            continue
        cnts, _ = cv2.findContours((lab[y:y + bh, x:x + bw] == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        rect_fill = area / max(rw * rh, 1.0)
        squareness = min(rw, rh) / max(rw, rh, 1.0)
        if rect_fill > 0.8 and squareness > 0.7:        # ArUco marker
            continue
        rough = cv2.arcLength(c, True) / (2 * math.sqrt(math.pi * area))
        if rough < p['min_roughness']:
            continue
        border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        out.append((float(cen[i][0]), float(cen[i][1]), float(area), border))
    return out


class TreeMapper(ThermalMapper):
    def __init__(self, cfg):
        p = dict(DEFAULT_TREES)
        p.update(cfg.get('trees', {}))
        self.p = p
        h = p['tree_height_m']
        # parameters the shared ray/triangulation code reads
        self.t = {'merge_radius_m': p['merge_radius_m'], 'target_height_m': h / 2,
                  'height_prior_weight': 0.05, 'max_source_height_m': h + 0.5}
        self.R = cfg['separation']['drone_radius_m']
        self.bounds = cfg['survey_bounds']
        self.hfov = cfg.get('rgb_camera', {}).get('hfov_deg', 90.0)
        self.yolo = None
        self.reset()

    def reset(self):
        self.anomalies = []          # trees (name kept for the shared code)
        self.next_id = 1
        self.frames = 0

    @property
    def trees(self):
        return self.anomalies

    def _new(self, x, y, z, err, peak, t):
        return {'id': 'E%d' % self.next_id, 'kind': 'tree', 'x': x, 'y': y, 'z': z, 'err': err,
                'crown_m': None, 'dT_max': 0.0, 'observations': 0, 't_first': t, 't_last': t,
                'status': 'new', 'infected': False, 'anomaly': None, 'visual_mark': False,
                'drone': None, 'detections': [], 'label': None, 'photo': None, 'photo_annotated': None,
                'photos': [], 'sources': {}, 'nn_conf': None, '_best': -1e9, 'thermal': None, '_crowns': []}

    def _absorb(self, a, b):
        a['_crowns'] = (a['_crowns'] + b['_crowns'])[-60:]
        a['visual_mark'] = a['visual_mark'] or b['visual_mark']

    # ------------------------------------------------------------ detection (worker thread)
    def detect(self, jpeg):
        """JPEG -> (w, h, [(cx, cy, area_px, border, marked)]). Runs in an executor."""
        import cv2
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if self.p['detector'] == 'yolo':
            if self.yolo is None:
                from .vision import YoloDetector
                self.yolo = YoloDetector(self.p['yolo_weights'], self.p['yolo_conf'], 640)
            if self.yolo.ok:
                dets, _ = self.yolo.detect(jpeg)
                out = []
                for d in dets:
                    name = d['name'].lower()
                    if 'tree' in name or 'ёлк' in name or 'елк' in name or name.startswith('el'):
                        x0, y0, x1, y1 = d['box']
                        border = x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2
                        marked = any(k in name for k in ('mark', 'infect', 'зараж', 'sick'))
                        out.append(((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * (y1 - y0) * 0.75, border, marked))
                return w, h, out
        return w, h, [(cx, cy, a, b, False) for cx, cy, a, b in detect_dark(img, self.p)]

    # ------------------------------------------------------------ mapping (event loop)
    def usable(self, pose):
        if pose is None or not all(math.isfinite(pose.get(k, float('nan'))) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
            return False
        if math.degrees(max(abs(pose['roll']), abs(pose['pitch']))) > self.p['max_tilt_deg']:
            return False
        return pose['z'] >= self.p['min_alt_m']

    def process(self, det, pose, geom, others, t):
        """det from detect(); others: [(x, y, z, flying)]. Returns touched trees."""
        if det is None or not self.usable(pose):
            return []
        w, h, blobs = det
        self.frames += 1
        if not blobs:
            return []
        Rw = rotation(pose['roll'], pose['pitch'], pose['yaw'])
        rays = geom.rays([(b[0], b[1]) for b in blobs], w, h) @ Rw.T
        origin = np.array([pose['x'], pose['y'], pose['z']])
        f = geom.focal(w)
        zmid = self.p['tree_height_m'] / 2
        x0, x1, y0, y1 = self.bounds
        touched = set()
        for (cx, cy, area, border, marked), ray in zip(blobs, rays):
            if border:                                   # cut by the frame: centre and size are wrong
                continue
            g = self._hit(origin, ray, zmid)
            if g is None or not (x0 - 0.1 <= g[0] <= x1 + 0.1 and y0 - 0.1 <= g[1] <= y1 + 0.1):
                continue                                 # outside the field: walls, nets, people
            dist = (pose['z'] - zmid) / max(-ray[2], 1e-3)
            crown = 2 * math.sqrt(area / math.pi) * dist / f
            if not self.p['min_crown_m'] <= crown <= self.p['max_crown_m']:
                continue
            if any(self._is_drone(origin, ray, o) for o in others):
                continue                                 # a drone below us is dark and spiky too
            a = self._merge(origin, ray, 1.0, 0.0, 0.05 + 0.02 * dist, t, touched)
            a['_crowns'] = (a['_crowns'] + [crown])[-60:]
            a['crown_m'] = round(float(np.median(a['_crowns'])), 2)
            a['visual_mark'] = a['visual_mark'] or marked
            a['sources']['rgb'] = a['sources'].get('rgb', 0) + 1
        for a in self.anomalies:
            if a['id'] in touched:
                a['observations'] += 1
        return [a for a in self.anomalies if a['id'] in touched]

    def _is_drone(self, origin, ray, o):
        ox, oy, oz, flying = o
        z = oz if flying else 0.1
        p = self._hit(origin, ray, z)
        return p is not None and math.hypot(p[0] - ox, p[1] - oy) < self.R + 0.25

    def promote(self, now, passed_s=1.0):
        out = []
        for a in self.anomalies:
            if a['status'] == 'new' and a['observations'] >= self.p['confirm_observations'] and \
                    (now - a['t_last'] > passed_s or a['observations'] >= 15):
                a['status'] = 'confirmed'
                out.append(a)
        return out
