#!/usr/bin/env python3
"""Thermal datasets for the IR neural detector (К3-09).

  synth   synthetic frames from the simulator camera model, exact labels
          python tools/thermal_dataset.py synth --out dataset/thermal_synth --n 900
  export  real frames from a flight record + pre-labels from the threshold
          detector, to be corrected by hand in Label Studio / CVAT
          python tools/thermal_dataset.py export --run runs/<date> --out dataset/thermal_raw

Images are produced by server.thermal_nn.model_image — the same preprocessing
the server uses in flight. Classes: 0 thermal_anomaly, 1 drone.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from server.thermal import project, orient, downsample, components  # noqa: E402
from server.thermal_nn import model_image  # noqa: E402

CLASSES = ['thermal_anomaly', 'drone']


def _write(out, split, name, img, boxes, w, h):
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(out, sub, split), exist_ok=True)
    img.save(os.path.join(out, 'images', split, name + '.png'))
    with open(os.path.join(out, 'labels', split, name + '.txt'), 'w') as f:
        for c, (x0, y0, x1, y1) in boxes:
            f.write('%d %.6f %.6f %.6f %.6f\n' % (c, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h))


def _yaml(out, splits):
    with open(os.path.join(out, 'data.yaml'), 'w') as f:
        f.write('path: %s\n' % os.path.abspath(out))
        for s in splits:
            f.write('%s: images/%s\n' % (s, s))
        f.write('names:\n' + ''.join('  %d: %s\n' % (i, c) for i, c in enumerate(CLASSES)))


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def synth_frame(rng, cfg, w=None, h=None):
    """One random scene seen by the thermal camera. Returns frame (C), boxes, pose."""
    t = cfg['thermal']
    w, h = (w, h) if w else t.get('sim_size', [32, 24])
    sx0, sx1, sy0, sy1 = cfg['survey_bounds']
    pose = {'x': rng.uniform(sx0 + 0.5, sx1 - 0.5), 'y': rng.uniform(sy0 + 0.5, sy1 - 0.5),
            'z': rng.uniform(1.6, 2.4), 'roll': rng.normal(0, 0.03), 'pitch': rng.normal(0, 0.03),
            'yaw': rng.normal(0, 0.15)}
    pts, ok, rays = project(w, h, t['hfov_deg'], t['vfov_deg'], pose, 0.0)
    gx, gy = pts[..., 0], pts[..., 1]
    T = rng.uniform(15, 30) + rng.uniform(-0.2, 0.2) * gx + rng.uniform(-0.2, 0.2) * gy
    # distractors that are NOT anomalies: broad warm/cold patches (sun, wet soil)
    for _ in range(rng.integers(0, 3)):
        cx, cy = pose['x'] + rng.uniform(-2, 2), pose['y'] + rng.uniform(-1.5, 1.5)
        T = T + rng.uniform(-3, 2) * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * rng.uniform(0.6, 1.2) ** 2))
    boxes = []
    for _ in range(rng.integers(0, 4)):      # hot objects (simulated infected plants)
        cx, cy = pose['x'] + rng.uniform(-1.4, 1.4), pose['y'] + rng.uniform(-1.0, 1.0)
        r, dT = rng.uniform(0.08, 0.25), rng.uniform(3.0, 11.0)
        g = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * r ** 2))
        T = T + dT * g
        b = bbox(g > 0.35)
        if b:
            boxes.append((0, b))
    if rng.random() < 0.5:                   # a neighbour drone below: body + 4 hot motors
        R = cfg['separation']['drone_radius_m']
        cx, cy, oz = pose['x'] + rng.uniform(-1.3, 1.3), pose['y'] + rng.uniform(-0.9, 0.9), rng.uniform(0.8, 1.4)
        dz = rays[..., 2]
        s = (oz - pose['z']) / np.where(dz < -1e-3, dz, -1)
        px, py = pose['x'] + s * rays[..., 0], pose['y'] + s * rays[..., 1]
        body = np.hypot(px - cx, py - cy) < R * 0.5
        T = np.where(body, T.mean() + rng.uniform(6, 12), T)
        a0 = rng.uniform(0, math.pi / 2)
        motors = np.zeros_like(body)
        for k in range(4):
            mx, my = cx + R * 0.75 * math.cos(a0 + k * math.pi / 2), cy + R * 0.75 * math.sin(a0 + k * math.pi / 2)
            m = np.hypot(px - mx, py - my) < 0.06
            motors |= m
        T = np.where(motors, T.mean() + rng.uniform(15, 25), T)
        b = bbox(body | motors | (np.hypot(px - cx, py - cy) < R))
        if b:
            boxes.append((1, b))
    T = T + rng.normal(0, rng.uniform(0.1, 0.4), T.shape)
    return T.astype(np.float32), boxes, pose


def synth(a):
    cfg = json.load(open(os.path.join(ROOT, 'config.json'), encoding='utf-8'))
    rng = np.random.default_rng(a.seed)
    nn = cfg['thermal']
    counts = {'train': [0, 0], 'val': [0, 0], 'test': [0, 0]}
    for i in range(a.n):
        split = 'train' if i < a.n * 0.7 else ('val' if i < a.n * 0.85 else 'test')
        T, boxes, _ = synth_frame(rng, cfg)
        dT = T - np.median(T)
        img, _ = model_image(dT, nn['nn_range'], nn['nn_imgsz'])
        _write(a.out, split, 'syn_%05d' % i, img, boxes, T.shape[1], T.shape[0])
        for c, _ in boxes:
            counts[split][c] += 1
    _yaml(a.out, ['train', 'val', 'test'])
    json.dump({'kind': 'synthetic', 'seed': a.seed, 'frames': a.n, 'classes': CLASSES, 'boxes': counts,
               'preprocess': {'nn_range': nn['nn_range'], 'imgsz': nn['nn_imgsz']}},
              open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)
    print('frames %d, boxes per split [anomaly, drone]: %s' % (a.n, counts))
    print('data:', os.path.join(a.out, 'data.yaml'))


def export(a):
    """Real recorded frames -> images + threshold pre-labels (class 0) for manual correction."""
    cfg = json.load(open(os.path.join(a.run, 'config.json'), encoding='utf-8'))
    t = cfg['thermal']
    raw = open(os.path.join(a.run, 'thermal.f32'), 'rb').read()
    n = 0
    for line in open(os.path.join(a.run, 'thermal.jsonl')):
        r = json.loads(line)
        if a.used_only and not r.get('used'):
            continue
        if n % a.every:
            n += 1
            continue
        frame = np.frombuffer(raw[r['offset']:r['offset'] + r['nbytes']], '<f4').reshape(r['h'], r['w'])
        frame, _ = orient(frame, t['rotate_deg'], t['flip_h'], t['flip_v'])
        small = downsample(frame)
        dT = small - np.median(small)
        boxes = [(0, [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
                 for ys, xs in components(dT >= t['threshold']) if len(ys) >= t['min_pixels']]
        img, _ = model_image(dT, t.get('nn_range', [-2, 12]), t.get('nn_imgsz', 160))
        _write(a.out, 'all', 'f%06d' % r['index'], img, boxes, dT.shape[1], dT.shape[0])
        n += 1
    open(os.path.join(a.out, 'classes.txt'), 'w').write('\n'.join(CLASSES) + '\n')
    print('exported %d frames to %s (labels are threshold PRE-labels: check every frame by hand)' % (n // a.every, a.out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('synth')
    s.add_argument('--out', default='dataset/thermal_synth')
    s.add_argument('--n', type=int, default=900)
    s.add_argument('--seed', type=int, default=42)
    s = sub.add_parser('export')
    s.add_argument('--run', required=True)
    s.add_argument('--out', default='dataset/thermal_raw')
    s.add_argument('--every', type=int, default=3, help='take every N-th frame (neighbours are near-duplicates)')
    s.add_argument('--used-only', action='store_true', help='only frames used for the map (survey altitude)')
    a = p.parse_args()
    {'synth': synth, 'export': export}[a.cmd](a)


if __name__ == '__main__':
    main()
