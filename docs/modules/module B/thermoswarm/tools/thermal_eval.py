#!/usr/bin/env python3
"""mAP@0.5 of the IR neural detector (case requirement К3-09: mAP@0.5 >= 0.85 in flight).

  dataset   python tools/thermal_eval.py dataset --weights models/thermal.pt --data dataset/thermal_synth/data.yaml --split test
  flight    python tools/thermal_eval.py flight --run runs/<date> --labels dataset/thermal_flight/labels/all
            (detections made IN FLIGHT, from runs/<date>/nn.jsonl, against hand labels of the same
             frames exported with tools/thermal_dataset.py export --every 1)
            python tools/thermal_eval.py flight --run runs/<date>-sim --sim-truth
            (simulator flights: ground truth from the known hot objects and the recorded poses)

AP is the all-point interpolated area under precision/recall at IoU 0.5 (VOC 2010+,
same definition as Ultralytics mAP50). Writes <out>.json with per-class AP.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CLASSES = ['thermal_anomaly', 'drone']


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def average_precision(preds, gts, thr=0.5):
    """preds: [(frame, conf, box)], gts: {frame: [box]} for one class."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return None, 0, 0
    used = {f: [False] * len(v) for f, v in gts.items()}
    tp, fp = [], []
    for f, _, box in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0.0, -1
        for i, g in enumerate(gts.get(f, [])):
            o = iou(box, g)
            if o > best:
                best, bi = o, i
        if best >= thr and not used[f][bi]:
            used[f][bi] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos if len(tp) else np.array([0.0])
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.array([0.0])
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, npos, len(preds)


def report(per_class, out, meta):
    aps = [v['AP50'] for v in per_class.values() if v['AP50'] is not None]
    res = dict(meta, per_class=per_class, mAP50=round(float(np.mean(aps)), 4) if aps else None)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if out:
        json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
        print('saved', out)
    return res


def dataset(a):
    from ultralytics import YOLO
    m = YOLO(a.weights)
    r = m.val(data=a.data, split=a.split, imgsz=a.imgsz, conf=0.001, iou=0.6, device=a.device, verbose=False,
              project=os.path.join(ROOT, 'runs_yolo'), name='eval', exist_ok=True)
    per = {}
    for i, name in m.names.items():
        if i < len(r.box.ap50):
            per[name] = {'AP50': round(float(r.box.ap50[i]), 4)}
    report(per, a.out, {'mode': 'dataset', 'weights': a.weights, 'data': a.data, 'split': a.split,
                        'images': int(r.seen) if hasattr(r, 'seen') else None, 'mAP50_ultralytics': round(float(r.box.map50), 4)})


def load_labels(d, w_h):
    gts = {}
    for fn in os.listdir(d):
        if not fn.endswith('.txt'):
            continue
        idx = int(''.join(ch for ch in fn if ch.isdigit()))
        w, h = w_h.get(idx, (None, None))
        if w is None:
            continue
        for line in open(os.path.join(d, fn)):
            if not line.strip():
                continue
            c, x, y, bw, bh = map(float, line.split())
            gts.setdefault(int(c), {}).setdefault(idx, []).append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return gts


def sim_truth(run, frames, cfg):
    from server.thermal import project
    t = cfg['thermal']
    poses = {}
    for line in open(os.path.join(run, 'thermal.jsonl')):
        r = json.loads(line)
        poses[r['index']] = r['pose']
    gts = {0: {}}
    for idx, (w, h) in frames.items():
        pose = poses.get(idx)
        if not pose:
            continue
        pts, ok, _ = project(w, h, t['hfov_deg'], t['vfov_deg'], pose, 0.0)
        for o in cfg['sim']['objects']:
            g = np.exp(-((pts[..., 0] - o['x']) ** 2 + (pts[..., 1] - o['y']) ** 2) / (2 * o['r'] ** 2))
            ys, xs = np.nonzero(g > 0.35)
            if len(xs) >= 2:
                gts[0].setdefault(idx, []).append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return gts


def flight(a):
    cfg = json.load(open(os.path.join(a.run, 'config.json'), encoding='utf-8'))
    recs = [json.loads(l) for l in open(os.path.join(a.run, 'nn.jsonl'))]
    recs = [r for r in recs if r.get('index') is not None]
    frames = {r['index']: (r['w'], r['h']) for r in recs}
    preds = {}
    for r in recs:
        for b in r['boxes']:
            c = CLASSES.index(b['cls']) if b['cls'] in CLASSES else -1
            preds.setdefault(c, []).append((r['index'], b['conf'], b['xyxy']))
    if a.sim_truth:
        gts = sim_truth(a.run, frames, cfg)
        classes = [0]           # drones in sim frames are not labelled: evaluate the anomaly class
    else:
        gts = load_labels(a.labels, frames)
        classes = sorted(gts)
    per = {}
    for c in classes:
        ap, npos, npred = average_precision(preds.get(c, []), gts.get(c, {}))
        per[CLASSES[c]] = {'AP50': round(ap, 4) if ap is not None else None, 'objects': npos, 'detections': npred}
    report(per, a.out, {'mode': 'flight', 'run': a.run, 'frames_with_nn': len(frames),
                        'truth': 'simulator objects' if a.sim_truth else a.labels,
                        'conf_threshold_in_flight': cfg['thermal'].get('nn_conf')})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('dataset')
    s.add_argument('--weights', default='models/thermal.pt')
    s.add_argument('--data', required=True)
    s.add_argument('--split', default='test')
    s.add_argument('--imgsz', type=int, default=128)
    s.add_argument('--device', default='cpu')
    s.add_argument('--out')
    s = sub.add_parser('flight')
    s.add_argument('--run', required=True)
    s.add_argument('--labels', help='YOLO txt labels named by frame index (f000123.txt)')
    s.add_argument('--sim-truth', action='store_true')
    s.add_argument('--out')
    a = p.parse_args()
    if a.cmd == 'flight' and not (a.labels or a.sim_truth):
        p.error('--labels or --sim-truth')
    {'dataset': dataset, 'flight': flight}[a.cmd](a)


if __name__ == '__main__':
    main()
