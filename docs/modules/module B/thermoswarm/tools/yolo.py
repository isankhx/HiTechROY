#!/usr/bin/env python3
"""YOLO helper for inspection photos.

  1) collect:  UI button 📷 on a drone card saves frames to dataset/raw/
  2) label:    Label Studio / CVAT / Roboflow -> export "YOLO" (images + .txt + classes.txt)
  3) split:    python tools/yolo.py split --src export_dir --out dataset/yolo
  4) train:    python tools/yolo.py train --data dataset/yolo/data.yaml   (GPU or Colab recommended)
  5) test:     python tools/yolo.py test --weights models/rgb.pt --src dataset/raw
The server picks up models/rgb.pt on start (config.json -> yolo.weights).
"""
import argparse
import os
import random
import shutil
import sys

IMG = ('.jpg', '.jpeg', '.png', '.bmp')


def find_classes(src):
    y = os.path.join(src, 'data.yaml')               # Roboflow / Ultralytics export
    if os.path.exists(y):
        try:
            import yaml
            names = yaml.safe_load(open(y, encoding='utf-8'))['names']
        except ImportError:
            import ast
            import re
            txt = open(y, encoding='utf-8').read()
            m = re.search(r'^names:\s*(\[.*?\])', txt, re.M | re.S)
            names = ast.literal_eval(m.group(1)) if m else re.findall(r'^\s+\d+:\s*(.+)$', txt, re.M)
        return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for name in ('classes.txt', 'obj.names', 'notes.json'):
        p = os.path.join(src, name)
        if os.path.exists(p) and name != 'notes.json':
            return [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
        if os.path.exists(p):
            import json
            return [c['name'] for c in sorted(json.load(open(p))['categories'], key=lambda c: c['id'])]
    sys.exit('classes.txt not found in %s (export the dataset in YOLO format)' % src)


def split(a):
    classes = find_classes(a.src)
    pairs = []
    for root, _, files in os.walk(a.src):
        for f in files:
            if f.lower().endswith(IMG):
                stem = os.path.splitext(f)[0]
                lbl = None
                for cand in (os.path.join(root, stem + '.txt'), os.path.join(root.replace('images', 'labels'), stem + '.txt')):
                    if os.path.exists(cand):
                        lbl = cand
                        break
                pairs.append((os.path.join(root, f), lbl))
    if len(pairs) < 10:
        sys.exit('only %d images found — label at least ~100 per class for a usable model' % len(pairs))
    random.Random(42).shuffle(pairs)
    n_val = max(1, int(len(pairs) * a.val))
    if os.path.exists(a.out):
        sys.exit('%s exists, choose a new folder' % a.out)
    counts = [0] * len(classes)
    for i, (img, lbl) in enumerate(pairs):
        part = 'val' if i < n_val else 'train'
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(a.out, sub, part), exist_ok=True)
        base = '%05d_%s' % (i, os.path.basename(img))
        shutil.copy(img, os.path.join(a.out, 'images', part, base))
        txt = os.path.join(a.out, 'labels', part, os.path.splitext(base)[0] + '.txt')
        if lbl:
            shutil.copy(lbl, txt)
            for line in open(lbl):
                if line.strip():
                    counts[int(line.split()[0])] += 1
        else:
            open(txt, 'w').close()  # background image (no objects) — useful negatives
    with open(os.path.join(a.out, 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write('path: %s\ntrain: images/train\nval: images/val\nnames:\n' % os.path.abspath(a.out))
        for i, c in enumerate(classes):
            f.write('  %d: %s\n' % (i, c))
    print('images: %d (val %d). boxes per class: %s' % (len(pairs), n_val, dict(zip(classes, counts))))
    print('next: python tools/yolo.py train --data %s' % os.path.join(a.out, 'data.yaml'))


def train(a):
    from ultralytics import YOLO
    model = YOLO(a.model)
    extra = {}
    if a.thermal:
        # thermal image: grey level = temperature difference on a fixed scale -> never alter it
        extra = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=os.path.abspath('runs_yolo'), name='thermal' if a.thermal else 'rgb', exist_ok=False,
                patience=30, seed=0, deterministic=True, workers=a.workers,
                degrees=10, flipud=0.5, fliplr=0.5, **extra)   # camera looks down: any rotation/flip is valid
    best = os.path.join(str(model.trainer.save_dir), 'weights', 'best.pt')
    out = a.out or ('models/thermal.pt' if a.thermal else 'models/rgb.pt')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    shutil.copy(best, out)
    m = YOLO(out).val(data=a.data, imgsz=a.imgsz, device=a.device, split='val',
                      project=os.path.abspath('runs_yolo'), name='val', exist_ok=True)
    print('val mAP50=%.3f  mAP50-95=%.3f' % (m.box.map50, m.box.map))
    print('saved %s — restart the server to use it' % out)


def test(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    srcs = [a.src] if os.path.isfile(a.src) else [os.path.join(a.src, f) for f in sorted(os.listdir(a.src)) if f.lower().endswith(IMG)]
    os.makedirs(a.out, exist_ok=True)
    for p in srcs:
        r = model.predict(p, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        names = ['%s %.2f' % (model.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
        r.save(filename=os.path.join(a.out, os.path.basename(p)))
        print('%-40s %s' % (os.path.basename(p), ', '.join(names) or '—'))
    print('annotated images in', a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('split')
    s.add_argument('--src', required=True)
    s.add_argument('--out', default='dataset/yolo')
    s.add_argument('--val', type=float, default=0.2)
    s = sub.add_parser('train')
    s.add_argument('--data', required=True)
    s.add_argument('--model', default='yolo11n.pt', help='pretrained start point (downloaded automatically)')
    s.add_argument('--epochs', type=int, default=100)
    s.add_argument('--imgsz', type=int, default=640)
    s.add_argument('--batch', type=int, default=16)
    s.add_argument('--device', default='0', help='GPU index, or cpu (slow)')
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--thermal', action='store_true', help='thermal dataset: no colour augmentation, saves models/thermal.pt')
    s.add_argument('--out', help='where to copy best.pt (default models/rgb.pt or models/thermal.pt)')
    s = sub.add_parser('test')
    s.add_argument('--weights', default='models/rgb.pt')
    s.add_argument('--src', required=True)
    s.add_argument('--out', default='runs_yolo/test')
    s.add_argument('--conf', type=float, default=0.35)
    s.add_argument('--imgsz', type=int, default=640)
    a = p.parse_args()
    {'split': split, 'train': train, 'test': test}[a.cmd](a)


if __name__ == '__main__':
    main()
