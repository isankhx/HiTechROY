"""Object recognition on inspection photos.

Real: Ultralytics YOLO weights (models/rgb.pt, any detect model; class names
come from the weights). Simulation: colour-blob detector for the synthetic
scene, clearly labelled as SIM.
"""
import io
import os


class YoloDetector:
    def __init__(self, weights, conf=0.35, imgsz=640):
        self.ok, self.name, self.error = False, 'YOLO не загружен', None
        self.conf, self.imgsz = conf, imgsz
        if not weights or not os.path.exists(weights):
            self.error = 'нет файла весов %s' % weights
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.ok = True
            self.name = 'YOLO %s (%s)' % (os.path.basename(weights), ', '.join(self.model.names.values()))
        except Exception as e:  # ultralytics missing or broken weights
            self.error = str(e)

    def detect(self, jpeg):
        """Returns (detections, annotated_jpeg or None). Blocking: run in executor."""
        if not self.ok:
            return [], None
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert('RGB')
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append({'name': self.model.names[int(k)], 'conf': round(float(c), 3),
                         'box': [round(v, 1) for v in xyxy]})
        dets.sort(key=lambda d: -d['conf'])
        return dets, annotate(img, dets)


SIM_COLORS = {'fire': (255, 110, 20), 'person': (220, 40, 160), 'battery': (40, 90, 230), 'tree': (25, 85, 35), 'heater': (150, 150, 230)}


class SimDetector:
    ok = True
    name = 'SIM-детектор по цвету (не YOLO)'

    def detect(self, jpeg):
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(jpeg)).convert('RGB')
        a = np.asarray(img).astype(np.int16)
        dets = []
        import cv2
        for name, col in SIM_COLORS.items():
            m = (np.abs(a - np.array(col)).sum(axis=2) < 70).astype(np.uint8)
            n, _, st, _ = cv2.connectedComponentsWithStats(m)
            for i in range(1, n):                   # one box per object, like a real detector
                x, y, w, h, area = st[i]
                if area > 60:
                    dets.append({'name': name, 'conf': 0.9, 'box': [float(x), float(y), float(x + w), float(y + h)]})
        return dets, annotate(img, dets)


def annotate(img, dets):
    from PIL import ImageDraw
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for d in dets:
        x0, y0, x1, y1 = d['box']
        dr.rectangle([x0, y0, x1, y1], outline=(255, 230, 0), width=3)
        label = '%s %.2f' % (d['name'], d['conf'])
        dr.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(label), y0], fill=(255, 230, 0))
        dr.text((x0 + 3, max(0, y0 - 15)), label, fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return buf.getvalue()
