"""Neural detector for thermal frames (case requirement К3-09).

One preprocessing function is shared by training (tools/thermal_dataset.py),
in-flight inference (server/mission.py) and evaluation (tools/thermal_eval.py):

    dT (frame minus its median, any resolution)  ->  fixed scale nn_range  ->
    8-bit grey  ->  bilinear resize to imgsz x imgsz*h/w  ->  RGB image

A FIXED temperature scale (no per-frame min/max stretching) keeps "how hot"
meaningful for the network and identical between training and flight.
"""
import os
import numpy as np

CLASS_NAME = 'thermal_anomaly'


def model_image(dT, nn_range=(-2.0, 12.0), imgsz=160):
    from PIL import Image
    lo, hi = nn_range
    g = np.uint8(np.clip((np.asarray(dT, np.float32) - lo) / (hi - lo), 0, 1) * 255)
    h, w = g.shape
    W, H = imgsz, max(1, int(round(imgsz * h / w)))
    return Image.fromarray(g).resize((W, H), Image.BILINEAR).convert('RGB'), (W / w, H / h)


class ThermalNN:
    def __init__(self, weights, conf=0.4, imgsz=160, nn_range=(-2.0, 12.0)):
        self.ok, self.error, self.model = False, None, None
        self.conf, self.imgsz, self.range = conf, imgsz, tuple(nn_range)
        self.name = 'ИК-нейросеть не загружена'
        if not weights or not os.path.exists(weights):
            self.error = 'нет файла весов %s' % weights
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), verbose=False)  # warm-up
            self.ok = True
            self.name = 'ИК-YOLO %s' % os.path.basename(weights)
        except Exception as e:
            self.error = str(e)

    def detect(self, dT):
        """Boxes in pixel coordinates of dT: [{'xyxy': [x0,y0,x1,y1], 'conf': c, 'cls': name}]. Blocking."""
        img, (sx, sy) = model_image(dT, self.range, self.imgsz)
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            out.append({'xyxy': [xyxy[0] / sx, xyxy[1] / sy, xyxy[2] / sx, xyxy[3] / sy],
                        'conf': round(float(c), 3), 'cls': self.model.names[int(k)]})
        return out
