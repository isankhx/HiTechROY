"""Run folder: events, telemetry, raw thermal frames, photos, anomalies."""
import io
import json
import os
import time
import zipfile


class Recorder:
    def __init__(self, root, tag):
        self.dir = os.path.join(root, time.strftime('%Y%m%d-%H%M%S') + ('-' + tag if tag else ''))
        os.makedirs(os.path.join(self.dir, 'photos'), exist_ok=True)
        self._events = open(os.path.join(self.dir, 'events.jsonl'), 'a', encoding='utf-8')
        self._telem = open(os.path.join(self.dir, 'telemetry.jsonl'), 'a', encoding='utf-8')
        self._tbin = open(os.path.join(self.dir, 'thermal.f32'), 'ab')
        self._tidx = open(os.path.join(self.dir, 'thermal.jsonl'), 'a', encoding='utf-8')
        self.thermal_count = 0

    def event(self, rec):
        self._events.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self._events.flush()

    def telemetry(self, rec):
        self._telem.write(json.dumps(rec) + '\n')

    def thermal(self, meta, payload):
        """Raw frame exactly as received (float32 LE) + index line."""
        off = self._tbin.tell()
        self._tbin.write(payload)
        rec = dict(meta, offset=off, nbytes=len(payload), index=self.thermal_count)
        self._tidx.write(json.dumps(rec) + '\n')
        self.thermal_count += 1
        if self.thermal_count % 20 == 0:
            self._tbin.flush()
            self._tidx.flush()
        return rec['index']

    def photo(self, name, data):
        path = os.path.join(self.dir, 'photos', name)
        with open(path, 'wb') as f:
            f.write(data)
        return name

    def save_json(self, name, obj):
        tmp = os.path.join(self.dir, name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.dir, name))

    def flush(self):
        for f in (self._events, self._telem, self._tbin, self._tidx):
            f.flush()

    def zip_bytes(self):
        self.flush()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(self.dir):
                for fn in files:
                    if fn.endswith('.tmp'):
                        continue
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, self.dir))
        return buf.getvalue()

    def close(self):
        for f in (self._events, self._telem, self._tbin, self._tidx):
            try:
                f.close()
            except Exception:
                pass
