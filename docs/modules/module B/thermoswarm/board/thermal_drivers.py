"""Thermal camera drivers for the onboard agent. Python 3.6+, stdlib first.

Every driver: read() -> (rows, units) where rows is a list of lists (h x w)
or a numpy array, units 'C' (degrees) or 'raw'. Only temperature DIFFERENCES
are used on the server, so 'raw' intensity also works.

  none        - no thermal camera
  mlx_serial  - GY-MCU90640 / MLX90640 boards with the 0x5A5A binary stream (USB-UART)
  serial_csv  - any MCU (Arduino/ESP32 + MLX90640/AMG8833...) printing one frame
                per line: comma/space separated temperatures, w*h values
  uvc         - USB thermal camera seen as a webcam (/dev/videoN): grey levels
  uvc_split16 - USB cameras that put 16-bit temperatures in the lower half of a
  (= tr256i)    YUYV frame (Mileseey TR256i, InfiRay P2 Pro, TOPDON TC001 family): K*64 -> C
"""
import os
import select
import struct
import sys
import termios
import time

BAUDS = {9600: termios.B9600, 19200: termios.B19200, 38400: termios.B38400, 57600: termios.B57600,
         115200: termios.B115200, 230400: termios.B230400}
for _b in (460800, 921600, 1000000, 2000000):
    if hasattr(termios, 'B%d' % _b):
        BAUDS[_b] = getattr(termios, 'B%d' % _b)


class SerialPort:
    """Raw serial port via termios (no pyserial needed)."""

    def __init__(self, dev, baud):
        if baud not in BAUDS:
            raise ValueError('unsupported baud %s' % baud)
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = 0                                   # iflag
        a[1] = 0                                   # oflag
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] = 0                                   # lflag: raw
        a[4] = a[5] = BAUDS[baud]
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def read(self, n=4096, timeout=0.5):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return b''
        try:
            return os.read(self.fd, n)
        except BlockingIOError:
            return b''

    def write(self, data):
        os.write(self.fd, data)

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


class NoThermal:
    name = 'none'

    def read(self):
        time.sleep(1)
        return None, None

    def close(self):
        pass


class MlxSerial:
    """GY-MCU90640: 5A 5A 02 06 header, 768 int16 BE (C*100), ambient, checksum."""
    name = 'mlx_serial'
    HEADER = b'\x5a\x5a\x02\x06'
    HEADER_ALT = b'\x5a\x5a\x06\x02'
    SIZE = 1544

    def __init__(self, c):
        self.port = SerialPort(c.get('device', '/dev/ttyUSB0'), int(c.get('baud', 115200)))
        for h in c.get('init_hex', []):          # e.g. ["A5 25 01 CB"] if the module needs a start command
            self.port.write(bytes.fromhex(h))
            time.sleep(0.05)
        self.buf = bytearray()
        self.bad = 0
        self.t_last = time.time()

    def read(self):
        while True:
            data = self.port.read()
            if data:
                self.buf.extend(data)
            elif time.time() - self.t_last > 3:
                raise IOError('no frames from %s for 3 s' % self.name)
            while True:
                i = self.buf.find(b'\x5a\x5a')
                if i < 0:
                    del self.buf[:-1]
                    break
                del self.buf[:i]
                if len(self.buf) < self.SIZE:
                    break
                if bytes(self.buf[:4]) not in (self.HEADER, self.HEADER_ALT):
                    del self.buf[0]
                    continue
                frame = bytes(self.buf[:self.SIZE])
                body = frame[:-2]
                ok = (sum(body) & 0xFFFF) == struct.unpack('>H', frame[-2:])[0] or \
                     (sum(struct.unpack('<%dH' % (len(body) // 2), body)) & 0xFFFF) == struct.unpack('<H', frame[-2:])[0]
                if not ok:
                    self.bad += 1
                    del self.buf[0]
                    continue
                del self.buf[:self.SIZE]
                vals = struct.unpack('>768h', frame[4:1540])
                if min(vals) < -4000 or max(vals) > 30000:   # little-endian module variant
                    vals = struct.unpack('<768h', frame[4:1540])
                self.t_last = time.time()
                return [[vals[r * 32 + col] / 100.0 for col in range(32)] for r in range(24)], 'C'

    def close(self):
        self.port.close()


class SerialCsv:
    name = 'serial_csv'

    def __init__(self, c):
        self.port = SerialPort(c.get('device', '/dev/ttyUSB0'), int(c.get('baud', 115200)))
        self.w, self.h = int(c.get('width', 32)), int(c.get('height', 24))
        self.units = c.get('units', 'C')
        self.buf = b''
        self.t_last = time.time()

    def read(self):
        while True:
            data = self.port.read()
            if data:
                self.buf += data
            elif time.time() - self.t_last > 3:
                raise IOError('no lines from serial_csv for 3 s')
            while b'\n' in self.buf:
                line, self.buf = self.buf.split(b'\n', 1)
                parts = line.replace(b',', b' ').replace(b';', b' ').split()
                if len(parts) != self.w * self.h:
                    continue
                try:
                    v = [float(p) for p in parts]
                except ValueError:
                    continue
                self.t_last = time.time()
                return [v[r * self.w:(r + 1) * self.w] for r in range(self.h)], self.units
            if len(self.buf) > 1 << 20:
                self.buf = b''

    def close(self):
        self.port.close()


class Uvc:
    name = 'uvc'

    def __init__(self, c):
        import cv2
        self.cv2 = cv2
        dev = c.get('device', 0)
        self.cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev)
        if c.get('width'):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c['width']))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c['height']))
        if not self.cap.isOpened():
            raise IOError('cannot open %s' % dev)
        self.crop = c.get('crop')            # [x0, y0, x1, y1] to drop overlays
        self.invert = bool(c.get('invert', False))

    def read(self):
        ok, img = self.cap.read()
        if not ok:
            raise IOError('uvc read failed')
        if img.ndim == 3:
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        if self.crop:
            x0, y0, x1, y1 = self.crop
            img = img[y0:y1, x0:x1]
        img = img.astype('float32')
        if self.invert:
            img = 255 - img
        return img, 'raw'

    def close(self):
        self.cap.release()


class UvcSplit16(Uvc):
    """USB-C phone thermal cameras on a 256x192 core (Mileseey TR256i/TR256C, InfiRay P2,
    TOPDON TC001 ...). Over UVC they usually stream YUYV 256x384:
    upper half = picture, lower half = 16-bit temperature, value = Kelvin*64.
    If the camera sends only a picture (no temperature half), we fall back to
    grey levels (units 'raw'): only differences are used, so detection still works."""
    name = 'uvc_split16'

    def __init__(self, c):
        import cv2
        self.cv2 = cv2
        self.w, self.h = int(c.get('width', 256)), int(c.get('height', 192))
        dev = c.get('device', 'auto')
        if dev in (None, '', 'auto'):
            dev = find_thermal_uvc(self.w, self.h)
            if dev is None:
                raise IOError('thermal camera not found among /dev/video* (expected %dx%d YUYV); '
                              'run: python3 agent.py --probe-thermal' % (self.w, self.h * 2))
        self.device = dev
        self.cap = self._open(dev)
        link = stable_name(dev)
        print('thermal: %s%s' % (dev, (' = ' + link) if link else ''), flush=True)
        self.scale = float(c.get('scale', 1 / 64.0))
        self.offset = float(c.get('offset', -273.15))
        self.layout = None

    def _open(self, dev):
        cv2 = self.cv2
        cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            who = users_of(dev)
            raise IOError('cannot open %s%s' % (dev, (' — busy: ' + ', '.join(who) +
                          ' (stop it: sudo systemctl stop thermoswarm-agent, or fix the ROS camera device)') if who
                          else ' (busy or not a camera; check: v4l2-ctl --list-devices)'))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h * 2)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return cap

    def read(self):
        import numpy as np
        ok, raw = self.cap.read()
        if not ok or raw is None:
            who = users_of(self.device) if getattr(self, 'device', None) else []
            raise IOError('uvc read failed on %s%s' % (getattr(self, 'device', '?'), (' — busy: ' + ', '.join(who)) if who else ''))
        a = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
        W = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH)) or self.w
        H = int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT)) or self.h * 2
        if a.size >= W * H * 2 and H >= 2 * self.h - 2:
            if self.layout != 'split16':
                self.layout = 'split16'
                print('thermal: %dx%d YUYV, temperature half %dx%d (K*64)' % (W, H, W, H // 2), flush=True)
            t = a[:W * H * 2].view('<u2').reshape(H, W)[H // 2:]
            return t.astype('float32') * self.scale + self.offset, 'C'
        # picture only: take the luma channel
        if self.layout != 'grey':
            self.layout = 'grey'
            print('thermal: no temperature half in %dx%d frame -> grey levels (units raw)' % (W, H), flush=True)
        img = np.asarray(raw)
        if img.ndim == 3 and img.shape[2] == 2:
            img = img[:, :, 0]
        elif img.ndim == 3:
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2 and img.shape[1] == W * 2:
            img = img[:, 0::2]
        return img.astype('float32'), 'raw'


def _video_nodes():
    import glob
    # /dev/video0.. only (skip codec nodes like /dev/video-dec0)
    return sorted(glob.glob('/dev/video[0-9]*'), key=lambda p: int(''.join(ch for ch in p if ch.isdigit()) or 0))


def _v4l2_formats(dev):
    import subprocess
    try:
        return subprocess.check_output(['v4l2-ctl', '-d', dev, '--list-formats-ext'],
                                       stderr=subprocess.STDOUT, timeout=5).decode(errors='replace')
    except Exception:
        return None


def stable_name(dev):
    """/dev/v4l/by-id/... link for a /dev/videoN (survives renumbering after replugging)."""
    import glob
    for link in glob.glob('/dev/v4l/by-id/*'):
        if os.path.realpath(link) == os.path.realpath(dev):
            return link
    return None


def users_of(dev):
    """Processes holding the device (needs fuser; empty if unknown)."""
    import subprocess
    try:
        out = subprocess.check_output(['fuser', dev], stderr=subprocess.STDOUT, timeout=5).decode(errors='replace')
        pids = [p for p in out.replace(dev + ':', '').split() if p.strip().isdigit()]
        names = []
        for p in pids:
            try:
                names.append('%s(%s)' % (open('/proc/%s/comm' % p).read().strip(), p))
            except Exception:
                names.append(p)
        return names
    except Exception:
        return []


def find_thermal_uvc(w=256, h=192):
    """/dev/videoN of the thermal camera. First by the formats the device advertises
    (v4l2-ctl: YUYV w x 2h = picture + temperature, then w x h) - this does not open
    the device, so it works even if something else holds it. Fallback: try to open."""
    nodes = _video_nodes()
    for want in ('%dx%d' % (w, 2 * h), '%dx%d' % (w, h)):
        for dev in nodes:
            fm = _v4l2_formats(dev)
            if fm and 'YUYV' in fm and want in fm:
                return dev
    import cv2
    for dev in nodes:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h * 2)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            ok, _ = cap.read()
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if ok and W == w and H in (h, 2 * h):
                return dev
        finally:
            cap.release()
    return None


def describe_video_devices():
    """Text report for --probe-thermal: USB devices and V4L2 formats."""
    import subprocess
    out = []
    for cmd in (['lsusb'], ['v4l2-ctl', '--list-devices']):
        try:
            out.append('$ ' + ' '.join(cmd) + '\n' + subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5).decode(errors='replace'))
        except Exception as e:
            out.append('$ %s: %s' % (' '.join(cmd), e))
    import glob
    for dev in sorted(glob.glob('/dev/video*')):
        try:
            fm = subprocess.check_output(['v4l2-ctl', '-d', dev, '--list-formats-ext'], stderr=subprocess.STDOUT, timeout=5).decode(errors='replace')
            out.append('$ v4l2-ctl -d %s --list-formats-ext\n%s' % (dev, fm))
        except Exception as e:
            out.append('%s: %s' % (dev, e))
    return '\n'.join(out)


DRIVERS = {'none': NoThermal, 'mlx_serial': MlxSerial, 'serial_csv': SerialCsv, 'uvc': Uvc, 'uvc_split16': UvcSplit16,
           'tr256i': UvcSplit16}


def open_driver(c):
    name = (c or {}).get('driver', 'none')
    if name not in DRIVERS:
        raise ValueError('unknown thermal driver %r, choose: %s' % (name, ', '.join(DRIVERS)))
    return DRIVERS[name](c) if name != 'none' else NoThermal()


def binned(rows, k):
    """Block mean k x k (on-board reduction of Wi-Fi traffic: TR256i 256x192 -> 128x96 at k=2)."""
    if k <= 1:
        return rows
    import numpy as np
    a = np.asarray(rows, np.float32)
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def to_f32(rows):
    """(h, w, little-endian float32 bytes) from list-of-lists or numpy."""
    try:
        import numpy as np
        a = np.ascontiguousarray(np.asarray(rows, dtype='<f4'))
        return a.shape[0], a.shape[1], a.tobytes()
    except ImportError:
        from array import array
        h, w = len(rows), len(rows[0])
        arr = array('f', [v for r in rows for v in r])
        if sys.byteorder == 'big':
            arr.byteswap()
        return h, w, arr.tobytes()
