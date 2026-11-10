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
                YUYV frame (InfiRay P2 Pro / TOPDON TC001 family): K*64 -> C
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
    """Frame = [image half; 16-bit temperature half], value = Kelvin*64."""
    name = 'uvc_split16'

    def __init__(self, c):
        import cv2
        self.cv2 = cv2
        dev = c.get('device', 0)
        self.cap = cv2.VideoCapture(int(dev) if str(dev).isdigit() else dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened():
            raise IOError('cannot open %s' % dev)
        self.scale = float(c.get('scale', 1 / 64.0))
        self.offset = float(c.get('offset', -273.15))

    def read(self):
        import numpy as np
        ok, raw = self.cap.read()
        if not ok:
            raise IOError('uvc read failed')
        a = np.asarray(raw).reshape(-1)
        h2 = int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        a = a[:h2 * w * 2].view('<u2').reshape(h2, w)
        temp = a[h2 // 2:].astype('float32') * self.scale + self.offset
        return temp, 'C'


DRIVERS = {'none': NoThermal, 'mlx_serial': MlxSerial, 'serial_csv': SerialCsv, 'uvc': Uvc, 'uvc_split16': UvcSplit16}


def open_driver(c):
    name = (c or {}).get('driver', 'none')
    if name not in DRIVERS:
        raise ValueError('unknown thermal driver %r, choose: %s' % (name, ', '.join(DRIVERS)))
    return DRIVERS[name](c) if name != 'none' else NoThermal()


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
