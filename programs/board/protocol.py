"""Wire protocol between the onboard agent and the ground server.

Frame: 8-byte header '>II' (json_len, payload_len), UTF-8 JSON, raw payload.
Stdlib only, Python 3.6+ (Orange Pi ROS Noetic images). The same file is used
by the server (server/protocol.py imports it).
"""
import json
import struct

HDR = struct.Struct('>II')
MAX_JSON = 1 << 20
MAX_PAYLOAD = 32 << 20
DEFAULT_PORT = 7700


def pack(msg, payload=b''):
    data = json.dumps(msg, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return HDR.pack(len(data), len(payload)) + data + payload


def _check(jlen, plen):
    if jlen > MAX_JSON or plen > MAX_PAYLOAD:
        raise ValueError('frame too large: %d/%d' % (jlen, plen))


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('connection closed')
        buf.extend(chunk)
    return bytes(buf)


def recv_blocking(sock):
    """Blocking read of one frame from a socket. Returns (msg, payload)."""
    jlen, plen = HDR.unpack(_recv_exact(sock, HDR.size))
    _check(jlen, plen)
    msg = json.loads(_recv_exact(sock, jlen).decode('utf-8'))
    payload = _recv_exact(sock, plen) if plen else b''
    return msg, payload


async def recv_async(reader):
    """asyncio StreamReader version. Returns (msg, payload)."""
    jlen, plen = HDR.unpack(await reader.readexactly(HDR.size))
    _check(jlen, plen)
    msg = json.loads((await reader.readexactly(jlen)).decode('utf-8'))
    payload = (await reader.readexactly(plen)) if plen else b''
    return msg, payload


def finite(*values):
    for v in values:
        if v is None or not isinstance(v, (int, float)) or v != v or v in (float('inf'), float('-inf')):
            return False
    return True
