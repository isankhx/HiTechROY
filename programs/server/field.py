"""ArUco map file (aruco_pose format: id length x y z rot_z rot_y rot_x)."""
import hashlib


def load_map(path):
    raw = open(path, 'rb').read()
    markers = []
    for n, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError('%s:%d: expected "id length x y z ..."' % (path, n))
        mid, length, x, y, z = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        yaw = float(parts[5]) if len(parts) > 5 else 0.0
        markers.append({'id': mid, 'length': length, 'x': x, 'y': y, 'z': z, 'yaw': yaw})
    ids = [m['id'] for m in markers]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate marker ids in ' + path)
    return {'markers': markers, 'sha256': hashlib.sha256(raw).hexdigest(), 'file': path}
