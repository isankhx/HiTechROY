"""2D/3D helpers. Separation is checked in XY only: a drone must never be
above another one (downwash, hot motors in the thermal image)."""
import math


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def d3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def point_seg(p, a, b):
    ax, ay = a[0], a[1]
    dx, dy = b[0] - ax, b[1] - ay
    L = dx * dx + dy * dy
    if L < 1e-12:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def seg_seg(a, b, c, d):
    """Minimum XY distance between segments ab and cd."""
    d1, d2_, d3_, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    if ((d1 > 0) != (d2_ > 0)) and ((d3_ > 0) != (d4 > 0)) and d1 != 0 and d2_ != 0 and d3_ != 0 and d4 != 0:
        return 0.0
    return min(point_seg(a, c, d), point_seg(b, c, d), point_seg(c, a, b), point_seg(d, a, b))


def point_poly(p, poly):
    if not poly:
        return float('inf')
    if len(poly) == 1:
        return d2(p, poly[0])
    return min(point_seg(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def seg_poly(a, b, poly):
    if not poly:
        return float('inf')
    if len(poly) == 1:
        return point_seg(poly[0], a, b)
    return min(seg_seg(a, b, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def toward(a, b, max_len):
    """Point from a toward b (3D) at most max_len away."""
    L = d3(a, b)
    if L <= max_len or L < 1e-9:
        return tuple(b)
    k = max_len / L
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


def truncate(poly, length):
    """Polyline prefix of the given XY length."""
    if not poly:
        return []
    out, left = [poly[0]], length
    for i in range(1, len(poly)):
        seg = d2(poly[i - 1], poly[i])
        if seg >= left:
            k = left / seg if seg > 1e-9 else 0
            out.append(tuple(poly[i - 1][j] + (poly[i][j] - poly[i - 1][j]) * k for j in range(len(poly[i]))))
            return out
        out.append(poly[i])
        left -= seg
    return out


def inside(p, bounds, inset=0.0):
    x0, x1, y0, y1 = bounds
    return x0 + inset <= p[0] <= x1 - inset and y0 + inset <= p[1] <= y1 - inset


def clamp_in(p, bounds, inset=0.0):
    x0, x1, y0, y1 = bounds
    return (min(max(p[0], x0 + inset), x1 - inset), min(max(p[1], y0 + inset), y1 - inset)) + tuple(p[2:])
