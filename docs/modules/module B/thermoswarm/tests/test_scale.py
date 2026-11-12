"""Scalability check: the same mission with 3 and with 5 simulated drones
(1 thermal + N-1 photo drones). Measures mission time, inspected anomalies,
minimal distance between drone centres and server CPU load.

  python tests/test_scale.py [--speed 3]
Writes tests/scale_results.json. Uses the 8-target layout of the benchmark.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_swarm_vs_solo import LAYOUTS, min_separation, latencies  # noqa: E402

HTTP, DPORT = 8187, 7817
STARTS = [[0.0, -1.5], [-2.25, -1.5], [2.25, -1.5], [-2.25, 1.5], [2.25, 1.5], [0.0, 1.5]]
COLORS = ['#ff7a45', '#4aa8ff', '#5fd17a', '#c77dff', '#ffd43b', '#4dd4c6']


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (HTTP, path), data=data, method='POST' if data else 'GET',
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def cpu_seconds(pid):
    f = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf('SC_CLK_TCK')


def run(n, speed):
    cfg = json.load(open(os.path.join(ROOT, 'config.json'), encoding='utf-8'))
    cfg['drones'] = [{'id': 'T1', 'name': 'T1 · тепло', 'role': 'thermal', 'priority': n + 1, 'color': COLORS[0]}] + [
        {'id': 'T%d' % (i + 1), 'name': 'T%d · фото' % (i + 1), 'role': 'rgb', 'priority': n - i, 'color': COLORS[i]} for i in range(1, n)]
    cfg['sim']['starts'] = STARTS[:n]
    cfg['sim']['objects'] = LAYOUTS[8]
    tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, dir=ROOT, prefix='scale_cfg_')
    json.dump(cfg, tmp, ensure_ascii=False)
    tmp.close()
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'run_server.py'), '--config', tmp.name, '--sim', '--speed', str(speed),
                            '--http-port', str(HTTP), '--drone-port', str(DPORT)], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    lines = []
    threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in srv.stdout], daemon=True).start()
    try:
        time.sleep(4)
        c0, w0 = cpu_seconds(srv.pid), time.time()
        api('/api/start', {'drones': [d['id'] for d in cfg['drones']]})
        while True:
            time.sleep(1)
            s = api('/api/state')
            if s['phase'] == 'done':
                break
            if time.time() - w0 > 900:
                raise AssertionError('not finished')
        cpu = (cpu_seconds(srv.pid) - c0) / (time.time() - w0)
        errs = [l for l in lines if 'Ошибка цикла' in l or 'Traceback' in l]
        assert not errs, errs[:3]
        an = s['anomalies']
        lat = latencies(s['run_dir'], speed)
        return {'drones': n, 'mission_s': s['mission_s'], 'anomalies': len(an),
                'inspected': sum(1 for a in an if a['status'] == 'inspected'),
                'latency_mean_s': round(sum(lat) / len(lat), 1) if lat else None,
                'min_dist_m': round(min_separation(s['run_dir']), 3),
                # the simulated drones (physics, camera rendering) run in the same process
                'server_cpu_share_of_one_core_incl_sim': round(cpu, 2), 'speed': speed,
                'run_dir': os.path.relpath(s['run_dir'], ROOT)}
    finally:
        srv.terminate()
        srv.wait(10)
        os.unlink(tmp.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speed', type=float, default=3)
    a = p.parse_args()
    res = []
    for n in (3, 5):
        r = run(n, a.speed)
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump({'date': time.strftime('%Y-%m-%d %H:%M'), 'results': res}, open(os.path.join(ROOT, 'tests', 'scale_results.json'), 'w'),
              ensure_ascii=False, indent=1)
    assert all(r['min_dist_m'] >= 1.06 for r in res), 'separation violated'
    assert all(r['inspected'] == r['anomalies'] for r in res), 'not all anomalies inspected'
    print('PASS')


if __name__ == '__main__':
    main()
