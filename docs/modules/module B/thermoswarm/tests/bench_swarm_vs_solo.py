"""Benchmark: swarm (T1 thermal + T2/T3 photo) vs a single drone (T1 surveys, then
inspects every anomaly itself; T2/T3 are absent from the field).

Same simulated field, same hot objects, same parameters (config.json); for each
target layout (4 and 8 hot objects) and each mode, N series.

Metrics (simulated seconds = real seconds at 1x):
  mission_s     start -> all drones landed
  latency_s     per anomaly: confirmed by the thermal drone -> result of the RGB inspection
  inspected     anomalies with a finished RGB inspection / anomalies found
  coverage_pct  share of the field seen by the thermal camera
  min_dist_m    minimal distance between drone centres in flight (swarm only)

  python tests/bench_swarm_vs_solo.py [--series 2] [--speed 4]
Writes tests/bench_results.json.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTTP, DPORT = 8186, 7816

LAYOUTS = {
    4: None,  # the objects from config.json
    8: [
        {"x": -3.0, "y": 0.2, "dT": 6.0, "r": 0.12, "cls": "battery"},
        {"x": -1.9, "y": 0.9, "dT": 9.0, "r": 0.18, "cls": "fire"},
        {"x": -0.4, "y": -0.6, "dT": 5.0, "r": 0.15, "cls": None},
        {"x": 0.6, "y": 1.5, "dT": 7.0, "r": 0.15, "cls": "person"},
        {"x": 1.3, "y": -0.9, "dT": 8.0, "r": 0.15, "cls": "fire"},
        {"x": 2.6, "y": -0.2, "dT": 6.0, "r": 0.12, "cls": "battery"},
        {"x": 2.9, "y": 1.4, "dT": 7.0, "r": 0.15, "cls": "person"},
        {"x": -2.6, "y": 1.7, "dT": 6.5, "r": 0.14, "cls": "battery"},
    ],
}


def api(path, body=None, post=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (HTTP, path), data=data,
                                 method='POST' if (post or data is not None) else 'GET',
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def min_separation(run_dir):
    by = collections.defaultdict(dict)
    for l in open(os.path.join(run_dir, 'telemetry.jsonl')):
        r = json.loads(l)
        by[round(r['t'] * 2) / 2][r['id']] = r
    best = None
    for d in by.values():
        ids = [k for k, v in d.items() if v['x'] is not None and v['armed'] and (v['z'] or 0) > 0.3]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = d[ids[i]], d[ids[j]]
                dd = math.hypot(a['x'] - b['x'], a['y'] - b['y'])
                best = dd if best is None else min(best, dd)
    return best


def latencies(run_dir, speed):
    found, done = {}, {}
    for l in open(os.path.join(run_dir, 'events.jsonl')):
        e = json.loads(l)
        m = e['msg']
        if m.startswith('Тепловая аномалия '):
            found[m.split()[2].rstrip(':')] = e['wall']
        elif ': итог по ' in m:
            done[m.split(':')[0]] = e['wall']
    return [round((done[k] - found[k]) * speed, 1) for k in found if k in done]


def run_once(cfg_path, mode, speed):
    sim = 'T1' if mode == 'solo' else 'T1,T2,T3'
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'run_server.py'), '--config', cfg_path, '--sim',
                            '--speed', str(speed), '--sim-drones', sim, '--http-port', str(HTTP), '--drone-port', str(DPORT)],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env=dict(os.environ, PYTHONUNBUFFERED='1'))
    lines = []
    threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in srv.stdout], daemon=True).start()
    try:
        time.sleep(4)
        api('/api/start', {'drones': sim.split(',')})
        t0 = time.time()
        while True:
            time.sleep(1)
            s = api('/api/state')
            if s['phase'] == 'done':
                break
            if time.time() - t0 > 900:
                raise AssertionError('%s: mission not finished' % mode)
        errs = [l for l in lines if 'Ошибка цикла' in l or 'Traceback' in l]
        assert not errs, errs[:3]
        an = s['anomalies']
        lat = latencies(s['run_dir'], speed)
        return {'mode': mode, 'drones': sim, 'mission_s': s['mission_s'], 'coverage_pct': s['coverage'],
                'anomalies': len(an), 'inspected': sum(1 for a in an if a['status'] == 'inspected'),
                'photos': sum(len(a.get('photos') or []) for a in an),
                'latency_mean_s': round(sum(lat) / len(lat), 1) if lat else None, 'latency_max_s': max(lat) if lat else None,
                'labels': {a['id']: a.get('label') for a in an},
                'min_dist_m': None if mode == 'solo' else round(min_separation(s['run_dir']), 3),
                'run_dir': os.path.relpath(s['run_dir'], ROOT)}
    finally:
        srv.terminate()
        srv.wait(10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--series', type=int, default=2)
    p.add_argument('--speed', type=float, default=4)
    p.add_argument('--targets', default='4,8')
    a = p.parse_args()
    base = json.load(open(os.path.join(ROOT, 'config.json'), encoding='utf-8'))
    results = []
    for n in [int(x) for x in a.targets.split(',')]:
        cfg = json.loads(json.dumps(base))
        if LAYOUTS.get(n):
            cfg['sim']['objects'] = LAYOUTS[n]
        tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, dir=ROOT, prefix='bench_cfg_')
        json.dump(cfg, tmp, ensure_ascii=False)
        tmp.close()
        try:
            for i in range(a.series):
                for mode in ('swarm', 'solo'):
                    r = run_once(tmp.name, mode, a.speed)
                    r.update(targets=n, series=i + 1)
                    results.append(r)
                    print('%d targets, %-5s series %d: %5.0f s, inspected %d/%d, coverage %.0f%%, latency mean %s s, min dist %s' % (
                        n, mode, i + 1, r['mission_s'], r['inspected'], r['anomalies'], r['coverage_pct'],
                        r['latency_mean_s'], r['min_dist_m']), flush=True)
        finally:
            os.unlink(tmp.name)
    summary = {}
    for n in sorted({r['targets'] for r in results}):
        row = {}
        for mode in ('swarm', 'solo'):
            rs = [r for r in results if r['targets'] == n and r['mode'] == mode]
            row[mode] = {'mission_s': [r['mission_s'] for r in rs], 'mean_mission_s': round(sum(r['mission_s'] for r in rs) / len(rs), 1),
                         'inspected': ['%d/%d' % (r['inspected'], r['anomalies']) for r in rs],
                         'latency_mean_s': [r['latency_mean_s'] for r in rs]}
        row['time_ratio_solo_to_swarm'] = round(row['solo']['mean_mission_s'] / row['swarm']['mean_mission_s'], 2)
        summary['%d_targets' % n] = row
    out = {'date': time.strftime('%Y-%m-%d %H:%M'), 'speed': a.speed, 'results': results, 'summary': summary}
    json.dump(out, open(os.path.join(ROOT, 'tests', 'bench_results.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
