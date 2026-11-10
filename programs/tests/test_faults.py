"""Fault-scenario test on the simulator: every failure mode is injected during a
running mission and the expected reaction (log line, final state) is asserted.

  python tests/test_faults.py            # all scenarios (~5 min)
  python tests/test_faults.py link rc    # only these

Each scenario: start -> wait for 'survey' -> inject -> expect log text within
a timeout -> expect the mission to end with everyone on the ground.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTTP, DPORT, SPEED = 8184, 7814, 3

# name: (drone, fault, extra, expected log substring, timeout_s, allow_unreached_anomalies)
SCENARIOS = {
    'link':          ('T2', 'link', {}, 'T2: нет связи', 30, True),
    'map':           ('T2', 'map', {}, 'T2: метки снова видны', 20, False),
    'map_long':      ('T3', 'map', {'seconds': 20}, 'T3: метки не видны', 40, True),
    'rc':            ('T3', 'rc', {}, 'T3: ручное управление с пульта', 15, True),
    'battery':       ('T1', 'battery', {}, 'T1: батарея', 15, True),
    'thermal':       ('T1', 'thermal', {}, 'нет тепловых кадров', 20, True),
    'camera':        ('T2', 'camera', {}, 'T2: нет кадров с камеры', 60, False),
    'reject_goto':   ('T1', 'reject_goto', {'seconds': 4}, 'T1: команда goto не выполнена', 10, False),
    'wind':          ('T2', 'wind', {'seconds': 5}, 'T2', 20, False),
    'jump':          ('T2', 'jump', {}, 'T2: скачок позиции', 10, False),
    'latency':       ('T1', 'latency', {'seconds': 6}, 'T1', 20, False),
    'crash':         ('T3', 'crash', {}, 'T3: на земле (разоружён)', 15, True),
    'disarm':        ('T2', 'disarm', {}, 'T2: на земле (разоружён)', 20, True),
    'takeoff_refuse': ('T2', 'takeoff_refuse', {}, 'T2: взлёт отклонён', 20, True),
    'thermal_link':  ('T1', 'link', {}, 'Тепловой облёт прерван', 40, True),
}


def api(path, post=False, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (HTTP, path), data=data, method='POST' if post else 'GET',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise AssertionError('%s -> %s %s' % (path, e.code, e.read().decode()))


def wait_phase(phases, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = api('/api/state')
        if s['phase'] in phases:
            return s
        time.sleep(0.5)
    raise AssertionError('phase %s not reached in %ds (now %s)' % (phases, timeout, s['phase']))


def main(names):
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'run_server.py'), '--sim', '--speed', str(SPEED),
                            '--http-port', str(HTTP), '--drone-port', str(DPORT)],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = []
    threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in srv.stdout], daemon=True).start()
    results = {}
    try:
        time.sleep(3)
        for name in names:
            drone, fault, extra, expect, tmo, allow_miss = SCENARIOS[name]
            print('\n=== %s: %s %s %s' % (name, drone, fault, extra or ''))
            mark = len(lines)
            api('/api/reset', post=True)
            if fault == 'takeoff_refuse':
                api('/api/sim/fault', post=True, body=dict(drone=drone, fault=fault, **extra))
                api('/api/start', post=True)
            else:
                api('/api/start', post=True)
                wait_phase(['survey'], 40)
                time.sleep(3 / SPEED)
                api('/api/sim/fault', post=True, body=dict(drone=drone, fault=fault, **extra))
                mark = len(lines)
            t0 = time.time()
            seen = None
            while time.time() - t0 < tmo:
                seen = next((l for l in lines[mark:] if expect in l and 'СИМ-ОТКАЗ' not in l), None)
                if seen:
                    break
                time.sleep(0.3)
            assert seen, 'expected log %r not seen; tail:\n%s' % (expect, '\n'.join(lines[-15:]))
            print('   reaction: ' + seen.split('] ', 1)[-1])
            s = wait_phase(['done'], 150)
            for l in lines[mark:]:
                if 'Ошибка цикла' in l or 'Traceback' in l:
                    raise AssertionError('server exception: ' + l)
            flying = [d['id'] for d in s['drones'] if d['pos'] and d['pos'][2] > 0.1 and d['armed']]
            if fault in ('rc',):                       # pilot lands it manually after the mission
                api('/api/sim/fault', post=True, body=dict(drone=drone, fault='rc_release'))
                time.sleep(6 / SPEED)
                s = api('/api/state')
                flying = [d['id'] for d in s['drones'] if d['pos'] and d['pos'][2] > 0.1 and d['armed']]
            assert not flying, 'still flying: %s' % flying
            status = [(a['id'], a['status']) for a in s['anomalies']]
            bad = [x for x in status if x[1] not in ('inspected', 'ignored')]
            if bad and not allow_miss:
                raise AssertionError('anomalies not inspected: %s' % bad)
            api('/api/sim/fault', post=True, body=dict(drone=drone, fault='clear'))
            results[name] = 'OK  %s  anomalies=%s' % (seen.split('] ', 1)[-1][:70], status)
            print('   final: %s' % status)
            time.sleep(1)
        print('\nALL PASS: %d scenarios' % len(results))
    finally:
        srv.terminate()
        open(os.path.join(ROOT, 'tests', 'last_faults.log'), 'w').write('\n'.join(lines))
        print('server log: tests/last_faults.log')


if __name__ == '__main__':
    main(sys.argv[1:] or list(SCENARIOS))
