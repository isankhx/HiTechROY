"""Integration test of the REAL control path: run_server.py (real mode, no --sim)
+ three board/agent.py instances on top of FakeRos. Mid-mission the link of T3
is cut: its agent must hover, then land by itself; T1/T2 finish the mission.

  python tests/test_real_mode.py
"""
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
from fake_ros import FakeRos, FakeThermal  # noqa: E402
import agent as board_agent  # noqa: E402  (board/ is on sys.path via fake_ros)

HTTP, DPORT = 8183, 7812


def api(path, post=False):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (HTTP, path), method='POST' if post else 'GET')
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    cfg = json.load(open(os.path.join(ROOT, 'config.json'), encoding='utf-8'))
    cfg['server'].update(drone_host='127.0.0.1', drone_port=DPORT, http_port=HTTP)
    cfg['flight']['speed_mps'] = 0.7
    cfg['thermal']['speed_mps'] = 0.6
    cfg['yolo']['weights'] = 'models/__none__.pt'
    tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, dir=ROOT)
    json.dump(cfg, tmp)
    tmp.close()
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, 'run_server.py'), '--config', tmp.name],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    threading.Thread(target=lambda: [lines.append(l.rstrip()) for l in srv.stdout], daemon=True).start()
    fakes, agents = [], {}
    try:
        time.sleep(2.5)
        starts = {'T1': (0.0, -1.5), 'T2': (-2.25, -1.5), 'T3': (2.25, -1.5)}
        for did, st in starts.items():
            ros = FakeRos(st, fakes)
            c = json.loads(json.dumps(board_agent.DEFAULTS))
            c.update(id=did, server='127.0.0.1', port=DPORT)
            if did == 'T1':
                c['thermal'] = {'driver': 'fake', 'fps': 8}
                FakeThermal.ros = ros
            ag = board_agent.Agent(c, ros)
            agents[did] = (ag, ros)
            threading.Thread(target=ag.run, daemon=True).start()
        time.sleep(3)
        s = api('/api/state')
        assert all(d['connected'] for d in s['drones']), 'agents did not connect'
        print('agents connected, starting')
        api('/api/start', post=True)
        t0, cut, phase = time.time(), False, None
        dropped = False
        min_sep = 9.0
        while time.time() - t0 < 240:
            time.sleep(0.5)
            s = api('/api/state')
            if s['phase'] != phase:
                phase = s['phase']
                print('%5.1fs phase %s' % (time.time() - t0, phase))
            fl = [agents[d][1] for d in agents if agents[d][1].armed and agents[d][1].pos[2] > 0.3]
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    dd = ((fl[i].pos[0] - fl[j].pos[0]) ** 2 + (fl[i].pos[1] - fl[j].pos[1]) ** 2) ** 0.5
                    min_sep = min(min_sep, dd)
            if not dropped and phase == 'survey' and time.time() - t0 > 9:
                agents['T2'][1].drop_map(4.0)
                dropped = time.time()
                print('%5.1fs >>> T2 loses the markers for 4 s' % (dropped - t0))
            if not cut and phase == 'survey' and time.time() - t0 > 20:
                ag, ros = agents['T3']
                ag.c['port'] = 1          # cannot reconnect
                ag.sock.shutdown(2)
                cut = time.time()
                print('%5.1fs >>> link to T3 cut' % (cut - t0))
            if phase == 'done':
                break
        r3 = agents['T3'][1]
        print('T3 calls after cut:', [c for c in r3.calls][-3:])
        print('T3 armed=%s z=%.2f mode=%s' % (r3.armed, r3.pos[2], r3.mode))
        anomalies = s['anomalies']
        print('anomalies:', [(a['id'], round(a['x'], 2), round(a['y'], 2), a['status']) for a in anomalies])
        print('min XY centre distance between flying drones: %.2f m (required %.2f)' % (min_sep, s['guard']))
        assert phase == 'done', 'mission did not finish: ' + str(phase)
        assert ('land',) in r3.calls and not r3.armed, 'T3 did not land after link loss'
        assert not any(agents[d][1].armed for d in agents), 'someone still armed'
        assert len(anomalies) == 4, 'expected 4 anomalies'
        assert min_sep >= s['guard'] - 0.02, 'separation violated'
        assert ('navigate', 'body', 0.0, 0.0, 0.0) in r3.calls[-6:], 'T3 did not hover first'
        assert any('T2: метки потеряны' in l for l in lines), 'server did not notice lost markers'
        assert any('T2: метки снова видны' in l for l in lines), 'server did not resume T2'
        assert not any('Ошибка цикла' in l for l in lines), 'exception in mission loop'
        print('PASS')
    finally:
        srv.terminate()
        os.unlink(tmp.name)
        open(os.path.join(ROOT, 'tests', 'last_server.log'), 'w').write('\n'.join(lines)); print('server log: tests/last_server.log')


if __name__ == '__main__':
    main()
