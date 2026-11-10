#!/usr/bin/env python3
"""ThermoSwarm ground server.

  python run_server.py --sim          # simulation, nothing leaves this computer
  python run_server.py                # real drones (agents connect to port 7700)
"""
import argparse
import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from server import clock  # noqa: E402
from server.field import load_map  # noqa: E402
from server.hub import Hub  # noqa: E402
from server.mission import Mission  # noqa: E402
from server.vision import SimDetector, YoloDetector  # noqa: E402
from server.web import Broadcaster, make_app  # noqa: E402


async def main(a):
    import uvicorn
    cfg = json.load(open(a.config, encoding='utf-8'))
    field = load_map(cfg['map_file'])
    print('Карта %s: %d меток, ID %d–%d, sha256 %s' % (cfg['map_file'], len(field['markers']),
          min(m['id'] for m in field['markers']), max(m['id'] for m in field['markers']), field['sha256'][:16]))
    mode = 'sim' if a.sim else 'real'
    bc = Broadcaster()
    hub = Hub(cfg, print)
    if a.sim and not os.path.exists(cfg['yolo']['weights']):
        detector = SimDetector()
    else:
        detector = YoloDetector(cfg['yolo']['weights'], cfg['yolo']['conf'], cfg['yolo']['imgsz'])
        if not detector.ok:
            print('YOLO: ' + str(detector.error))
    mission = Mission(cfg, hub, detector, mode, os.path.join(ROOT, 'runs'), bc)
    world = None
    s = cfg['server']
    drone_host = '127.0.0.1' if a.sim else s['drone_host']
    drone_port = a.drone_port or s['drone_port']
    await hub.start(drone_host, drone_port)
    asyncio.ensure_future(mission.run())
    if a.sim:
        from server.sim import run_sim
        world = await run_sim(cfg, field, drone_port)
        mission.log('warn', 'РЕЖИМ СИМУЛЯЦИИ: виртуальные дроны, реальные борта не подключаются')
    host, port = a.http_host or s['http_host'], a.http_port or s['http_port']
    mission.log('info', 'Детектор: ' + detector.name)
    mission.log('info', 'Интерфейс оператора: http://%s:%d' % ('127.0.0.1' if host == '0.0.0.0' else host, port))
    server = uvicorn.Server(uvicorn.Config(make_app(cfg, field, hub, mission, bc, world), host=host, port=port,
                                           log_level='warning', access_log=False))
    await server.serve()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='config.json')
    p.add_argument('--sim', action='store_true', help='simulated drones (separate from the real control loop)')
    p.add_argument('--speed', type=float, default=1.0, help='simulation time acceleration')
    p.add_argument('--http-host', help='0.0.0.0 = open the UI from other devices in the network')
    p.add_argument('--http-port', type=int)
    p.add_argument('--drone-port', type=int)
    a = p.parse_args()
    if a.speed != 1.0:
        if not a.sim:
            p.error('--speed only with --sim')
        clock.SCALE = a.speed
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
