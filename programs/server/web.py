"""HTTP API + operator UI. SSE for state, MJPEG for cameras, POST for commands."""
import asyncio
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from .thermal import public

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')


class Broadcaster:
    def __init__(self):
        self.clients = set()
        self.last = {}

    def __call__(self, event, data):
        if event in ('state', 'heat', 'trails'):
            self.last[event] = data
        msg = 'event: %s\ndata: %s\n\n' % (event, json.dumps(data, ensure_ascii=False, separators=(',', ':')))
        for q in list(self.clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(msg)


def make_app(cfg, field, hub, mission, bc, world=None):
    app = FastAPI(title='ThermoSwarm', docs_url=None, redoc_url=None)

    def ok(**kw):
        return JSONResponse(dict(ok=True, **kw))

    def guarded(fn):
        try:
            fn()
        except (RuntimeError, KeyError) as e:
            mission.log('err', 'Команда отклонена: %s' % e)
            raise HTTPException(409, str(e))
        return ok()

    @app.get('/')
    def index():
        return FileResponse(os.path.join(WEB, 'index.html'), headers={'Cache-Control': 'no-store'})

    @app.get('/web/{name}')
    def static(name: str):
        path = os.path.join(WEB, os.path.basename(name))
        if not os.path.exists(path):
            raise HTTPException(404)
        return FileResponse(path, headers={'Cache-Control': 'no-store'})

    @app.get('/api/config')
    def config():
        return {'markers': field['markers'], 'map_sha256': field['sha256'], 'map_file': field['file'],
                'flight_bounds': cfg['flight_bounds'], 'survey_bounds': cfg['survey_bounds'],
                'drones': cfg['drones'], 'thermal': cfg['thermal'], 'inspect': cfg['inspect'],
                'separation': cfg['separation'], 'mode': mission.mode}

    @app.get('/api/state')
    def state():
        return mission.state()

    @app.get('/api/events')
    async def events():
        q = asyncio.Queue(maxsize=60)
        bc.clients.add(q)

        async def gen():
            try:
                for ev in list(mission.events)[-80:]:
                    yield 'event: log\ndata: %s\n\n' % json.dumps(ev, ensure_ascii=False)
                for k, v in bc.last.items():
                    yield 'event: %s\ndata: %s\n\n' % (k, json.dumps(v, ensure_ascii=False))
                while True:
                    try:
                        yield await asyncio.wait_for(q.get(), 15)
                    except asyncio.TimeoutError:
                        yield ': ping\n\n'
            finally:
                bc.clients.discard(q)
        return StreamingResponse(gen(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.get('/api/video/{did}')
    async def video(did: str):
        d = hub.drones.get(did)
        if d is None:
            raise HTTPException(404)

        async def gen():
            seq = -1
            while True:
                async with d.rgb_cond:
                    try:
                        await asyncio.wait_for(d.rgb_cond.wait_for(lambda: d.rgb_seq != seq), 5)
                    except asyncio.TimeoutError:
                        continue
                seq, jpg = d.rgb_seq, d.rgb
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n'
        return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    @app.get('/api/frame/{did}.jpg')
    def frame(did: str):
        d = hub.drones.get(did)
        if d is None or d.rgb is None:
            raise HTTPException(404)
        return Response(d.rgb, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.post('/api/start')
    def start():
        return guarded(mission.start)

    @app.post('/api/pause')
    def pause():
        return guarded(lambda: mission.pause(True))

    @app.post('/api/resume')
    def resume():
        return guarded(lambda: mission.pause(False))

    @app.post('/api/return')
    def ret():
        return guarded(mission.return_home)

    @app.post('/api/land_all')
    def land_all():
        return guarded(mission.land_all)

    @app.post('/api/reset')
    def reset():
        return guarded(mission.reset)

    @app.post('/api/drone/{did}/land')
    def land(did: str):
        return guarded(lambda: mission.land_one(did))

    @app.post('/api/drone/{did}/snapshot')
    async def snapshot(did: str):
        try:
            return ok(file=await mission.snapshot(did))
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.get('/api/sim/faults')
    def sim_faults():
        if world is None:
            return {'enabled': False, 'faults': {}}
        from .sim import FAULT_NAMES
        return {'enabled': True, 'faults': FAULT_NAMES,
                'active': {d.id: sorted(d.f.keys()) + (['rc'] if d.manual else []) for d in world.drones.values()}}

    @app.post('/api/sim/fault')
    async def sim_fault(body: dict):
        if world is None:
            raise HTTPException(409, 'Инжекция отказов доступна только в симуляции')
        try:
            text = world.inject(body['drone'], body['fault'], body.get('seconds'), body.get('value'))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        mission.log('warn', 'СИМ-ОТКАЗ: ' + text)
        return ok(text=text)

    @app.get('/api/anomaly/{aid}')
    def anomaly(aid: str):
        a = mission.anomaly(aid)
        if a is None:
            raise HTTPException(404)
        return dict(public(a), thermal=a.get('thermal'))

    @app.get('/api/photo/{name}')
    def photo(name: str):
        if not mission.rec:
            raise HTTPException(404)
        path = os.path.join(mission.rec.dir, 'photos', os.path.basename(name))
        if not os.path.exists(path):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get('/api/report.zip')
    def report():
        if not mission.rec:
            raise HTTPException(404, 'Миссия ещё не запускалась')
        mission.save_anomalies()
        name = os.path.basename(mission.rec.dir) + '.zip'
        return Response(mission.rec.zip_bytes(), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="%s"' % name})

    return app
