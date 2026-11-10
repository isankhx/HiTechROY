# Протокол борт ↔ сервер, HTTP API, форматы записей

## 1. Транспорт

TCP, борт подключается к серверу (`server.drone_port`, 7700). Кадр:

```
┌──────────┬───────────┬─────────────────┬──────────────┐
│ json_len │ bin_len   │ JSON (UTF-8)    │ payload      │
│ uint32 BE│ uint32 BE │ json_len байт   │ bin_len байт │
└──────────┴───────────┴─────────────────┴──────────────┘
```

Реализация: `board/protocol.py` (`pack`, `recv_blocking`, `recv_async`). Лимиты: JSON ≤ 1 МБ, payload ≤ 32 МБ. `NaN/Inf` в JSON запрещены (`allow_nan=False`): отсутствующие значения передаются как `null`.

## 2. Сообщения борт → сервер

**`hello`** — первое сообщение после подключения.
```json
{"type":"hello","id":"T1","agent":"agent-1.0","caps":{"thermal":true,"thermal_driver":"mlx_serial"}}
```

**`telem`** — 10 Гц.
```json
{"type":"telem","t":1794328098.5,
 "x":0.12,"y":-1.48,"z":1.01,"roll":0.01,"pitch":-0.02,"yaw":0.05,"vx":0.0,"vy":0.0,
 "armed":true,"mode":"OFFBOARD","connected":true,
 "voltage":24.6,"cell_voltage":4.1,
 "map_age":0.08,"markers":6,"manual":false,"guard":false,"wd":null}
```
`x y z` — в системе карты `aruco_map` (метры, z вверх), `null` если трансформация недоступна. `map_age` — секунды с последней позы по меткам (99 до первой). `manual` — пилот взял управление. `guard` — сторож дистанции остановил дрон. `wd` — состояние сторожа связи (`null`/`hold`/`land`).

**`rgb`** — JPEG в payload, ≤ `rgb_fps`. `{"type":"rgb","t":…}`

**`thermal`** — payload: `h×w` float32 little-endian, построчно, в исходной ориентации камеры.
```json
{"type":"thermal","t":…,"w":32,"h":24,"units":"C","pose":{"x":…,"y":…,"z":…,"roll":…,"pitch":…,"yaw":…}}
```
`units`: `C` — градусы, `raw` — условные единицы. `pose` — поза в момент кадра (может быть `null`, тогда сервер берёт телеметрию).

**`ack`** — ответ на команду. `{"type":"ack","req":17,"op":"goto","ok":false,"error":"Can't transform from aruco_map"}`. На успешный `goto` `ack` не шлётся (экономия), на остальные — всегда.

**`photo`** — ответ на запрос фото, JPEG в payload. `{"type":"photo","req":18,"pose":{…},"t":…}`

**`log`** — `{"type":"log","level":"warn","msg":"…"}` → журнал сервера.

## 3. Сообщения сервер → борт

**`hb`** — 5 Гц. Соседи и дистанция для сторожа.
```json
{"type":"hb","guard":1.06,"guard_landed":0.66,"neighbors":[{"id":"T2","x":-2.1,"y":-1.5,"z":1.3,"flying":true}]}
```
`guard` — минимальная дистанция до летящего соседа, `guard_landed` — до севшего (`flying:false`).
```
```

**`cmd`** — `{"type":"cmd","req":N,"op":…}`:

| op | поля | борт |
|---|---|---|
| `takeoff` | `z`, `speed` | `navigate(0,0,z, frame=body, auto_arm=true)` |
| `goto` | `x y z yaw speed` | проверка сторожа, `navigate(x,y,z,yaw, frame=aruco_map)` |
| `hold` | — | `navigate(0,0,0, frame=body)` — зависнуть |
| `land` | — | `land()` |

**`photo`** — `{"type":"photo","req":N}` → борт отвечает `photo` или отрицательным `ack`.

## 4. HTTP API сервера

Базовый адрес `http://127.0.0.1:8080`. Все POST без тела, ответ `{"ok":true}` или `409 {"detail":"причина"}`.

| метод | путь | назначение |
|---|---|---|
| GET | `/` | интерфейс |
| GET | `/api/config` | карта меток, границы, дроны, параметры тепловизора/осмотра |
| GET | `/api/state` | полный снимок состояния (то же, что событие `state`) |
| GET | `/api/events` | SSE: `state`, `heat`, `trails`, `thermal`, `log` |
| GET | `/api/video/{id}` | MJPEG с камеры дрона |
| GET | `/api/frame/{id}.jpg` | последний кадр |
| POST | `/api/start` | старт роя |
| POST | `/api/pause`, `/api/resume` | пауза (все зависают) / продолжение |
| POST | `/api/return` | всем домой и сесть |
| POST | `/api/land_all` | посадка всех на месте |
| POST | `/api/drone/{id}/land` | посадка одного |
| POST | `/api/drone/{id}/snapshot` | сохранить кадр камеры в `dataset/raw/` |
| POST | `/api/reset` | очистить карту и аномалии (вне миссии) |
| GET | `/api/anomaly/{id}` | аномалия с тепловым кадром |
| GET | `/api/photo/{name}` | фото текущей записи |
| GET | `/api/report.zip` | архив текущей записи |
| GET | `/api/sim/faults` | список отказов и активные (только `--sim`) |
| POST | `/api/sim/fault` | тело `{"drone":"T2","fault":"link","seconds":8,"value":…}` |

### Снимок состояния (`state`)

```json
{"mode":"sim|real","phase":"idle|takeoff|survey|return|done|aborted","phase_name":"…","paused":false,
 "detector":"YOLO rgb.pt (fire, person)","coverage":97.3,"frames":476,"pending":["A3"],
 "survey_path":[[x,y],…],"survey_info":{"swath_m":2.08,"lanes":3,"spacing_m":1.46},
 "D":1.16,"guard":1.06,"R":0.28,"run_dir":"runs/…",
 "drones":[{"id":"T1","name":"…","role":"thermal","color":"#…","prio":3,
            "connected":true,"link_age":0.05,"pos":[x,y,z],"yaw":0.0,"armed":true,"mode":"OFFBOARD",
            "manual":false,"voltage":24.8,"cell":4.13,"map_age":0.0,"markers":6,"locked":true,"lock_lost":false,
            "guard":false,"state":"тепловой облёт","task":"тепловой облёт","target":[x,y,z],"route":[[x,y,z],…],
            "blocked_by":[],"home":[x,y],"thermal":true,"rgb_age":0.2,"thermal_age":0.1}],
 "anomalies":[{"id":"A1","x":-0.4,"y":-0.6,"err":0.21,"dT_max":5.4,"observations":72,
               "t_first":…,"t_last":…,"status":"inspected","drone":"T2","label":"fire",
               "detections":[{"name":"fire","conf":0.9,"box":[x0,y0,x1,y1]}],
               "photo":"A1_T2.jpg","photo_annotated":"A1_T2_det.jpg","photo_pose":{…}}]}
```

Событие `heat`: сетка `nx×ny` байт в base64 (0 — нет данных, 1…255 — `ΔT` от `lo` до `hi`), начало `x0,y0`, шаг `cell`. Событие `thermal`: живой кадр `w×h` байт, растянутый по min/max кадра.

## 5. Записи `runs/<дата>/`

| файл | содержимое |
|---|---|
| `config.json` | конфигурация на момент старта |
| `events.jsonl` | `{"t":"ЧЧ:ММ:СС","wall":unix,"level":"info|ok|warn|err","msg":"…"}` |
| `telemetry.jsonl` | 2 Гц: `{"t","id","x","y","z","yaw","v","mode","armed","map_age","state"}` |
| `thermal.f32` | все сырые тепловые кадры подряд, float32 LE, как приняты |
| `thermal.jsonl` | индекс: `{"index","offset","nbytes","t","drone","w","h","units","pose","used"}`; `used` — кадр пошёл в карту |
| `photos/A1_T2.jpg`, `A1_T2_det.jpg` | фото осмотра и с разметкой YOLO |
| `anomalies.json` | аномалии как в `state`, плюс `thermal` — лучший кадр (`dT` список `w×h`, `bg`, `pose`, `frame_index`) |

Чтение теплового кадра из записи:
```python
import json, numpy as np
idx = [json.loads(l) for l in open('thermal.jsonl')]
raw = open('thermal.f32','rb').read()
r = idx[100]; frame = np.frombuffer(raw[r['offset']:r['offset']+r['nbytes']], '<f4').reshape(r['h'], r['w'])
```
