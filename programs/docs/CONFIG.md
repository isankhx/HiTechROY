# Настройки

## `config.json` (сервер)

| ключ | по умолчанию | смысл |
|---|---|---|
| `server.http_host/http_port` | `127.0.0.1:8080` | интерфейс оператора; `0.0.0.0` — доступ с других устройств |
| `server.drone_host/drone_port` | `0.0.0.0:7700` | порт для бортов (в `--sim` всегда только 127.0.0.1) |
| `map_file` | `aruco_map.txt` | карта меток в формате `aruco_pose`: `id length x y z rot_z rot_y rot_x` |
| `flight_bounds` | `[-3.6, 3.6, -2.1, 2.1]` | `[x0 x1 y0 y1]`, куда могут залетать **центры** дронов |
| `survey_bounds` | `[-3.75, 3.75, -2.25, 2.25]` | что покрывает тепловизор; тепловая сетка строится в этих границах |
| **separation** | | |
| `drone_radius_m` | 0.28 | радиус дрона с винтами — измерьте |
| `min_gap_m` | 0.5 | зазор между корпусами |
| `position_margin_m` | 0.1 | запас на ошибку позиции; `D = 2R + gap + margin` |
| `lookahead_m` | 2.5 | сколько пути вперёд защищается от стоящих дронов |
| **link** | | сторож связи (одинаково на борту и сервере) |
| `hold_after_s` / `land_after_s` | 1.5 / 6.0 | без `hb` — зависнуть / сесть |
| **localization** | | |
| `hold_after_s` / `land_after_s` | 2.0 / 10.0 | без меток — зависнуть / сесть |
| **battery** | | |
| `cell_warn_v` / `cell_land_v` | 3.5 / 3.4 | В на элемент: предупреждение / домой и сесть |
| **flight** | | |
| `takeoff_alt_m` | 1.0 | высота первичного подъёма до захвата меток |
| `takeoff_stagger_s` | 2.0 | интервал между взлётами |
| `lock_timeout_s` | 15 | не нашёл метки за это время → посадка |
| `speed_mps` | 0.4 | скорость T2/T3 и перелётов |
| `step_m` | 0.5 | длина шага; меньше — точнее дистанция, больше команд |
| `yaw_deg` | 0 | курс всех дронов в полёте (вдоль +X карты) |
| `max_mission_s` | 900 | лимит времени, затем возврат |
| **thermal** | | |
| `alt_m`, `speed_mps` | 2.0, 0.3 | высота и скорость облёта |
| `hfov_deg`, `vfov_deg` | 55, 35 | углы обзора тепловизора по ширине и высоте кадра |
| `rotate_deg`, `flip_h`, `flip_v` | 0, false, false | ориентация кадра: после них верх кадра = нос дрона |
| `overlap` | 0.3 | перекрытие галсов |
| `threshold` | 3.0 | аномалия: теплее фона на столько (°C или единиц `raw`) |
| `min_pixels` | 2 | минимальный размер пятна в пикселях |
| `target_height_m` | 0.0 | высота поверхности, где ищем нагрев (кроны — их высота) |
| `merge_radius_m` | 0.6 | наблюдения ближе этого — одна аномалия |
| `confirm_observations` | 2 | наблюдений до подтверждения |
| `max_tilt_deg` | 20 | кадры с большим креном не используются |
| `grid_cell_m` | 0.05 | ячейка тепловой карты |
| `display_range` | `[-2, 10]` | шкала ΔT на карте |
| **inspect** | | |
| `cruise_alt_m` / `photo_alt_m` | 1.3 / 1.0 | высота перелёта / съёмки |
| `settle_s` | 2.0 | успокоение перед фото |
| `task_timeout_s` | 90 | не добрался — пропуск |
| **yolo** | | |
| `weights` | `models/rgb.pt` | веса; нет файла → детектор отключён (в `--sim` — SIM-детектор) |
| `conf`, `imgsz` | 0.35, 640 | порог и размер входа |
| `ignore_classes` | `["drone"]` | классы, означающие «это дрон, не находка» |
| **drones[]** | | `id` (совпадает с `agent.json`), `name`, `role` (`thermal`/`rgb`), `priority`, `color` |
| **sim** | | стартовые точки, фон, нагретые объекты (`x y dT r cls`) |

Высоты: `thermal.alt_m` (2.0) выше `inspect.cruise_alt_m` (1.3) — это не защита (пролёт над другим запрещён в XY), а чтобы тепловизор видел широкую полосу, а фото было детальным.

## `board/agent.json` (борт)

| ключ | по умолчанию | смысл |
|---|---|---|
| `id` | `T1` | должен совпадать с `config.json → drones[].id` |
| `server`, `port` | — | адрес ноутбука |
| `camera_topic` | `/main_camera/image_raw/compressed` | топик JPEG-кадров |
| `map_pose_topic`, `markers_topic` | `/aruco_map/pose`, `/aruco_detect/markers` | откуда `map_age` и число меток |
| `map_frame` | `aruco_map` | система координат для `goto` и телеметрии |
| `takeoff_frame` | `body` | система для первичного подъёма |
| `srv_packages` | `["technic.srv","clover.srv"]` | где искать `Navigate`/`GetTelemetry` |
| `rgb_fps`, `telem_hz` | 5, 10 | частоты |
| `hold_after_s`, `land_after_s` | 1.5, 6.0 | сторож связи борта |
| `max_speed` | 1.0 | ограничение скорости любой команды |
| `thermal.driver` | `none` | `mlx_serial` / `serial_csv` / `uvc` / `uvc_split16` |
| `thermal.device`, `baud`, `fps`, `width`, `height`, `crop`, `invert`, `init_hex` | | параметры драйвера (см. `thermal_drivers.py`) |

Запуск: `python run_server.py [--sim] [--speed N] [--http-host 0.0.0.0] [--http-port P] [--drone-port P] [--config файл]`; борт: `python3 agent.py [--check] [--probe-thermal] [--id T2] [--server IP]`.
