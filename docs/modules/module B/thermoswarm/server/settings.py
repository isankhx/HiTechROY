"""Operator-editable mission parameters (UI «Параметры»): whitelist with safe limits.
Changes are allowed only between missions and are saved to config.json."""
import json
import os

# (section, key, label, unit, min, max, step)
PARAMS = [
    ('thermal', 'alt_m', 'Высота теплового облёта (T1)', 'м', 1.2, 4.5, 0.1),
    ('thermal', 'speed_mps', 'Скорость теплового облёта', 'м/с', 0.1, 0.8, 0.05),
    ('thermal', 'overlap', 'Перекрытие галсов', '', 0.0, 0.6, 0.05),
    ('thermal', 'threshold', 'Порог нагрева ΔT', '°', 1.0, 10.0, 0.5),
    ('thermal', 'target_height_m', 'Высота крон (по ней считаются координаты)', 'м', 0.0, 2.5, 0.05),
    ('inspect', 'cruise_alt_m', 'Высота перелёта T2/T3', 'м', 0.8, 4.0, 0.1),
    ('inspect', 'photo_alt_m', 'Высота съёмки', 'м', 0.6, 3.5, 0.1),
    ('inspect', 'orbit_points', 'Ракурсов по кругу', '', 0, 8, 1),
    ('inspect', 'orbit_radius_m', 'Радиус облёта цели', 'м', 0.2, 1.0, 0.05),
    ('inspect', 'settle_s', 'Пауза перед снимком', 'с', 0.5, 5.0, 0.5),
    ('flight', 'speed_mps', 'Скорость T2/T3', 'м/с', 0.1, 1.0, 0.05),
    ('flight', 'takeoff_alt_m', 'Высота взлёта (поиск меток)', 'м', 0.6, 1.5, 0.1),
    ('separation', 'min_gap_m', 'Зазор между корпусами', 'м', 0.5, 1.5, 0.05),
    ('battery', 'cell_land_v', 'Посадка при напряжении элемента', 'В', 3.2, 3.7, 0.05),
]


def describe(cfg):
    return [{'section': s, 'key': k, 'label': lab, 'unit': u, 'min': lo, 'max': hi, 'step': st,
             'value': cfg.get(s, {}).get(k)} for s, k, lab, u, lo, hi, st in PARAMS]


def apply(cfg, changes):
    """Validate and apply {"section.key": value}; returns the list of applied changes."""
    spec = {'%s.%s' % (s, k): (lo, hi, st) for s, k, _, _, lo, hi, st in PARAMS}
    new = json.loads(json.dumps({s: cfg.get(s, {}) for s in {p[0] for p in PARAMS}}))
    done = []
    for name, v in changes.items():
        if name not in spec:
            raise ValueError('параметр %s нельзя менять из интерфейса' % name)
        lo, hi, st = spec[name]
        v = float(v)
        if not lo <= v <= hi:
            raise ValueError('%s: %g вне пределов %g…%g' % (name, v, lo, hi))
        if st == 1:
            v = int(round(v))
        s, k = name.split('.')
        if new[s].get(k) != v:
            new[s][k] = v
            done.append('%s = %g' % (name, v))
    if new['inspect']['photo_alt_m'] > new['inspect']['cruise_alt_m']:
        raise ValueError('высота съёмки не может быть выше высоты перелёта')
    crown = new['thermal'].get('target_height_m', 0.0)
    for name, alt, margin in (('высота теплового облёта', new['thermal']['alt_m'], 0.8),
                              ('высота перелёта T2/T3', new['inspect']['cruise_alt_m'], 0.6),
                              ('высота съёмки', new['inspect']['photo_alt_m'], 0.4)):
        if alt < crown + margin:
            raise ValueError('%s %.1f м слишком близко к кронам (%.1f м): нужно не ниже %.1f м' % (name, alt, crown, crown + margin))
    for s in new:
        cfg[s].update(new[s])   # keep dict identity: modules hold references to these sections
    return done


def save(cfg, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write('\n')
    os.replace(tmp, path)
