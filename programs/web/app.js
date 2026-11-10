'use strict';
// ThermoSwarm operator UI: vanilla JS, works offline.
const $ = (s) => document.querySelector(s);
const S = { cfg: null, state: null, heat: null, trails: {}, live: null, layers: {}, hits: [] };

// ---------- colormap (inferno-like) ----------
const STOPS = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
const LUT = new Uint8ClampedArray(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255 * (STOPS.length - 1), k = Math.min(Math.floor(t), STOPS.length - 2), f = t - k;
  for (let c = 0; c < 3; c++) LUT[i * 3 + c] = STOPS[k][c] * (1 - f) + STOPS[k + 1][c] * f;
}
const lutCss = (i) => `rgb(${LUT[i * 3]},${LUT[i * 3 + 1]},${LUT[i * 3 + 2]})`;
$('#heatScale').style.background = `linear-gradient(90deg, ${[0, 64, 128, 192, 255].map(lutCss).join(',')})`;

const STATUS = { new: 'наблюдение', confirmed: 'ждёт осмотра', assigned: 'дрон летит', inspected: 'осмотрена', ignored: 'дрон?',
  photo_failed: 'нет фото', unreachable: 'недоступна', not_inspected: 'не осмотрена' };
const AN_COLOR = { new: '#8b97a6', confirmed: '#ff9f43', assigned: '#ffd43b', inspected: '#ff4d6d' };

// ---------- API ----------
async function post(url) {
  try {
    const r = await fetch(url, { method: 'POST' });
    if (!r.ok) { const j = await r.json().catch(() => ({})); addLog({ t: now(), level: 'err', msg: j.detail || r.statusText }); }
  } catch (e) { addLog({ t: now(), level: 'err', msg: 'Сервер недоступен: ' + e }); }
}
const now = () => new Date().toTimeString().slice(0, 8);
$('#btnStart').onclick = () => post('/api/start');
$('#btnPause').onclick = () => post(S.state && S.state.paused ? '/api/resume' : '/api/pause');
$('#btnHome').onclick = () => post('/api/return');
$('#btnLand').onclick = () => post('/api/land_all');
$('#btnReset').onclick = () => post('/api/reset');
document.querySelectorAll('[data-layer]').forEach((el) => {
  S.layers[el.dataset.layer] = el.checked;
  el.onchange = () => { S.layers[el.dataset.layer] = el.checked; drawMap(); };
});

// ---------- map ----------
const cv = $('#map'), ctx = cv.getContext('2d');
let view = null;
function setupView() {
  const b = S.cfg.survey_bounds, pad = 0.35;
  const W = cv.clientWidth, worldW = b[1] - b[0] + 2 * pad, worldH = b[3] - b[2] + 2 * pad;
  const H = Math.round(W * worldH / worldW), dpr = window.devicePixelRatio || 1;
  cv.style.height = H + 'px'; cv.width = W * dpr; cv.height = H * dpr;
  const k = W / worldW;
  view = { k, dpr, W, H, x0: b[0] - pad, y1: b[3] + pad };
  drawMap();
}
const X = (x) => (x - view.x0) * view.k, Y = (y) => (view.y1 - y) * view.k;
const invX = (px) => px / view.k + view.x0, invY = (py) => view.y1 - py / view.k;

const heatCanvas = document.createElement('canvas');
function buildHeat() {
  const h = S.heat; if (!h) return;
  const raw = atob(h.data), n = h.nx * h.ny;
  heatCanvas.width = h.nx; heatCanvas.height = h.ny;
  const hc = heatCanvas.getContext('2d'), img = hc.createImageData(h.nx, h.ny);
  for (let i = 0; i < n; i++) {
    const q = raw.charCodeAt(i), row = Math.floor(i / h.nx), col = i % h.nx;
    const o = ((h.ny - 1 - row) * h.nx + col) * 4;
    if (!q) { img.data[o + 3] = 0; continue; }
    const v = h.lo + (q - 1) / 254 * (h.hi - h.lo);        // ΔT, °C
    const t = Math.max(0, Math.min(1, v / h.hi));
    const li = Math.round(40 + t * 215);
    img.data[o] = LUT[li * 3]; img.data[o + 1] = LUT[li * 3 + 1]; img.data[o + 2] = LUT[li * 3 + 2];
    img.data[o + 3] = Math.round(255 * (0.32 + 0.63 * t));
  }
  hc.putImageData(img, 0, 0);
  $('#heatRange').textContent = `ΔT 0 … +${h.hi}°`;
}

function drawMap() {
  if (!view || !S.cfg) return;
  const c = ctx, k = view.k, st = S.state;
  c.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  c.clearRect(0, 0, view.W, view.H);
  const sb = S.cfg.survey_bounds, fb = S.cfg.flight_bounds;
  c.fillStyle = '#1a1f26';
  c.fillRect(X(sb[0]), Y(sb[3]), (sb[1] - sb[0]) * k, (sb[3] - sb[2]) * k);
  // grid every 0.75 m
  c.strokeStyle = '#232a33'; c.lineWidth = 1; c.beginPath();
  for (let x = Math.ceil(sb[0] / 0.75) * 0.75; x <= sb[1]; x += 0.75) { c.moveTo(X(x), Y(sb[2])); c.lineTo(X(x), Y(sb[3])); }
  for (let y = Math.ceil(sb[2] / 0.75) * 0.75; y <= sb[3]; y += 0.75) { c.moveTo(X(sb[0]), Y(y)); c.lineTo(X(sb[1]), Y(y)); }
  c.stroke();
  if (S.layers.heat && S.heat) {
    c.imageSmoothingEnabled = true;
    const h = S.heat;
    c.drawImage(heatCanvas, X(h.x0), Y(h.y0 + h.ny * h.cell), h.nx * h.cell * k, h.ny * h.cell * k);
  }
  c.setLineDash([6, 4]); c.strokeStyle = '#3a4350';
  c.strokeRect(X(fb[0]), Y(fb[3]), (fb[1] - fb[0]) * k, (fb[3] - fb[2]) * k); c.setLineDash([]);
  if (S.layers.markers) {
    c.font = `${Math.max(9, k * 0.09)}px ui-monospace, monospace`; c.textAlign = 'center'; c.textBaseline = 'middle';
    for (const m of S.cfg.markers) {
      const L = m.length * k;
      c.strokeStyle = m.id === 122 ? '#4aa8ff' : '#56606d'; c.lineWidth = m.id === 122 ? 2 : 1;
      c.strokeRect(X(m.x) - L / 2, Y(m.y) - L / 2, L, L);
      if (k > 45) { c.fillStyle = '#6b7685'; c.fillText(m.id, X(m.x), Y(m.y)); }
    }
    // origin axes
    c.strokeStyle = '#ff6b6b'; c.lineWidth = 2; c.beginPath(); c.moveTo(X(0), Y(0)); c.lineTo(X(0.5), Y(0)); c.stroke();
    c.strokeStyle = '#5fd17a'; c.beginPath(); c.moveTo(X(0), Y(0)); c.lineTo(X(0), Y(0.5)); c.stroke();
  }
  if (!st) return;
  if (S.layers.route && st.survey_path.length) {
    c.strokeStyle = 'rgba(255,122,69,.45)'; c.setLineDash([3, 5]); c.lineWidth = 1.5; c.beginPath();
    st.survey_path.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1]))));
    c.stroke(); c.setLineDash([]);
  }
  if (S.layers.trail) {
    for (const d of st.drones) {
      const tr = (S.trails[d.id] || []).slice();
      if (d.pos) tr.push([d.pos[0], d.pos[1]]);
      if (tr.length < 2) continue;
      c.strokeStyle = d.color; c.globalAlpha = 0.55; c.lineWidth = 1.6; c.beginPath();
      tr.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1]))));
      c.stroke(); c.globalAlpha = 1;
    }
  }
  // anomalies
  S.hits = [];
  for (const a of st.anomalies) {
    const col = AN_COLOR[a.status] || '#6b7280', r = Math.max(7, a.err * k);
    c.fillStyle = col + '33'; c.strokeStyle = col; c.lineWidth = 2;
    c.beginPath(); c.arc(X(a.x), Y(a.y), r, 0, 7); c.fill(); c.stroke();
    c.beginPath(); c.arc(X(a.x), Y(a.y), 3, 0, 7); c.fillStyle = col; c.fill();
    const label = a.id + (a.label ? ' · ' + a.label : (a.status === 'inspected' ? ' · ?' : ''));
    c.font = '600 12px system-ui, sans-serif'; c.textAlign = 'left'; c.textBaseline = 'bottom';
    c.lineWidth = 3; c.strokeStyle = '#0a0d12'; c.strokeText(label, X(a.x) + r + 2, Y(a.y) - 2);
    c.fillStyle = '#fff'; c.fillText(label, X(a.x) + r + 2, Y(a.y) - 2);
    S.hits.push({ x: X(a.x), y: Y(a.y), r: r + 6, id: a.id });
  }
  // drones
  const R = st.R, gapR = st.guard / 2;
  for (const d of st.drones) {
    if (!d.pos) continue;
    const x = X(d.pos[0]), y = Y(d.pos[1]);
    if (S.layers.route && d.route.length) {
      c.strokeStyle = d.color; c.globalAlpha = 0.8; c.setLineDash([5, 4]); c.lineWidth = 1.5; c.beginPath(); c.moveTo(x, y);
      if (d.target) c.lineTo(X(d.target[0]), Y(d.target[1]));
      d.route.forEach((p) => c.lineTo(X(p[0]), Y(p[1]))); c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
    }
    if (S.layers.zone) {
      c.strokeStyle = d.blocked_by.length ? '#ffd43b' : d.color; c.globalAlpha = 0.5; c.setLineDash([2, 3]); c.lineWidth = 1;
      c.beginPath(); c.arc(x, y, gapR * k, 0, 7); c.stroke(); c.setLineDash([]); c.globalAlpha = 1;
    }
    c.fillStyle = d.color + (d.armed ? 'cc' : '55'); c.strokeStyle = '#fff'; c.lineWidth = 1.5;
    c.beginPath(); c.arc(x, y, R * k, 0, 7); c.fill(); c.stroke();
    if (d.yaw != null) {
      const cs = Math.cos(d.yaw), sn = Math.sin(d.yaw);
      c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.moveTo(x + cs * R * k, y - sn * R * k);
      c.lineTo(x + cs * R * k * 1.45, y - sn * R * k * 1.45); c.stroke();
    }
    c.font = '700 12px system-ui, sans-serif'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#fff';
    c.fillText(d.id, x, y);
    c.font = '11px ui-monospace, monospace'; c.fillStyle = '#d0d7de';
    c.fillText(d.pos[2].toFixed(1) + ' м', x, y + R * k + 9);
  }
}

cv.addEventListener('mousemove', (e) => {
  if (!view) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  $('#cursor').textContent = `x ${invX(px).toFixed(2)}  y ${invY(py).toFixed(2)} м`;
  cv.style.cursor = S.hits.some((h) => Math.hypot(h.x - px, h.y - py) < h.r) ? 'pointer' : 'crosshair';
});
cv.addEventListener('click', (e) => {
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const h = S.hits.find((h) => Math.hypot(h.x - px, h.y - py) < h.r);
  if (h) openAnomaly(h.id);
});
window.addEventListener('resize', () => S.cfg && setupView());

// ---------- drones & cameras ----------
function fmt(v, d = 1, unit = '') { return v == null ? '—' : Number(v).toFixed(d) + unit; }
function buildDrones() {
  const box = $('#drones'), cams = $('#cams');
  box.innerHTML = ''; cams.innerHTML = '';
  for (const d of S.cfg.drones) {
    const el = document.createElement('div');
    el.className = 'drone'; el.style.setProperty('--c', d.color); el.id = 'dr-' + d.id;
    el.innerHTML = `<span class="sw"></span><div><span class="name">${d.name || d.id}</span><span class="state"></span></div>
      <div class="btns"><button class="btn sm ghost" data-snap title="Сохранить кадр в dataset/raw для обучения YOLO">📷</button>
      <button class="btn sm outline-danger" data-land>Посадить</button></div><div class="kv"></div>`;
    el.querySelector('[data-land]').onclick = () => post(`/api/drone/${d.id}/land`);
    el.querySelector('[data-snap]').onclick = () => post(`/api/drone/${d.id}/snapshot`);
    box.appendChild(el);
    const cam = document.createElement('div');
    cam.className = 'cam'; cam.style.setProperty('--c', d.color); cam.id = 'cam-' + d.id;
    cam.innerHTML = `<img src="/api/video/${d.id}" alt=""><div class="nofeed">нет видео</div><span class="lbl">${d.id} · RGB</span><span class="info mono"></span>`;
    cams.appendChild(cam);
  }
  const th = S.cfg.drones.find((d) => d.role === 'thermal');
  const tc = document.createElement('div');
  tc.className = 'cam'; tc.style.setProperty('--c', th ? th.color : '#ff7a45');
  tc.innerHTML = `<canvas id="liveThermal" width="320" height="240"></canvas><div class="nofeed" id="thNo">нет тепловых кадров</div>
    <span class="lbl">${th ? th.id : ''} · тепловизор</span><span class="info mono" id="thInfo"></span>`;
  cams.appendChild(tc);
}

function updateDrones() {
  for (const d of S.state.drones) {
    const el = document.getElementById('dr-' + d.id); if (!el) continue;
    el.querySelector('.state').textContent = d.state;
    const link = !d.connected ? '<b class="bad">нет</b>' : `<b class="${d.link_age > 0.8 ? 'mid' : 'good'}">${(d.link_age * 1000).toFixed(0)} мс</b>`;
    const lock = d.map_age == null || d.map_age > 50 ? '<b class="bad">нет</b>'
      : `<b class="${d.map_age < 0.5 ? 'good' : d.map_age < 2 ? 'mid' : 'bad'}">${d.map_age.toFixed(1)} с</b>`;
    const cell = d.cell != null ? ` (${d.cell.toFixed(2)})` : '';
    el.querySelector('.kv').innerHTML =
      `<span>связь ${link}</span><span>АКБ <b class="${d.cell != null && d.cell < 3.5 ? 'bad' : ''}">${fmt(d.voltage, 1, ' В')}${cell}</b></span>` +
      `<span>режим <b>${d.mode || '—'}${d.armed ? ' ✈' : ''}</b></span><span>метки <b>${d.markers ?? '—'}</b> · карта ${lock}</span>` +
      `<span>xyz <b>${d.pos ? d.pos.map((v) => v.toFixed(2)).join(' ') : '—'}</b></span>` +
      (d.guard ? '<span><b class="bad">СТОП: сосед ближе 0.5 м</b></span>' : '') +
      (d.thermal_age != null ? `<span>тепловизор <b class="${d.thermal_age > 3 ? 'bad' : 'good'}">${d.thermal_age > 3 ? 'нет кадров' : 'ок'}</b></span>` : '') +
      (d.lock_lost ? '<span><b class="bad">НЕТ МЕТОК: зависание</b></span>' : '');
    const cam = document.getElementById('cam-' + d.id);
    if (cam) {
      cam.querySelector('.nofeed').style.display = d.rgb_age == null || d.rgb_age > 3 ? 'grid' : 'none';
      cam.querySelector('.info').textContent = d.pos ? `z ${d.pos[2].toFixed(2)} м · ${d.state}` : d.state;
    }
  }
}

function drawLive() {
  const L = S.live, c = document.getElementById('liveThermal'); if (!L || !c) return;
  const off = document.createElement('canvas'); off.width = L.w; off.height = L.h;
  const oc = off.getContext('2d'), img = oc.createImageData(L.w, L.h), raw = atob(L.data);
  for (let i = 0; i < L.w * L.h; i++) {
    const q = raw.charCodeAt(i);
    img.data[i * 4] = LUT[q * 3]; img.data[i * 4 + 1] = LUT[q * 3 + 1]; img.data[i * 4 + 2] = LUT[q * 3 + 2]; img.data[i * 4 + 3] = 255;
  }
  oc.putImageData(img, 0, 0);
  const g = c.getContext('2d'); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.drawImage(off, 0, 0, c.width, c.height);
  $('#thNo').style.display = 'none';
  const u = L.units === 'C' ? '°' : '';
  $('#thInfo').textContent = `фон ${L.bg.toFixed(1)}${u} · ΔT ${L.lo.toFixed(1)}…+${L.hi.toFixed(1)}${u}`;
}

// ---------- header, table, log ----------
function updateHeader() {
  const st = S.state;
  const badge = $('#modeBadge');
  badge.textContent = st.mode === 'sim' ? 'СИМУЛЯЦИЯ' : 'РЕАЛЬНЫЕ ДРОНЫ';
  badge.className = 'badge ' + st.mode;
  $('#phase').textContent = st.phase_name + (st.paused ? ' · ПАУЗА' : '');
  const running = ['takeoff', 'survey', 'return'].includes(st.phase);
  $('#phaseDot').className = 'dot ' + (st.phase === 'aborted' ? 'alarm' : st.paused ? 'pause' : running ? 'run' : '');
  $('#cov').textContent = st.coverage.toFixed(0);
  $('#frames').textContent = st.frames;
  $('#nAn').textContent = st.anomalies.length;
  $('#nIns').textContent = st.anomalies.filter((a) => a.status === 'inspected').length;
  $('#btnStart').disabled = running || st.phase === 'aborted';
  $('#btnPause').disabled = !running;
  $('#btnPause').textContent = st.paused ? '▶ Продолжить' : '⏸ Пауза';
  $('#btnHome').disabled = !['takeoff', 'survey'].includes(st.phase);
  $('#btnReset').disabled = running;
  $('#detName').textContent = st.detector;
  $('#sepInfo').textContent = `зазор ${(st.guard - 2 * st.R).toFixed(2)} м, центры ≥ ${st.D} м`;
}

let tableKey = '';
function updateTable() {
  const an = S.state.anomalies;
  const key = JSON.stringify(an.map((a) => [a.id, a.status, a.label, a.observations, a.drone, Math.round(a.x * 100), Math.round(a.y * 100)]));
  if (key === tableKey) return; tableKey = key;
  $('#anEmpty').style.display = an.length ? 'none' : 'block';
  $('#anTable tbody').innerHTML = an.map((a) => {
    const det = a.detections && a.detections.length ? a.detections.map((d) => `${d.name} ${(d.conf * 100).toFixed(0)}%`).join(', ')
      : (a.status === 'inspected' ? 'не распознан' : '—');
    return `<tr data-id="${a.id}"><td><b>${a.id}</b></td><td>${a.x.toFixed(2)}</td><td>${a.y.toFixed(2)}</td><td>${a.err.toFixed(2)}</td>` +
      `<td>+${a.dT_max.toFixed(1)}</td><td>${a.observations}</td><td><span class="st ${a.status}">${STATUS[a.status] || a.status}</span></td>` +
      `<td>${a.drone || '—'}</td><td>${det}</td></tr>`;
  }).join('');
  document.querySelectorAll('#anTable tbody tr').forEach((tr) => (tr.onclick = () => openAnomaly(tr.dataset.id)));
}

function addLog(ev) {
  const div = document.createElement('div');
  div.innerHTML = `<span class="t">${ev.t}</span><span class="${ev.level}"></span>`;
  div.lastChild.textContent = ev.msg;
  const log = $('#log'); log.prepend(div);
  while (log.childNodes.length > 300) log.removeChild(log.lastChild);
}

// ---------- anomaly modal ----------
let modalId = null;
async function openAnomaly(id) {
  modalId = id;
  const r = await fetch('/api/anomaly/' + id); if (!r.ok) return;
  const a = await r.json();
  $('#modal').hidden = false;
  $('#mTitle').textContent = `Аномалия ${a.id} · ${STATUS[a.status] || a.status}`;
  const c = $('#mThermal'), g = c.getContext('2d');
  g.fillStyle = '#07090c'; g.fillRect(0, 0, c.width, c.height);
  if (a.thermal) {
    const T = a.thermal, off = document.createElement('canvas'); off.width = T.w; off.height = T.h;
    const oc = off.getContext('2d'), img = oc.createImageData(T.w, T.h);
    const lo = Math.min(...T.dT), hi = Math.max(...T.dT), span = Math.max(hi - lo, 0.5);
    let peak = 0;
    T.dT.forEach((v, i) => {
      const q = Math.round((v - lo) / span * 255);
      img.data.set([LUT[q * 3], LUT[q * 3 + 1], LUT[q * 3 + 2], 255], i * 4);
      if (v > T.dT[peak]) peak = i;
    });
    oc.putImageData(img, 0, 0);
    c.height = Math.round(c.width * T.h / T.w);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(off, 0, 0, c.width, c.height);
    const sx = c.width / T.w, px = (peak % T.w + 0.5) * sx, py = (Math.floor(peak / T.w) + 0.5) * sx;
    g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(px, py, 12, 0, 7); g.stroke();
    $('#mThermalInfo').textContent = `фон ${T.bg}° · ΔT ${lo.toFixed(1)}…+${hi.toFixed(1)} · дрон z=${T.pose.z} м · ${T.w}×${T.h}`;
  } else $('#mThermalInfo').textContent = 'кадр недоступен';
  const showPhoto = () => {
    const name = $('#mAnn').checked && a.photo_annotated ? a.photo_annotated : a.photo;
    $('#mPhoto').style.display = name ? 'block' : 'none'; $('#mNoPhoto').style.display = name ? 'none' : 'block';
    if (name) $('#mPhoto').src = '/api/photo/' + encodeURIComponent(name);
  };
  $('#mAnn').onchange = showPhoto; showPhoto();
  $('#mDets').innerHTML = (a.detections || []).map((d) => `<span class="det">${d.name} ${(d.conf * 100).toFixed(0)}%</span>`).join('') ||
    (a.photo ? '<div class="empty">YOLO не нашёл известных объектов. Тепловая аномалия при этом остаётся.</div>' : '');
  const tt = (s) => s ? new Date(s * 1000).toLocaleTimeString() : '—';
  $('#mMeta').innerHTML = `координаты (${a.x.toFixed(2)}; ${a.y.toFixed(2)}) ±${a.err.toFixed(2)} м · ΔTmax +${a.dT_max.toFixed(1)}° · ` +
    `наблюдений ${a.observations} · первое ${tt(a.t_first)}, последнее ${tt(a.t_last)} · осмотр: ${a.drone || '—'}`;
}
$('#mClose').onclick = () => { $('#modal').hidden = true; modalId = null; };
$('#modal').onclick = (e) => { if (e.target.id === 'modal') { $('#modal').hidden = true; modalId = null; } };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { $('#modal').hidden = true; modalId = null; } });

// ---------- fault injection (simulation only) ----------
async function initFaults() {
  const r = await (await fetch('/api/sim/faults')).json();
  if (!r.enabled) return;
  $('#faultCard').hidden = false;
  $('#faultDrone').innerHTML = S.cfg.drones.map((d) => `<option value="${d.id}">${d.id}</option>`).join('');
  $('#faultKind').innerHTML = Object.entries(r.faults).filter(([k]) => k !== 'clear').map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
  const send = (fault) => fetch('/api/sim/fault', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ drone: $('#faultDrone').value, fault }) }).then(refresh);
  const refresh = async () => {
    const a = (await (await fetch('/api/sim/faults')).json()).active || {};
    $('#faultActive').textContent = Object.entries(a).filter(([, v]) => v.length).map(([k, v]) => `${k}: ${v.map((f) => r.faults[f] || f).join(', ')}`).join(' · ') || 'активных отказов нет';
  };
  $('#faultGo').onclick = () => send($('#faultKind').value);
  $('#faultClear').onclick = () => send('clear');
  setInterval(refresh, 2000); refresh();
}

// ---------- live stream ----------
let raf = 0;
const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; drawMap(); }); };
async function init() {
  S.cfg = await (await fetch('/api/config')).json();
  buildDrones(); setupView(); initFaults();
  const es = new EventSource('/api/events');
  es.addEventListener('state', (e) => {
    S.state = JSON.parse(e.data);
    updateHeader(); updateDrones(); updateTable(); redraw();
  });
  es.addEventListener('heat', (e) => { S.heat = JSON.parse(e.data); buildHeat(); redraw(); });
  es.addEventListener('trails', (e) => { S.trails = JSON.parse(e.data); });
  es.addEventListener('thermal', (e) => { S.live = JSON.parse(e.data); drawLive(); });
  es.addEventListener('log', (e) => addLog(JSON.parse(e.data)));
  es.onopen = () => { $('#log').innerHTML = ''; };
  es.onerror = () => { $('#phase').textContent = 'нет связи с сервером'; $('#phaseDot').className = 'dot alarm'; };
}
init();
