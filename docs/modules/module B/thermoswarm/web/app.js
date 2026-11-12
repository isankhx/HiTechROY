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
  photo_failed: 'нет фото', unreachable: 'недоступна', not_inspected: 'не осмотрена', outside: 'вне поля' };
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
  S.hits = [];
  // trees (RGB): green crowns, infected ones outlined red
  if (S.layers.trees !== false) {
    for (const e of st.trees || []) {
      const r = Math.max(8, ((e.crown_m || 0.5) / 2) * k);
      const inf = e.infected;
      c.fillStyle = inf ? 'rgba(255,77,109,0.18)' : 'rgba(46,160,80,0.28)';
      c.strokeStyle = inf ? '#ff4d6d' : '#3fbf6a'; c.lineWidth = inf ? 2.5 : 1.8;
      c.setLineDash(e.status === 'new' ? [3, 3] : []);
      c.beginPath();
      for (let i = 0; i <= 24; i++) {                      // spiky crown outline
        const ang = (i / 24) * 2 * Math.PI, rr = i % 2 ? r * 0.78 : r;
        const px = X(e.x) + rr * Math.cos(ang), py = Y(e.y) + rr * Math.sin(ang);
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      }
      c.fill(); c.stroke(); c.setLineDash([]);
      const label = e.id + (inf ? ' · заражена' : (e.status === 'new' ? '' : ' · здорова')) + (e.source === 'operator' ? ' · ручн.' : '');
      c.font = '600 11px system-ui, sans-serif'; c.textAlign = 'center'; c.textBaseline = 'top';
      c.lineWidth = 3; c.strokeStyle = '#0a0d12'; c.strokeText(label, X(e.x), Y(e.y) + r + 2);
      c.fillStyle = inf ? '#ffb3c1' : '#b7f0c8'; c.fillText(label, X(e.x), Y(e.y) + r + 2);
      S.hits.push({ x: X(e.x), y: Y(e.y), r: r + 4, id: e.id });
    }
  }
  // anomalies
  for (const a of st.anomalies) {
    const col = AN_COLOR[a.status] || '#6b7280', r = Math.max(7, a.err * k);
    c.fillStyle = col + '33'; c.strokeStyle = col; c.lineWidth = 2;
    c.beginPath(); c.arc(X(a.x), Y(a.y), r, 0, 7); c.fill(); c.stroke();
    c.beginPath(); c.arc(X(a.x), Y(a.y), 3, 0, 7); c.fillStyle = col; c.fill();
    const label = a.id + (a.label ? ' · ' + a.label : (a.status === 'inspected' ? ' · ?' : '')) + (a.source === 'operator' ? ' · ручн.' : '');
    c.font = '600 12px system-ui, sans-serif'; c.textAlign = 'left'; c.textBaseline = 'bottom';
    c.lineWidth = 3; c.strokeStyle = '#0a0d12'; c.strokeText(label, X(a.x) + r + 2, Y(a.y) - 2);
    c.fillStyle = '#fff'; c.fillText(label, X(a.x) + r + 2, Y(a.y) - 2);
    S.hits.push({ x: X(a.x), y: Y(a.y), r: r + 6, id: a.id });
  }
  // assigned inspection zones (orbit around the anomaly)
  if (S.layers.route) {
    for (const d of st.drones) {
      if (!d.zone) continue;
      const [zx, zy, zr] = d.zone;
      c.strokeStyle = d.color; c.fillStyle = d.color + '18'; c.lineWidth = 1.5; c.setLineDash([4, 3]);
      c.beginPath(); c.arc(X(zx), Y(zy), Math.max(zr * k, 8), 0, 7); c.fill(); c.stroke(); c.setLineDash([]);
      c.font = '600 11px system-ui, sans-serif'; c.textAlign = 'center'; c.textBaseline = 'top'; c.fillStyle = d.color;
      c.fillText('зона ' + d.id, X(zx), Y(zy) + zr * k + 3);
    }
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
  if ($('#manClick') && $('#manClick').checked) { manualCmd('goto', { x: +invX(px).toFixed(2), y: +invY(py).toFixed(2), z: +$('#manAlt').value }); return; }
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
    if (d.role === 'thermal') {
      cam.insertAdjacentHTML('beforeend', `<canvas class="ovl" id="ovl"></canvas>
        <div class="ovlbar"><label title="Тепло поверх RGB"><input type="checkbox" id="ovlOn"> тепло</label>
        <button class="btn sm ghost" id="ovlCfg" title="Подгонка наложения">⚙</button></div>
        <div class="ovlpanel" id="ovlPanel" hidden>
          <label>прозрачность <input type="range" data-k="alpha" min="0" max="1" step="0.05"></label>
          <label>нагрев от <input type="range" data-k="thr" min="-1" max="6" step="0.5"><span data-v="thr"></span></label>
          <label>масштаб <input type="range" data-k="scale" min="0.2" max="4" step="0.01"><span data-v="scale"></span></label>
          <label>сдвиг X <input type="range" data-k="dx" min="-1" max="1" step="0.005"></label>
          <label>сдвиг Y <input type="range" data-k="dy" min="-1" max="1" step="0.005"></label>
          <label>поворот <input type="range" data-k="rot" min="-180" max="180" step="0.5"><span data-v="rot"></span></label>
          <label><input type="checkbox" data-k="mirror"> зеркально</label>
          <button class="btn sm go" id="ovlCalib">Калибровка по точкам…</button>
          <button class="btn sm ghost" id="ovlReset">сброс</button>
        </div>`);
    }
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
    const lat = d.rtt != null ? d.rtt : d.link_age;
    const link = !d.connected ? '<b class="bad">нет</b>' : `<b class="${d.link_slow ? 'bad' : lat > 0.6 ? 'mid' : 'good'}" title="круговая задержка">${(lat * 1000).toFixed(0)} мс</b>`;
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

// ---------- thermal over RGB (thermal drone tile) ----------
const OVL_DEF = { on: false, alpha: 0.6, thr: 1.5, scale: 1.0, dx: 0, dy: 0, rot: 0, mirror: false };
let OVL = (() => { try { return Object.assign({}, OVL_DEF, JSON.parse(localStorage.getItem('ovl') || '{}')); } catch (e) { return Object.assign({}, OVL_DEF); } })();
function saveOvl() { try { localStorage.setItem('ovl', JSON.stringify(OVL)); } catch (e) { /* private mode */ } }
function initOverlay() {
  const on = $('#ovlOn'); if (!on) return;
  const panel = $('#ovlPanel');
  const sync = () => {
    on.checked = OVL.on;
    panel.querySelectorAll('[data-k]').forEach((el) => { if (el.type === 'checkbox') el.checked = !!OVL[el.dataset.k]; else el.value = OVL[el.dataset.k]; });
    panel.querySelector('[data-v="thr"]').textContent = (OVL.thr > 0 ? '+' : '') + OVL.thr + '°';
    panel.querySelector('[data-v="scale"]').textContent = '×' + Number(OVL.scale).toFixed(2);
    panel.querySelector('[data-v="rot"]').textContent = Number(OVL.rot).toFixed(1) + '°';
    $('#ovl').style.display = OVL.on ? 'block' : 'none';
  };
  on.onchange = () => { OVL.on = on.checked; saveOvl(); sync(); drawOverlay(); };
  $('#ovlCfg').onclick = () => { panel.hidden = !panel.hidden; };
  panel.querySelectorAll('[data-k]').forEach((el) => {
    el.oninput = () => { const k = el.dataset.k; OVL[k] = el.type === 'checkbox' ? el.checked : (k === 'rot' ? +el.value : parseFloat(el.value)); saveOvl(); sync(); drawOverlay(); };
  });
  $('#ovlReset').onclick = () => { OVL = Object.assign({}, OVL_DEF, { on: OVL.on }); saveOvl(); sync(); drawOverlay(); };
  $('#ovlCalib').onclick = () => openCalib(sync);
  OVL_SYNC = sync;
  sync();
}
let OVL_SYNC = () => {};
// thermal frame -> offscreen canvas (alpha fades in above thr; thr <= -1: everything opaque)
function thermalImage(L, thr) {
  const off = document.createElement('canvas'); off.width = L.w; off.height = L.h;
  const oc = off.getContext('2d'), img = oc.createImageData(L.w, L.h), raw = atob(L.data), span = L.hi - L.lo;
  for (let i = 0; i < L.w * L.h; i++) {
    const q = raw.charCodeAt(i), dT = L.lo + q / 255 * span;
    const a = thr <= -1 ? 1 : Math.max(0, Math.min(1, (dT - thr) / 3));
    img.data[i * 4] = LUT[q * 3]; img.data[i * 4 + 1] = LUT[q * 3 + 1]; img.data[i * 4 + 2] = LUT[q * 3 + 2]; img.data[i * 4 + 3] = Math.round(255 * a);
  }
  oc.putImageData(img, 0, 0);
  return off;
}
// same geometry everywhere: centre + (dx, dy)·size, rotation, mirror, width = scale·W
function paintThermal(g, W, H, off, P, alpha) {
  g.save(); g.globalAlpha = alpha; g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  g.translate(W / 2 + P.dx * W, H / 2 + P.dy * H); g.rotate(P.rot * Math.PI / 180); g.scale(P.mirror ? -1 : 1, 1);
  const tw = W * P.scale, th = tw * off.height / off.width;
  g.drawImage(off, -tw / 2, -th / 2, tw, th); g.restore();
}
function drawOverlay() {
  const c = document.getElementById('ovl'), L = S.live;
  if (!c || !OVL.on || !L) return;
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  paintThermal(g, W, H, thermalImage(L, OVL.thr), OVL, OVL.alpha);
}

// ---------- overlay calibration by anchor points ----------
// Pairs: the same physical point clicked on the thermal frame and on the RGB frame.
// Least-squares 2D similarity (complex numbers): q = a·p + t, a = s·e^{iθ}; mirrored variant p -> -conj(p).
const CAL = { pairs: [], pend: null, mode: 'points', rgb: null, L: null, P: null, drag: null };
const cx = (re, im) => ({ re, im });
const cmul = (a, b) => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
function fitSimilarity(pairs, aspT, aspR, mirror) {
  const P = pairs.map((p) => { let z = cx(p.t[0] - 0.5, (p.t[1] - 0.5) * aspT); if (mirror) z = cx(-z.re, z.im); return z; });
  const Q = pairs.map((p) => cx(p.r[0], p.r[1] * aspR));
  const n = P.length, mp = cx(P.reduce((s, z) => s + z.re, 0) / n, P.reduce((s, z) => s + z.im, 0) / n);
  const mq = cx(Q.reduce((s, z) => s + z.re, 0) / n, Q.reduce((s, z) => s + z.im, 0) / n);
  let num = cx(0, 0), den = 0;
  for (let i = 0; i < n; i++) {
    const pc = cx(P[i].re - mp.re, P[i].im - mp.im), qc = cx(Q[i].re - mq.re, Q[i].im - mq.im);
    num = cx(num.re + qc.re * pc.re + qc.im * pc.im, num.im + qc.im * pc.re - qc.re * pc.im);   // qc·conj(pc)
    den += pc.re * pc.re + pc.im * pc.im;
  }
  if (den < 1e-9) return null;
  const a = cx(num.re / den, num.im / den), ap = cmul(a, mp), t = cx(mq.re - ap.re, mq.im - ap.im);
  let err = 0;
  for (let i = 0; i < n; i++) { const z = cmul(a, P[i]); err += (z.re + t.re - Q[i].re) ** 2 + (z.im + t.im - Q[i].im) ** 2; }
  return { scale: Math.hypot(a.re, a.im), rot: Math.atan2(a.im, a.re) * 180 / Math.PI, dx: t.re - 0.5, dy: t.im / aspR - 0.5,
    mirror, rms: Math.sqrt(err / n) };
}
async function calibGrab() {
  const th = S.cfg.drones.find((d) => d.role === 'thermal');
  if (!S.live) { $('#calHint').textContent = 'нет теплового кадра'; return; }
  CAL.L = JSON.parse(JSON.stringify(S.live));
  const img = new Image();
  img.onload = () => { CAL.rgb = img; drawCalib(); };
  img.onerror = () => { $('#calHint').textContent = 'нет кадра RGB с ' + th.id; };
  img.src = `/api/frame/${th.id}.jpg?t=${Date.now()}`;
}
function openCalib(sync) {
  CAL.pairs = []; CAL.pend = null; CAL.P = Object.assign({}, OVL); CAL.mode = 'points';
  $('#calModal').hidden = false; calibGrab(); calibUI();
}
function coverDraw(g, img, W, H) {   // same as CSS object-fit: cover in the camera tile
  const s = Math.max(W / img.width, H / img.height), w = img.width * s, h = img.height * s;
  g.drawImage(img, (W - w) / 2, (H - h) / 2, w, h);
}
function drawCalib() {
  const ct = $('#calT'), cr = $('#calR');
  const gt = ct.getContext('2d'), gr = cr.getContext('2d');
  gt.fillStyle = '#07090c'; gt.fillRect(0, 0, ct.width, ct.height); gr.fillStyle = '#07090c'; gr.fillRect(0, 0, cr.width, cr.height);
  if (CAL.L) { const off = thermalImage(CAL.L, -1); gt.imageSmoothingEnabled = true; gt.drawImage(off, 0, 0, ct.width, ct.height); }
  if (CAL.rgb) coverDraw(gr, CAL.rgb, cr.width, cr.height);
  if (CAL.L && $('#calShow').checked) paintThermal(gr, cr.width, cr.height, thermalImage(CAL.L, -1), CAL.P, 0.45);   // whole frame: edges are visible
  const mark = (g, x, y, n, col) => { g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2; g.beginPath(); g.arc(x, y, 7, 0, 7); g.stroke();
    g.font = '700 13px system-ui'; g.fillText(n, x + 9, y - 7); };
  CAL.pairs.forEach((p, i) => { mark(gt, p.t[0] * ct.width, p.t[1] * ct.height, i + 1, '#4dd4c6'); mark(gr, p.r[0] * cr.width, p.r[1] * cr.height, i + 1, '#4dd4c6'); });
  if (CAL.pend) mark(gt, CAL.pend[0] * ct.width, CAL.pend[1] * ct.height, CAL.pairs.length + 1, '#ffd43b');
}
function calibUI() {
  $('#calPairs').innerHTML = CAL.pairs.map((p, i) => `<span class="det">${i + 1} <a href="#" data-i="${i}">✕</a></span>`).join('') || '<span class="muted">точек нет</span>';
  $('#calPairs').querySelectorAll('a').forEach((a) => (a.onclick = (e) => { e.preventDefault(); CAL.pairs.splice(+a.dataset.i, 1); calibUI(); drawCalib(); }));
  $('#calFit').disabled = CAL.pairs.length < 2;
  $('#calParams').textContent = `масштаб ×${CAL.P.scale.toFixed(3)} · поворот ${CAL.P.rot.toFixed(1)}° · сдвиг ${CAL.P.dx.toFixed(3)}, ${CAL.P.dy.toFixed(3)}${CAL.P.mirror ? ' · зеркально' : ''}`;
  $('#calHint').textContent = CAL.mode === 'drag' ? 'тяните тепловое изображение мышью · колесо — масштаб · Shift+колесо — поворот'
    : (CAL.pend ? `точка ${CAL.pairs.length + 1}: теперь кликните ту же точку на RGB справа` : `точка ${CAL.pairs.length + 1}: кликните тёплый предмет на тепловом кадре слева`);
  drawCalib();
}
function evtXY(c, e) { const r = c.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; }
function initCalib() {
  $('#calT').onclick = (e) => { if (CAL.mode !== 'points') return; CAL.pend = evtXY($('#calT'), e); calibUI(); };
  $('#calR').onmousedown = (e) => {
    const xy = evtXY($('#calR'), e);
    if (CAL.mode === 'points') { if (CAL.pend) { CAL.pairs.push({ t: CAL.pend, r: xy }); CAL.pend = null; calibUI(); } return; }
    CAL.drag = { xy, dx: CAL.P.dx, dy: CAL.P.dy };
  };
  window.addEventListener('mousemove', (e) => {
    if (!CAL.drag || $('#calModal').hidden) return;
    const xy = evtXY($('#calR'), e);
    CAL.P.dx = CAL.drag.dx + xy[0] - CAL.drag.xy[0]; CAL.P.dy = CAL.drag.dy + xy[1] - CAL.drag.xy[1]; calibUI();
  });
  window.addEventListener('mouseup', () => { CAL.drag = null; });
  $('#calR').addEventListener('wheel', (e) => {
    if (CAL.mode !== 'drag') return; e.preventDefault();
    if (e.shiftKey) CAL.P.rot += (e.deltaY > 0 ? 0.5 : -0.5); else CAL.P.scale *= (e.deltaY > 0 ? 0.98 : 1.02);
    calibUI();
  }, { passive: false });
  document.querySelectorAll('input[name="calMode"]').forEach((r) => (r.onchange = () => { CAL.mode = r.value; CAL.pend = null; calibUI(); }));
  $('#calShow').onchange = drawCalib;
  $('#calGrab').onclick = () => { CAL.pairs = []; CAL.pend = null; calibGrab(); calibUI(); };
  $('#calClear').onclick = () => { CAL.pairs = []; CAL.pend = null; calibUI(); };
  $('#calFit').onclick = () => {
    const aspT = CAL.L.h / CAL.L.w, aspR = $('#calR').height / $('#calR').width;
    const cands = CAL.pairs.length >= 3 ? [false, true] : [CAL.P.mirror];   // mirroring is decidable from 3 points
    const fits = cands.map((m) => fitSimilarity(CAL.pairs, aspT, aspR, m)).filter(Boolean).sort((a, b) => a.rms - b.rms);
    if (!fits.length) { $('#calRes').textContent = 'точки слишком близко друг к другу'; return; }
    const f = fits[0];
    Object.assign(CAL.P, { scale: f.scale, rot: f.rot, dx: f.dx, dy: f.dy, mirror: f.mirror });
    $('#calShow').checked = true;
    const px = f.rms * (CAL.rgb ? CAL.rgb.width : 640);
    $('#calRes').textContent = `подобрано по ${CAL.pairs.length} точкам · средняя ошибка ${px.toFixed(1)} пикс RGB (${(f.rms * 100).toFixed(1)}% ширины)` +
      (CAL.pairs.length < 3 ? ' · добавьте 3-ю точку, чтобы определить зеркальность' : '') + (px > 15 ? ' · ошибка велика: проверьте пары точек' : '');
    calibUI();
  };
  $('#calApply').onclick = () => {
    Object.assign(OVL, { scale: CAL.P.scale, rot: Math.round(CAL.P.rot * 10) / 10, dx: CAL.P.dx, dy: CAL.P.dy, mirror: CAL.P.mirror, on: true });
    saveOvl(); OVL_SYNC(); drawOverlay(); $('#calModal').hidden = true;
  };
  $('#calClose').onclick = () => { $('#calModal').hidden = true; };
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
  drawOverlay();
}

// ---------- header, table, log ----------
function updateHeader() {
  const st = S.state;
  const badge = $('#modeBadge');
  badge.textContent = st.mode === 'sim' ? 'СИМУЛЯЦИЯ' : 'РЕАЛЬНЫЕ ДРОНЫ';
  badge.className = 'badge ' + st.mode;
  $('#phase').textContent = st.phase_name + (st.paused ? ' · ПАУЗА' : '');
  const running = ['takeoff', 'survey', 'return', 'manual'].includes(st.phase);
  $('#phaseDot').className = 'dot ' + (st.phase === 'aborted' ? 'alarm' : st.paused ? 'pause' : running ? 'run' : '');
  $('#cov').textContent = st.coverage.toFixed(0);
  $('#frames').textContent = st.frames;
  $('#nAn').textContent = st.anomalies.length;
  $('#nIns').textContent = st.anomalies.filter((a) => a.status === 'inspected').length;
  $('#btnStart').disabled = running || st.phase === 'aborted';
  $('#btnPause').disabled = !running;
  $('#btnPause').textContent = st.paused ? '▶ Продолжить' : '⏸ Пауза';
  $('#btnHome').disabled = !['takeoff', 'survey', 'manual'].includes(st.phase);
  $('#btnReset').disabled = running;
  $('#detName').textContent = 'RGB: ' + st.detector + ' · ИК: ' + st.thermal_detector;
  $('#mtime').textContent = st.mission_s != null ? Math.round(st.mission_s) : '—';
  $('#sepInfo').textContent = `зазор ${(st.guard - 2 * st.R).toFixed(2)} м, центры ≥ ${st.D} м`;
}

let tableKey = '';
function updateTable() {
  const an = S.state.anomalies;
  const key = JSON.stringify(an.map((a) => [a.id, a.status, a.label, a.observations, a.drone, Math.round(a.x * 100), Math.round(a.y * 100), (a.photos || []).length, a.sources]));
  if (key === tableKey) return; tableKey = key;
  $('#anEmpty').style.display = an.length ? 'none' : 'block';
  const tr = S.state.trees || [];
  $('#nTr').textContent = tr.length;
  $('#nInf').textContent = tr.filter((e) => e.infected).length;
  $('#treeDet').textContent = 'детектор: ' + ({ dark: 'тёмные колючие кроны', yolo: 'YOLO', off: 'выключен' }[S.state.tree_detector] || S.state.tree_detector || '—');
  $('#trEmpty').style.display = tr.length ? 'none' : 'block';
  $('#trTable tbody').innerHTML = tr.map((e) => {
    const state = e.infected ? '<span class="st inf">заражена</span>' : (e.status === 'new' ? '<span class="st new">наблюдение</span>' : '<span class="st ok">здорова</span>');
    const ins = e.status === 'inspected' ? (e.label || 'снято') : (e.status === 'assigned' ? `летит ${e.drone}` : (STATUS[e.status] && e.status !== 'confirmed' && e.status !== 'infected' ? STATUS[e.status] : '—'));
    return `<tr data-id="${e.id}"><td><b>${e.id}</b>${e.source === 'operator' ? ' <span class="st manual" title="Отмечено оператором вручную">ручн.</span>' : ''}</td><td>${e.x.toFixed(2)}</td><td>${e.y.toFixed(2)}</td><td>${e.crown_m != null ? e.crown_m.toFixed(2) : '—'}</td>` +
      `<td>${e.observations}</td><td>${state}</td><td>${e.anomaly || '—'}</td><td>${ins}</td></tr>`;
  }).join('');
  document.querySelectorAll('#trTable tbody tr').forEach((r) => (r.onclick = () => openAnomaly(r.dataset.id)));
  $('#anTable tbody').innerHTML = an.map((a) => {
    const nph = (a.photos || []).length;
    const det = a.label ? `${a.label}${a.views_total ? ` (${a.label_views}/${a.views_total} ракурсов)` : ''}`
      : (a.status === 'inspected' ? 'не распознан' : (a.status === 'assigned' && nph ? `снято ${nph}…` : '—'));
    const src = a.sources ? Object.entries(a.sources).map(([k, v]) => (k === 'nn' ? 'NN' : 'порог') + ' ' + v).join(' · ') : '';
    return `<tr data-id="${a.id}"><td><b>${a.id}</b>${a.source === 'operator' ? ' <span class="st manual" title="Отмечено оператором вручную">ручн.</span>' : ''}</td><td>${a.x.toFixed(2)}</td><td>${a.y.toFixed(2)}</td><td>${a.z != null ? a.z.toFixed(2) : '—'}</td><td>${a.err.toFixed(2)}</td>` +
      `<td>+${a.dT_max.toFixed(1)}</td><td title="${src}">${a.observations}${a.sources && a.sources.nn ? ' <span class="nn">NN</span>' : ''}</td><td><span class="st ${a.status}">${STATUS[a.status] || a.status}</span></td>` +
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
  $('#mTitle').textContent = a.kind === 'tree'
    ? `Ёлка ${a.id} · ${a.infected ? 'заражена (' + a.anomaly + ')' : 'здорова'}`
    : `Аномалия ${a.id} · ${STATUS[a.status] || a.status}${a.tree ? ' · на ёлке ' + a.tree : ''}`;
  if (a.source === 'operator') $('#mTitle').textContent += ' · отмечено оператором вручную';
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
  const photos = (a.photos && a.photos.length) ? a.photos
    : (a.photo ? [{ file: a.photo, annotated: a.photo_annotated, view: 'над целью', detections: a.detections || [] }] : []);
  let cur = Math.max(0, photos.findIndex((p) => p.file === a.photo));
  const showPhoto = () => {
    const p = photos[cur];
    const name = p ? ($('#mAnn').checked && p.annotated ? p.annotated : p.file) : null;
    $('#mPhoto').style.display = name ? 'block' : 'none'; $('#mNoPhoto').style.display = name ? 'none' : 'block';
    if (name) $('#mPhoto').src = '/api/photo/' + encodeURIComponent(name);
    $('#mDets').innerHTML = (p && p.detections.length ? p.detections.map((d) => `<span class="det">${d.name} ${(d.conf * 100).toFixed(0)}%</span>`).join('')
      : (p ? '<div class="empty">На этом ракурсе YOLO не нашёл известных объектов. Тепловая аномалия при этом остаётся.</div>' : ''));
    $('#mViews').innerHTML = photos.map((q, i) => `<button class="btn sm ${i === cur ? '' : 'ghost'}" data-i="${i}">${q.view}${q.detections.length ? ' ✓' : ''}</button>`).join('');
    $('#mViews').querySelectorAll('button').forEach((b) => (b.onclick = () => { cur = +b.dataset.i; showPhoto(); }));
  };
  $('#mAnn').onchange = showPhoto; showPhoto();
  const tt = (s) => s ? new Date(s * 1000).toLocaleTimeString() : '—';
  $('#mMeta').innerHTML = `координаты (${a.x.toFixed(2)}; ${a.y.toFixed(2)}) ±${a.err.toFixed(2)} м, высота ${a.z != null ? a.z.toFixed(2) : '—'} м · ΔTmax +${a.dT_max.toFixed(1)}° · ` +
    `наблюдений ${a.observations} (${Object.entries(a.sources || {}).map(([k, v]) => (k === 'nn' ? 'нейросеть' : 'порог') + ' ' + v).join(', ') || '—'}` +
    `${a.nn_conf ? ', NN ' + Math.round(a.nn_conf * 100) + '%' : ''}) · первое ${tt(a.t_first)}, последнее ${tt(a.t_last)} · осмотр: ${a.drone || '—'}` +
    (a.label ? ` · итог: <b>${a.label}</b> на ${a.label_views} из ${a.views_total} ракурсов` : '');
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

// ---------- manual control from the map ----------
async function manualCmd(op, extra) {
  const drone = $('#manDrone').value, msg = $('#manMsg');
  const r = await fetch('/api/manual/' + op, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.assign({ drone }, extra || {})) });
  const j = await r.json().catch(() => ({}));
  msg.className = r.ok ? 'small' : 'small err';
  msg.textContent = r.ok ? ({ takeoff: 'взлёт…', goto: `летит в (${extra.x}; ${extra.y}), ${extra.z} м`, hold: 'висит', land: 'садится' })[op] : (j.detail || 'ошибка');
}
function initManual() {
  $('#manDrone').innerHTML = S.cfg.drones.map((d) => `<option value="${d.id}">${d.name || d.id}</option>`).join('');
  const alt = $('#manAlt'), def = (S.cfg.inspect && S.cfg.inspect.cruise_alt_m) || 2.4;
  alt.value = def; $('#manAltV').textContent = (+alt.value).toFixed(1) + ' м';
  alt.oninput = () => { $('#manAltV').textContent = (+alt.value).toFixed(1) + ' м'; };
  $('#manTakeoff').onclick = () => manualCmd('takeoff', { z: +alt.value });
  $('#manHold').onclick = () => manualCmd('hold');
  $('#manLand').onclick = () => manualCmd('land');
  $('#manClick').onchange = () => { cv.classList.toggle('manual-on', $('#manClick').checked); };
}

// ---------- flight parameters (between missions) ----------
const SEC_NAMES = { thermal: 'Тепловой облёт (T1)', inspect: 'Осмотр (T2/T3)', flight: 'Полёт', separation: 'Безопасность', battery: 'Безопасность' };
let SET = null;
async function openSettings() {
  const r = await (await fetch('/api/settings')).json();
  SET = r; $('#setModal').hidden = false;
  const list = $('#setList'); list.innerHTML = '';
  let sec = '';
  for (const p of r.params) {
    const name = SEC_NAMES[p.section] || p.section;
    if (name !== sec) { sec = name; list.insertAdjacentHTML('beforeend', `<div class="set-sec">${sec}</div>`); }
    const id = p.section + '.' + p.key;
    list.insertAdjacentHTML('beforeend', `<div class="set-row" data-id="${id}"><span>${p.label}</span>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}" ${r.editable ? '' : 'disabled'}>
      <span class="v">${p.value}${p.unit ? ' ' + p.unit : ''}</span></div>`);
  }
  list.querySelectorAll('.set-row').forEach((row) => {
    const p = r.params.find((q) => q.section + '.' + q.key === row.dataset.id), inp = row.querySelector('input');
    inp.oninput = () => { row.querySelector('.v').textContent = inp.value + (p.unit ? ' ' + p.unit : ''); row.classList.toggle('changed', +inp.value !== +p.value); };
  });
  $('#setNote').textContent = r.editable ? 'применяются к следующему старту' : 'идёт миссия — менять нельзя';
  $('#setSave').disabled = !r.editable;
  $('#setInfo').textContent = `облёт: ${r.survey.lanes} галса, полоса ${r.survey.swath_m} м, шаг ${r.survey.spacing_m} м · центры ≥ ${r.D} м`;
}
async function saveSettings() {
  const body = {};
  $('#setList').querySelectorAll('.set-row.changed').forEach((row) => { body[row.dataset.id] = +row.querySelector('input').value; });
  if (!Object.keys(body).length) { $('#setModal').hidden = true; return; }
  const r = await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { $('#setNote').textContent = j.detail || 'ошибка'; return; }
  await openSettings();
  $('#setNote').textContent = 'сохранено в config.json';
}
$('#btnSettings').onclick = openSettings;
$('#setClose').onclick = () => { $('#setModal').hidden = true; };
$('#setUndo').onclick = openSettings;
$('#setSave').onclick = saveSettings;
$('#setModal').onclick = (e) => { if (e.target.id === 'setModal') $('#setModal').hidden = true; };

// ---------- live stream ----------
let raf = 0;
const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; drawMap(); }); };
async function init() {
  S.cfg = await (await fetch('/api/config')).json();
  buildDrones(); setupView(); initFaults(); initOverlay(); initCalib(); initManual();
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
