import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from ultralytics import YOLO

WEIGHTS = "runs/symbols/weights/best.pt"
TARGET_W = 5800
TILE = 1024
OVERLAP = 256
RUN_CONF = 0.25
IGNORE = [(0.00, 0.80, 0.33, 1.00),
          (0.57, 0.82, 1.00, 1.00)]

model = YOLO(WEIGHTS)
CLASS_NAMES = [model.names[i] for i in range(len(model.names))]
STORE = {}

app = FastAPI()


def tile_starts(n, tile, step):
    starts = list(range(0, max(n - tile, 0) + 1, step))
    if starts[-1] + tile < n:
        starts.append(n - tile)
    return starts


def box_iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def contained(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0


def merge(dets, iou_thr=0.5):
    keep = []
    for d in sorted(dets, key=lambda r: -r[4]):
        if all(box_iou(d, k) < iou_thr and contained(d, k) < 0.8 for k in keep):
            keep.append(d)
    return keep


def detect(plan):
    H, W = plan.shape[:2]
    ys = tile_starts(H, TILE, TILE - OVERLAP)
    xs = tile_starts(W, TILE, TILE - OVERLAP)
    dets, m = [], 3
    for y0 in ys:
        for x0 in xs:
            tile = plan[y0:y0 + TILE, x0:x0 + TILE]
            res = model.predict(tile, conf=RUN_CONF, imgsz=TILE, iou=0.5,
                                agnostic_nms=True, verbose=False)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for (bx0, by0, bx1, by1), s, c in zip(boxes, scores, classes):
                cut = ((bx0 < m and x0 > 0) or (by0 < m and y0 > 0)
                       or (bx1 > TILE - m and x0 + TILE < W)
                       or (by1 > TILE - m and y0 + TILE < H))
                if not cut:
                    dets.append([float(bx0 + x0), float(by0 + y0),
                                 float(bx1 + x0), float(by1 + y0), float(s), int(c)])
    return merge(dets), len(ys) * len(xs), len(dets)


def in_ignored(d, W, H):
    cx, cy = (d[0] + d[2]) / 2 / W, (d[1] + d[3]) / 2 / H
    return any(a <= cx <= c and b <= cy <= e for a, b, c, e in IGNORE)


@app.post("/detect")
async def run(file: UploadFile = File(...)):
    raw = await file.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "that file could not be read as an image"}, status_code=400)

    f = TARGET_W / img.shape[1]
    plan = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    H, W = plan.shape[:2]

    dets, n_tiles, n_raw = detect(plan)
    dets = [d for d in dets if not in_ignored(d, W, H)]

    jid = uuid.uuid4().hex[:8]
    STORE[jid] = cv2.imencode(".jpg", plan, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    for old in list(STORE)[:-4]:
        STORE.pop(old)

    out = [{"id": i, "x0": d[0], "y0": d[1], "x1": d[2], "y1": d[3],
            "conf": d[4], "cls": d[5], "manual": False} for i, d in enumerate(dets)]

    return {"id": jid, "w": W, "h": H, "classes": CLASS_NAMES,
            "tiles": n_tiles, "raw": n_raw, "detections": out}


@app.get("/image/{jid}")
def image(jid: str):
    return Response(STORE.get(jid, b""), media_type="image/jpeg")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Plan symbol detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --ink:#101b27; --ink2:#55636f; --ink3:#8793a0;
  --sheet:#ffffff; --table:#d7dce2; --panel:#f6f7f9;
  --rule:#dde2e8; --rule2:#c6cdd5;
  --blue:#1554a6; --blue-soft:#e8eff8;
  --ok:#1a7a4c; --check:#a86f08;
  --font:"IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
}
* { box-sizing:border-box; }
html, body { height:100%; }
body { margin:0; font-family:var(--font); font-size:13px; color:var(--ink);
       background:var(--table); display:flex; flex-direction:column; }
button, select, input { font-family:inherit; font-size:inherit; color:inherit; }
:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }

/* header */
.top { display:flex; align-items:center; gap:14px; padding:0 16px; height:52px;
       background:var(--ink); color:#fff; border-bottom:3px solid var(--blue); flex:none; }
.mark { display:flex; align-items:baseline; gap:9px; }
.mark b { font-weight:600; font-size:15px; letter-spacing:.2px; }
.mark span { color:#93a2b2; font-size:12px; }
.grow { flex:1; }
.file { position:relative; overflow:hidden; display:inline-flex; }
.file input { position:absolute; inset:0; opacity:0; cursor:pointer; width:100%; }
.btn { background:#fff; border:1px solid var(--rule2); border-radius:3px; padding:6px 11px;
       cursor:pointer; line-height:1.2; }
.btn:hover { background:#eef1f5; }
.btn[disabled] { opacity:.45; cursor:default; }
.top .btn { background:#243344; border-color:#3a4c60; color:#fff; }
.top .btn:hover { background:#2f4257; }
.top .btn.primary { background:var(--blue); border-color:var(--blue); }
.top .btn.primary:hover { background:#1a64c4; }
.fname { color:#93a2b2; font-size:12px; max-width:200px; overflow:hidden;
         text-overflow:ellipsis; white-space:nowrap; }
.state { font-size:12px; color:#b7c3d0; display:flex; align-items:center; gap:7px; }
.dot { width:7px; height:7px; border-radius:50%; background:#5a6b7c; }
.dot.busy { background:#f0b429; animation:pulse 1s ease-in-out infinite; }
.dot.done { background:#45c08a; }
@keyframes pulse { 50% { opacity:.25; } }

/* frame */
.body { flex:1; display:flex; min-height:0; }
.side { background:var(--sheet); display:flex; flex-direction:column; flex:none; }
.side.left { width:252px; border-right:1px solid var(--rule2); }
.side.right { width:336px; border-left:1px solid var(--rule2); }
.blk { padding:14px 16px; border-bottom:1px solid var(--rule); }
.blk:last-child { border-bottom:0; }
.cap { font-size:12px; font-weight:600; letter-spacing:.15px; margin:0 0 10px; }
.sub { color:var(--ink3); font-size:11.5px; font-weight:400; }

/* legend rows */
.lg { display:flex; align-items:center; gap:9px; width:100%; padding:5px 6px; margin:0 -6px;
      border:0; background:none; text-align:left; cursor:pointer; border-radius:3px; }
.lg:hover { background:var(--panel); }
.lg[aria-pressed="false"] { opacity:.38; }
.key { width:15px; height:15px; border-radius:2px; flex:none; border:1.5px solid; }
.lg .nm { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.num { font-variant-numeric:tabular-nums; color:var(--ink2); min-width:22px; text-align:right; }
.tag { font-size:10px; padding:1px 5px; border-radius:2px; border:1px solid var(--rule2);
       color:var(--ink2); }
.pair { display:flex; gap:7px; margin-top:10px; }
.pair .btn { flex:1; padding:5px 0; }

/* confidence */
.slider { width:100%; accent-color:var(--blue); margin:4px 0 2px; }
.ends { display:flex; justify-content:space-between; color:var(--ink3); font-size:11.5px; }
.bands { display:flex; gap:12px; margin-top:11px; color:var(--ink2); font-size:11.5px; }
.bands i { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:5px; }

/* stage */
.main { flex:1; display:flex; flex-direction:column; min-width:0; }
.bar { display:flex; align-items:center; gap:10px; padding:8px 12px; background:var(--sheet);
       border-bottom:1px solid var(--rule2); flex:none; }
.seg { display:inline-flex; border:1px solid var(--rule2); border-radius:3px; overflow:hidden; }
.seg button { border:0; background:#fff; padding:6px 12px; cursor:pointer; }
.seg button + button { border-left:1px solid var(--rule2); }
.seg button[aria-pressed="true"] { background:var(--ink); color:#fff; }
.zoom { display:inline-flex; align-items:center; gap:2px; }
.zoom .btn { padding:5px 9px; border-radius:0; }
.zoom .btn:first-child { border-radius:3px 0 0 3px; }
.zoom .btn:nth-child(3) { border-left:0; border-right:0; }
.zoom .btn:last-child { border-radius:0 3px 3px 0; }
.pct { font-variant-numeric:tabular-nums; color:var(--ink2); min-width:46px; text-align:center; }
.view { flex:1; overflow:auto; position:relative; background:var(--table);
        background-image:linear-gradient(rgba(16,27,39,.045) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(16,27,39,.045) 1px, transparent 1px);
        background-size:28px 28px; }
.sheet { position:relative; margin:18px; box-shadow:0 1px 2px rgba(16,27,39,.18),
         0 10px 26px rgba(16,27,39,.16); background:#fff; }
.sheet img { display:block; width:100%; }
.overlay { position:absolute; inset:0; }
.overlay.in { animation:rise .28s ease-out; }
@keyframes rise { from { opacity:0; } to { opacity:1; } }
.bx { position:absolute; border:2px solid; cursor:pointer; }
.bx.on { outline:2.5px solid var(--ink); outline-offset:3px; box-shadow:0 0 0 3px #fff; z-index:4; }
.bx .lbl { position:absolute; left:-2px; top:-18px; font-size:11px; color:#fff; padding:1px 5px;
           border-radius:2px; white-space:nowrap; font-weight:500; }
.band { position:absolute; border:1.5px dashed var(--ink); background:rgba(21,84,166,.1); display:none; }
.empty { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
         justify-content:center; gap:6px; color:var(--ink2); }
.empty b { font-weight:600; font-size:14px; }

/* schedule */
.sched { flex:1; overflow-y:auto; }
.grp { position:sticky; top:0; z-index:2; background:var(--panel); padding:7px 16px;
       border-top:1px solid var(--rule2); border-bottom:1px solid var(--rule);
       display:flex; align-items:center; gap:9px; font-weight:600; }
.grp .num { font-weight:400; }
.rw { display:flex; align-items:center; gap:10px; width:100%; padding:7px 16px; border:0;
      background:none; cursor:pointer; text-align:left; border-bottom:1px solid #f0f2f5; }
.rw:hover { background:var(--panel); }
.rw.on { background:var(--blue-soft); }
.ix { color:var(--ink3); font-variant-numeric:tabular-nums; width:20px; }
.pos { flex:1; color:var(--ink2); font-variant-numeric:tabular-nums; }
.cf { font-variant-numeric:tabular-nums; display:inline-flex; align-items:center; gap:6px; }
.cf i { width:7px; height:7px; border-radius:50%; }
.ed { padding:10px 16px 14px; background:var(--panel); border-bottom:1px solid var(--rule2); }
.ed .meta { color:var(--ink2); margin-bottom:9px; }
.ed select, .ed input[type=text] { width:100%; padding:6px 7px; border:1px solid var(--rule2);
       border-radius:3px; background:#fff; }
.ed .acts { display:flex; gap:7px; margin-top:9px; }
.del { border-color:#dcb3b3; color:#9d2b2b; }
.del:hover { background:#fbeeee; }
.newtype { display:none; margin-top:8px; }
.newtype.show { display:block; }
.note { padding:14px 16px; color:var(--ink2); line-height:1.55; }
.foot { display:flex; align-items:center; gap:9px; padding:9px 16px;
        border-top:1px solid var(--rule2); background:#fff; }

/* title block */
.title-block { display:flex; flex:none; background:#fff; border-top:1px solid var(--rule2); }
.fld { padding:7px 16px; border-right:1px solid var(--rule); min-width:112px; }
.fld:last-child { border-right:0; }
.fld span { display:block; color:var(--ink3); font-size:11px; }
.fld b { font-weight:500; font-variant-numeric:tabular-nums; }

@media (prefers-reduced-motion: reduce) { * { animation:none !important; } }
</style>
</head>
<body>

<div class="top">
  <div class="mark"><b>Plan symbol detector</b><span>electrical takeoff</span></div>
  <label class="file btn">Choose plan<input type="file" id="file" accept="image/*"></label>
  <span class="fname" id="fname">no file chosen</span>
  <button class="btn primary" id="run">Run detection</button>
  <span class="grow"></span>
  <span class="state"><i class="dot" id="dot"></i><span id="state">waiting for a plan</span></span>
</div>

<div class="body">

  <aside class="side left">
    <div class="blk">
      <h2 class="cap">Confidence <span class="sub" id="thrval">0.50 and above</span></h2>
      <input type="range" class="slider" id="thr" min="25" max="95" value="50">
      <div class="ends"><span>0.25</span><span>0.95</span></div>
      <div class="bands">
        <span><i style="background:var(--ok)"></i>clear</span>
        <span><i style="background:var(--check)"></i>worth a check</span>
      </div>
    </div>
    <div class="blk">
      <h2 class="cap">Legend</h2>
      <div id="legend"></div>
      <div class="pair">
        <button class="btn" id="allon">Show all</button>
        <button class="btn" id="alloff">Hide all</button>
      </div>
    </div>
    <div class="blk">
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="labels" style="accent-color:var(--blue)"> Name every box
      </label>
    </div>
  </aside>

  <main class="main">
    <div class="bar">
      <div class="seg">
        <button id="m-sel" aria-pressed="true">Select</button>
        <button id="m-add" aria-pressed="false">Draw a box</button>
      </div>
      <span class="grow"></span>
      <div class="zoom">
        <button class="btn" id="zout">&minus;</button>
        <span class="pct" id="pct">100%</span>
        <button class="btn" id="zin">+</button>
        <button class="btn" id="zfit">Fit</button>
      </div>
    </div>
    <div class="view" id="view">
      <div class="empty" id="empty"><b>No plan loaded</b><span>Choose a plan image, then run detection.</span></div>
      <div class="sheet" id="sheet" style="display:none">
        <img id="plan" alt="construction plan">
        <div class="overlay" id="overlay"></div>
        <div class="band" id="band"></div>
      </div>
    </div>
  </main>

  <aside class="side right">
    <div class="blk" style="padding-bottom:12px">
      <h2 class="cap" style="margin:0">Schedule <span class="sub" id="shownum"></span></h2>
    </div>
    <div class="sched" id="sched">
      <div class="note">Run detection to build the schedule.</div>
    </div>
    <div class="foot">
      <span class="grow" style="color:var(--ink2)" id="hint">Pick a row to edit it</span>
      <button class="btn" id="export">Export JSON</button>
    </div>
  </aside>

</div>

<div class="title-block">
  <div class="fld"><span>Sheet</span><b id="f-size">&ndash;</b></div>
  <div class="fld"><span>Tiles</span><b id="f-tiles">&ndash;</b></div>
  <div class="fld"><span>Raw detections</span><b id="f-raw">&ndash;</b></div>
  <div class="fld"><span>Objects shown</span><b id="f-shown">&ndash;</b></div>
  <div class="fld"><span>Threshold</span><b id="f-thr">0.50</b></div>
  <div class="fld"><span>Manual edits</span><b id="f-edit">0</b></div>
  <div class="fld grow"><span>Model</span><b>YOLOv8s, synthetic training data</b></div>
</div>

<script>
const BASE = [[198,32,58],[22,122,76],[21,84,166],[214,110,16],
              [124,46,150],[18,146,148],[186,38,150],[112,104,24]];
const hue = i => (i * 47 + 205) % 360;
const col = c => c < BASE.length ? `rgb(${BASE[c].join(",")})` : `hsl(${hue(c)},58%,40%)`;
const colA = (c, a) => c < BASE.length ? `rgba(${BASE[c].join(",")},${a})`
                                       : `hsl(${hue(c)},58%,40%,${a})`;

const S = { w:0, h:0, classes:[], custom:new Set(), dets:[], nextId:0, sel:null,
            zoom:1, thr:0.5, vis:new Set(), labels:false, mode:"select",
            tiles:0, raw:0, edits:0 };

const $ = id => document.getElementById(id);
const view = $("view"), sheet = $("sheet"), overlay = $("overlay"), band = $("band");

const shownDets = () => S.dets.filter(d => d.conf >= S.thr && S.vis.has(d.cls));
const confCol = d => d.manual ? "var(--blue)" : (d.conf >= 0.7 ? "var(--ok)" : "var(--check)");

function setZoom(z, fx, fy, cx, cy) {
  S.zoom = Math.min(4, Math.max(0.05, z));
  sheet.style.width = (S.w * S.zoom) + "px";
  $("pct").textContent = Math.round(S.zoom * 100) + "%";
  if (fx !== undefined) {
    view.scrollLeft = fx * sheet.offsetWidth - cx;
    view.scrollTop = fy * sheet.offsetHeight - cy;
  }
}
const fit = () => { if (S.w) setZoom((view.clientWidth - 56) / S.w); };

function centreOn(d) {
  const r = overlay.getBoundingClientRect(), v = view.getBoundingClientRect();
  view.scrollLeft += (r.left + (d.x0 + d.x1) / 2 / S.w * r.width) - v.left - v.width / 2;
  view.scrollTop += (r.top + (d.y0 + d.y1) / 2 / S.h * r.height) - v.top - v.height / 2;
}

function drawBoxes() {
  overlay.innerHTML = "";
  for (const d of shownDets()) {
    const on = S.sel === d.id;
    const el = document.createElement("div");
    el.className = "bx" + (on ? " on" : "");
    el.style.cssText = `left:${d.x0/S.w*100}%;top:${d.y0/S.h*100}%;` +
                       `width:${(d.x1-d.x0)/S.w*100}%;height:${(d.y1-d.y0)/S.h*100}%;` +
                       `border-color:${col(d.cls)};background:${colA(d.cls, on ? .5 : .26)}`;
    if (S.labels || on) {
      const t = document.createElement("div");
      t.className = "lbl";
      t.style.background = col(d.cls);
      t.textContent = S.classes[d.cls] + (d.manual ? "" : " " + d.conf.toFixed(2));
      el.appendChild(t);
    }
    el.onclick = ev => {
      ev.stopPropagation();
      S.sel = on ? null : d.id;
      paint();
      focusRow();
    };
    overlay.appendChild(el);
  }
}

function drawLegend() {
  const shown = shownDets();
  const box = $("legend");
  box.innerHTML = "";
  S.classes.forEach((name, i) => {
    const on = S.vis.has(i);
    const b = document.createElement("button");
    b.className = "lg";
    b.setAttribute("aria-pressed", on);
    b.innerHTML = `<span class="key" style="border-color:${col(i)};background:${colA(i, .3)}"></span>` +
                  `<span class="nm">${name}</span>` +
                  (S.custom.has(i) ? `<span class="tag">added</span>` : ``) +
                  `<span class="num">${shown.filter(d => d.cls === i).length}</span>`;
    b.onclick = () => { on ? S.vis.delete(i) : S.vis.add(i); paint(); };
    box.appendChild(b);
  });
  $("shownum").textContent = S.dets.length ? shown.length + " of " + S.dets.length : "";
  $("f-shown").textContent = S.dets.length ? shown.length : "\\u2013";
  $("f-edit").textContent = S.edits;
}

function editor(d) {
  const ed = document.createElement("div");
  ed.className = "ed";
  ed.innerHTML =
    `<div class="meta">centre ${Math.round((d.x0+d.x1)/2)}, ${Math.round((d.y0+d.y1)/2)} px` +
    ` &nbsp;&middot;&nbsp; ${Math.round(d.x1-d.x0)} &times; ${Math.round(d.y1-d.y0)} px` +
    ` &nbsp;&middot;&nbsp; ${d.manual ? "added by hand" : "found by the model"}</div>` +
    `<select class="cls">` +
      S.classes.map((n, i) => `<option value="${i}"${i === d.cls ? " selected" : ""}>${n}</option>`).join("") +
      `<option value="new">Something else, name it&hellip;</option></select>` +
    `<div class="newtype"><input type="text" class="tname" placeholder="for example: exhaust fan" maxlength="36">` +
    `<div class="acts"><button class="btn create">Add this type</button>` +
    `<button class="btn cancel">Cancel</button></div></div>` +
    `<div class="acts"><button class="btn del">Delete this object</button></div>`;

  const sel = ed.querySelector(".cls"), nt = ed.querySelector(".newtype");
  const field = ed.querySelector(".tname");

  sel.onchange = () => {
    if (sel.value === "new") { nt.classList.add("show"); field.focus(); return; }
    d.cls = +sel.value;
    S.edits += 1;
    paint();
    focusRow("center");
  };
  ed.querySelector(".cancel").onclick = () => { nt.classList.remove("show"); sel.value = d.cls; };
  ed.querySelector(".create").onclick = () => {
    const name = field.value.trim();
    if (!name) { field.focus(); return; }
    let i = S.classes.findIndex(n => n.toLowerCase() === name.toLowerCase());
    if (i < 0) { i = S.classes.push(name) - 1; S.custom.add(i); }
    S.vis.add(i);
    d.cls = i;
    S.edits += 1;
    paint();
    focusRow("center");
  };
  ed.querySelector(".del").onclick = () => {
    S.dets = S.dets.filter(x => x.id !== d.id);
    S.sel = null;
    S.edits += 1;
    paint();
  };
  field.onkeydown = e => { if (e.key === "Enter") ed.querySelector(".create").click(); };
  return ed;
}

function drawSchedule() {
  const list = $("sched");
  list.innerHTML = "";
  const shown = shownDets().sort((a, b) => a.cls - b.cls || b.conf - a.conf);

  if (!shown.length) {
    list.innerHTML = `<div class="note">${S.dets.length
      ? "Nothing matches the current filters. Lower the threshold or show more types."
      : "Run detection to build the schedule."}</div>`;
    $("hint").textContent = "Pick a row to edit it";
    return;
  }

  let last = null, n = 0;
  for (const d of shown) {
    if (d.cls !== last) {
      last = d.cls; n = 0;
      const g = document.createElement("div");
      g.className = "grp";
      g.innerHTML = `<span class="key" style="border-color:${col(d.cls)};background:${colA(d.cls,.3)}"></span>` +
                    `<span class="nm">${S.classes[d.cls]}</span>` +
                    `<span class="num">${shown.filter(x => x.cls === d.cls).length}</span>`;
      list.appendChild(g);
    }
    n += 1;
    const on = S.sel === d.id;
    const r = document.createElement("button");
    r.className = "rw" + (on ? " on" : "");
    r.innerHTML = `<span class="ix">${n}</span>` +
                  `<span class="pos">${Math.round((d.x0+d.x1)/2)}, ${Math.round((d.y0+d.y1)/2)}</span>` +
                  `<span class="cf" style="color:${confCol(d)}"><i style="background:${confCol(d)}"></i>` +
                  `${d.manual ? "added" : d.conf.toFixed(2)}</span>`;
    r.onclick = () => {
      S.sel = on ? null : d.id;
      paint();
      if (!on) centreOn(d);
    };
    list.appendChild(r);
    if (on) list.appendChild(editor(d));
  }
  $("hint").textContent = S.sel === null ? "Pick a row to edit it" : "Pick it again to close";
}

function paint() { drawBoxes(); drawLegend(); drawSchedule(); }

function focusRow(how) {
  const r = document.querySelector(".rw.on");
  if (r) r.scrollIntoView({ block: how || "nearest" });
}

$("thr").oninput = e => {
  S.thr = e.target.value / 100;
  $("thrval").textContent = S.thr.toFixed(2) + " and above";
  $("f-thr").textContent = S.thr.toFixed(2);
  paint();
};
$("labels").onchange = e => { S.labels = e.target.checked; paint(); };
$("allon").onclick = () => { S.classes.forEach((_, i) => S.vis.add(i)); paint(); };
$("alloff").onclick = () => { S.vis.clear(); paint(); };
$("zin").onclick = () => setZoom(S.zoom * 1.3);
$("zout").onclick = () => setZoom(S.zoom / 1.3);
$("zfit").onclick = fit;

function setMode(m) {
  S.mode = m;
  $("m-sel").setAttribute("aria-pressed", m === "select");
  $("m-add").setAttribute("aria-pressed", m === "add");
  overlay.style.cursor = m === "add" ? "crosshair" : "grab";
}
$("m-sel").onclick = () => setMode("select");
$("m-add").onclick = () => setMode("add");

view.onwheel = e => {
  if (!S.w) return;
  e.preventDefault();
  const r = sheet.getBoundingClientRect(), v = view.getBoundingClientRect();
  setZoom(S.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15),
          (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height,
          e.clientX - v.left, e.clientY - v.top);
};

let drag = null;
sheet.onmousedown = e => {
  if (!S.w) return;
  const r = overlay.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  if (S.mode === "add") {
    drag = { kind:"add", fx, fy };
    band.style.display = "block";
    band.style.left = fx * 100 + "%";
    band.style.top = fy * 100 + "%";
    band.style.width = band.style.height = "0";
  } else {
    drag = { kind:"pan", x:e.clientX, y:e.clientY, sl:view.scrollLeft, st:view.scrollTop };
  }
  e.preventDefault();
};

window.onmousemove = e => {
  if (!drag) return;
  if (drag.kind === "pan") {
    view.scrollLeft = drag.sl - (e.clientX - drag.x);
    view.scrollTop = drag.st - (e.clientY - drag.y);
  } else {
    const r = overlay.getBoundingClientRect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    band.style.left = Math.min(fx, drag.fx) * 100 + "%";
    band.style.top = Math.min(fy, drag.fy) * 100 + "%";
    band.style.width = Math.abs(fx - drag.fx) * 100 + "%";
    band.style.height = Math.abs(fy - drag.fy) * 100 + "%";
    drag.to = [fx, fy];
  }
};

window.onmouseup = () => {
  const d = drag;
  drag = null;
  band.style.display = "none";
  if (!d || d.kind !== "add" || !d.to) return;

  setMode("select");
  const x0 = Math.min(d.to[0], d.fx) * S.w, x1 = Math.max(d.to[0], d.fx) * S.w;
  const y0 = Math.min(d.to[1], d.fy) * S.h, y1 = Math.max(d.to[1], d.fy) * S.h;
  if (x1 - x0 < 4 || y1 - y0 < 4) return;

  const cls = S.vis.size ? Math.min(...S.vis) : 0;
  S.vis.add(cls);
  S.dets.push({ id:S.nextId, x0, y0, x1, y1, conf:1, cls, manual:true });
  S.sel = S.nextId;
  S.nextId += 1;
  S.edits += 1;
  paint();
  focusRow("center");
};

window.addEventListener("keydown", e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
  if (e.key === "Escape" && S.sel !== null) { S.sel = null; paint(); }
  if ((e.key === "Delete" || e.key === "Backspace") && S.sel !== null) {
    e.preventDefault();
    S.dets = S.dets.filter(x => x.id !== S.sel);
    S.sel = null;
    S.edits += 1;
    paint();
  }
});

$("export").onclick = () => {
  const shown = shownDets();
  const counts = {};
  S.classes.forEach((n, i) => {
    const c = shown.filter(d => d.cls === i).length;
    if (c || !S.custom.has(i)) counts[n] = c;
  });
  const objects = shown.map(d => ({
    type: S.classes[d.cls],
    source: d.manual ? "added by hand" : "model",
    confidence: d.manual ? null : +d.conf.toFixed(3),
    x0: Math.round(d.x0), y0: Math.round(d.y0), x1: Math.round(d.x1), y1: Math.round(d.y1)
  }));
  const blob = new Blob([JSON.stringify({
    sheet: { width: S.w, height: S.h, tiles: S.tiles },
    threshold: S.thr,
    types_added_by_hand: [...S.custom].map(i => S.classes[i]),
    counts, objects
  }, null, 2)], { type:"application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "takeoff.json";
  a.click();
};

$("file").onchange = e => {
  const f = e.target.files[0];
  $("fname").textContent = f ? f.name : "no file chosen";
};

$("run").onclick = async () => {
  const f = $("file").files[0];
  if (!f) { $("state").textContent = "choose a plan image first"; return; }
  $("dot").className = "dot busy";
  $("state").textContent = "detecting across tiles";
  $("run").disabled = true;

  const fd = new FormData();
  fd.append("file", f);
  let j;
  try {
    const r = await fetch("/detect", { method:"POST", body:fd });
    if (!r.ok) throw new Error();
    j = await r.json();
  } catch (err) {
    $("dot").className = "dot";
    $("state").textContent = "detection failed, check the server window";
    $("run").disabled = false;
    return;
  }

  Object.assign(S, { w:j.w, h:j.h, classes:j.classes.slice(), tiles:j.tiles, raw:j.raw,
                     dets:j.detections, nextId:j.detections.length, sel:null, edits:0 });
  S.custom = new Set();
  S.vis = new Set(S.classes.map((_, i) => i));

  $("f-size").textContent = j.w + " \\u00d7 " + j.h + " px";
  $("f-tiles").textContent = j.tiles;
  $("f-raw").textContent = j.raw;

  const img = $("plan");
  img.src = "/image/" + j.id;
  img.onload = () => {
    $("empty").style.display = "none";
    sheet.style.display = "";
    fit();
    paint();
    overlay.classList.remove("in");
    void overlay.offsetWidth;
    overlay.classList.add("in");
  };
  $("dot").className = "dot done";
  $("state").textContent = j.detections.length + " objects found";
  $("run").disabled = false;
  setMode("select");
};
</script>
</body>
</html>
"""


# @app.get("/", response_class=HTMLResponse)
# def index():
#     return PAGE
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(PAGE, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
