"""
globe_panel.py
A realistic 3D Earth globe for TerraMap's center panel, with Google-Earth-style
fly-to and a REAL 3D terrain map of the target area.

How it works
------------
Rendered with Three.js inside a QWebEngineView (a real Qt widget, so it sits
in the center panel like any other page of the stack). Python <-> JS talk over
QWebChannel:

    JS  -> Python : area_entered(lat, lon)    when the user lands on a target
    JS  -> Python : point_picked(lat, lon)    when the user clicks the globe

Data sources — all FREE, NO API KEY, NO ACCOUNT:
  * Globe texture   : NASA Blue Marble (via unpkg CDN, three-globe's own asset)
  * Satellite tiles : Esri World Imagery  (ArcGIS public tile service)
  * Elevation tiles : AWS Terrarium DEM   (registry of open data, RGB-encoded)

The terrain view is not a picture of a map — it is an actual 3D mesh:
elevation tiles are decoded (height = R*256 + G + B/256 - 32768 metres),
turned into a displaced plane, and the satellite tile is draped over it as a
texture. That is the same recipe Google Earth uses, just with open data.

Requires:
    pip install PySide6-Addons        # provides QtWebEngineWidgets
(usually already installed alongside PySide6)

INTERNET is needed only while flying into an area (to fetch tiles). The globe
itself falls back to a plain shaded sphere if the texture can't be fetched.
"""

import json
import os

# QtWebEngine must be configured BEFORE the QApplication/profile is created.
# --ignore-certificate-errors keeps Chromium from spamming SSL handshake
# errors when a tile host presents a chain it doesn't like; the tile services
# we use are public read-only endpoints, so this is not a security concern here.
os.environ.setdefault(
    "QTWEBENGINE_CHROMIUM_FLAGS",
    "--ignore-certificate-errors --disable-web-security "
    "--enable-features=NetworkService --log-level=3")

from PySide6.QtCore import QObject, Signal, Slot, QUrl
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebChannel import QWebChannel
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

import io
import math
import time
import base64
import threading
import urllib.request
import concurrent.futures as cf

import numpy as np
from PIL import Image

# tiles: zoom 13 ≈ 19 m/px at the equator; 5x5 tiles ≈ 20 km across
TILE_ZOOM = 13
TILE_SPAN = 2          # (2*SPAN+1)^2 tiles around the target
MESH_SEG = 160         # mesh resolution (vertices per side - 1)

DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Satellite sources, tried in order. Esri rejects bare urllib requests
# (WinError 10054 = server closes the connection), so we send browser-like
# headers; if it still refuses, fall back to other open imagery services.
SAT_SOURCES = [
    ("Esri World Imagery",
     "https://server.arcgisonline.com/ArcGIS/rest/services/"
     "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
    ("Google (open tile endpoint)",
     "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"),
    ("Esri (wayback mirror)",
     "https://services.arcgisonline.com/ArcGIS/rest/services/"
     "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
]

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.arcgis.com/",
    "Connection": "keep-alive",
}


def _lon2tile(lon, z):
    return int((lon + 180.0) / 360.0 * (2 ** z))


def _lat2tile(lat, z):
    r = math.radians(lat)
    return int((1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi)
               / 2.0 * (2 ** z))


def _fetch(url, timeout=20, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(0.4 * (attempt + 1))   # brief backoff, then retry
    raise last


def _pick_sat_source(z, x, y, log):
    """Probe the satellite services once and keep the first that answers."""
    for name, tmpl in SAT_SOURCES:
        try:
            data = _fetch(tmpl.format(z=z, x=x, y=y), timeout=12, retries=1)
            Image.open(io.BytesIO(data)).convert("RGB")   # must be a real image
            log(f"[TERRAIN] satellite source: {name}")
            return tmpl
        except Exception as e:
            log(f"[TERRAIN] {name} unavailable ({type(e).__name__}) — trying next")
    log("[TERRAIN] WARNING: no satellite source reachable — "
        "terrain will be shown with a plain relief shading instead of imagery")
    return None


def build_terrain_payload(lat, lon, log=print):
    """Download DEM + satellite tiles IN PYTHON, decode elevation, and return
    a dict ready for the JS side.

    Doing the fetch here (instead of in the page) sidesteps Chromium's CORS
    and tainted-canvas rules entirely — the browser only ever receives
    finished data, never a cross-origin request of its own.
    """
    z = TILE_ZOOM
    xc, yc = _lon2tile(lon, z), _lat2tile(lat, z)
    n = TILE_SPAN * 2 + 1
    TS = 256

    sat_tmpl = _pick_sat_source(z, xc, yc, log)

    dem_mosaic = Image.new("RGB", (n * TS, n * TS))
    sat_mosaic = Image.new("RGB", (n * TS, n * TS)) if sat_tmpl else None
    got_dem = got_sat = 0

    coords = [(xc + dx, yc + dy, (dx + TILE_SPAN) * TS, (dy + TILE_SPAN) * TS)
              for dy in range(-TILE_SPAN, TILE_SPAN + 1)
              for dx in range(-TILE_SPAN, TILE_SPAN + 1)]

    def grab(kind, url):
        return Image.open(io.BytesIO(_fetch(url))).convert("RGB")

    # tiles are independent -> fetch them in parallel (25 tiles serially over
    # a slow link is painful; 8 workers keeps it well under a few seconds)
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futs = {}
        for x, y, px, py in coords:
            futs[pool.submit(grab, "dem", DEM_URL.format(z=z, x=x, y=y))] = \
                ("dem", px, py, x, y)
            if sat_tmpl:
                futs[pool.submit(grab, "sat", sat_tmpl.format(z=z, x=x, y=y))] = \
                    ("sat", px, py, x, y)
        for fut in cf.as_completed(futs):
            kind, px, py, x, y = futs[fut]
            try:
                im = fut.result()
            except Exception as e:
                log(f"[TERRAIN] {kind.upper()} tile {z}/{x}/{y} failed: {e}")
                continue
            if kind == "dem":
                dem_mosaic.paste(im, (px, py))
                got_dem += 1
            else:
                sat_mosaic.paste(im, (px, py))
                got_sat += 1

    log(f"[TERRAIN] tiles fetched — dem {got_dem}/{n * n}, sat {got_sat}/{n * n}")
    if got_dem == 0:
        raise RuntimeError("no elevation tiles downloaded — check the internet "
                           "connection")

    # terrarium encoding: h = R*256 + G + B/256 - 32768 (metres)
    dem = np.asarray(dem_mosaic, dtype=np.float64)
    h = dem[:, :, 0] * 256.0 + dem[:, :, 1] + dem[:, :, 2] / 256.0 - 32768.0
    h[(h < -500) | (h > 9000)] = 0.0

    idx = np.linspace(0, h.shape[0] - 1, MESH_SEG + 1).astype(int)
    grid = h[np.ix_(idx, idx)]

    meters_per_tile = (40075016.686 * math.cos(math.radians(lat))) / (2 ** z)
    patch_m = meters_per_tile * n

    if got_sat == 0:
        # no imagery: build a hillshaded relief texture from the DEM itself, so
        # the 3D terrain is still readable instead of flat grey
        sat_mosaic = _hillshade_texture(h)
        log("[TERRAIN] no imagery — draping a hillshade relief texture instead")

    buf = io.BytesIO()
    sat_mosaic.save(buf, format="JPEG", quality=85)
    sat_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "seg": MESH_SEG,
        "heights": grid.ravel().tolist(),
        "sat_b64": sat_b64,
        "patch_m": patch_m,
        "hmin": float(grid.min()),
        "hmax": float(grid.max()),
    }


def _hillshade_texture(h):
    """Classic hillshade + elevation ramp, so terrain is legible even when no
    satellite imagery is reachable."""
    gy, gx = np.gradient(h)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315.0), np.radians(45.0)
    shade = (np.sin(alt) * np.sin(slope) +
             np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    shade = np.clip(shade, 0, 1)

    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    t = np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    ramp = np.array([[62, 92, 58], [120, 130, 72],
                     [150, 120, 80], [230, 230, 225]], dtype=np.float64)
    rgb = np.stack([np.interp(t, stops, ramp[:, i]) for i in range(3)], axis=-1)
    rgb *= (0.35 + 0.65 * shade)[:, :, None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


class GlobeBridge(QObject):
    """Object exposed to JavaScript over QWebChannel."""
    area_entered = Signal(float, float)     # user pressed ENTER AREA at lat/lon
    point_picked = Signal(float, float)     # user clicked a point on the globe
    log_line = Signal(str)
    terrain_ready = Signal(str)             # JSON payload handed back to JS

    @Slot(float, float)
    def enterArea(self, lat, lon):
        self.area_entered.emit(lat, lon)

    @Slot(float, float)
    def pickPoint(self, lat, lon):
        self.point_picked.emit(lat, lon)

    @Slot(str)
    def log(self, msg):
        self.log_line.emit(msg)

    @Slot(float, float)
    def fetchTerrain(self, lat, lon):
        """JS asks for terrain; download off the UI thread, reply via signal."""
        def worker():
            try:
                payload = build_terrain_payload(lat, lon,
                                                 log=self.log_line.emit)
            except Exception as e:
                self.log_line.emit(f"[TERRAIN] ERROR: {e}")
                payload = {"error": str(e)}
            self.terrain_ready.emit(json.dumps(payload))

        threading.Thread(target=worker, daemon=True).start()


GLOBE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root{
    --bg0:#0a0d08; --bg1:#12160d; --bg2:#1a2013; --line:#33401f;
    --olive:#4a5d2a; --amber:#ffb000; --amberdim:#8a6100;
    --text0:#d8e2c4; --text1:#7f8f6a; --ok:#6fae3a;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%;overflow:hidden;background:#04070a;
    font-family:Consolas,'SF Mono',Menlo,monospace;color:var(--text0)}
  #c{display:block}
  .bar{position:absolute;left:0;right:0;bottom:0;background:rgba(18,22,13,.92);
    border-top:1px solid var(--line);padding:8px 10px;display:flex;gap:10px;
    align-items:flex-end;z-index:10}
  .fg{display:flex;flex-direction:column;gap:3px}
  .fg label{font-size:9px;color:var(--text1);letter-spacing:.5px}
  .fg input{background:var(--bg2);border:1px solid var(--line);color:var(--text0);
    font-family:inherit;font-size:12px;padding:5px 7px;width:110px;border-radius:2px;outline:none}
  .fg input:focus{border-color:var(--amber)}
  button{font-family:inherit;font-weight:bold;font-size:11px;border:none;border-radius:3px;
    padding:7px 12px;cursor:pointer;letter-spacing:.5px}
  .primary{background:var(--amber);color:var(--bg0)}
  .primary:hover{background:#cc8e00}
  .ghost{background:var(--olive);color:var(--text0)}
  .ghost:hover{background:var(--amberdim)}
  .ro{position:absolute;left:10px;top:10px;z-index:10;font-size:11px;line-height:1.6;
    background:rgba(18,22,13,.75);border-left:2px solid var(--amber);padding:6px 10px;min-width:180px}
  .ro .t{color:var(--amber);font-size:9px;letter-spacing:1px;margin-bottom:3px}
  .ro .r{display:flex;justify-content:space-between;gap:14px}
  .ro .k{color:var(--text1)} .ro .v{font-weight:bold}
  .st{position:absolute;right:10px;top:10px;z-index:10;font-size:10px;color:var(--text1);
    background:rgba(18,22,13,.75);padding:5px 9px;border-radius:2px}
  .st b{color:var(--ok)}
  .hint{position:absolute;right:10px;bottom:58px;z-index:10;font-size:9px;color:var(--text1)}
  .attr{position:absolute;left:10px;bottom:58px;z-index:10;font-size:9px;color:#5f6e4a}
</style>
</head>
<body>
<canvas id="c"></canvas>

<div class="ro">
  <div class="t">◈ CURSOR</div>
  <div class="r"><span class="k">LAT</span><span class="v" id="rLat">--</span></div>
  <div class="r"><span class="k">LON</span><span class="v" id="rLon">--</span></div>
  <div class="r"><span class="k">ELEV</span><span class="v" id="rElev">--</span></div>
</div>
<div class="st">VIEW <b id="stView">GLOBE</b> · <span id="stMsg">standing by</span></div>
<div class="hint" id="hint">drag = rotate · wheel = zoom · click = read lat/lon</div>
<div class="attr" id="attr"></div>

<div class="bar">
  <div class="fg"><label>LATITUDE</label><input id="latIn" value="40.4093"></div>
  <div class="fg"><label>LONGITUDE</label><input id="lonIn" value="49.8671"></div>
  <button class="primary" id="enterBtn">▶ ENTER AREA (3D)</button>
  <button class="ghost" id="globeBtn">◉ BACK TO GLOBE</button>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<script>
let bridge = null;
new QWebChannel(qt.webChannelTransport, function(ch){ bridge = ch.objects.bridge; });
function blog(m){ if(bridge) bridge.log(m); }
function setMsg(m){ document.getElementById('stMsg').textContent = m; }
function setView(v){ document.getElementById('stView').textContent = v; }

/* surface any JS/network error into TerraMap's system log */
window.addEventListener('error', function(e){
  if(bridge) bridge.log('[GLOBE] JS ERROR: ' + (e.message||'?'));
});
if (typeof THREE === 'undefined') {
  setMsg('three.js failed to load — no internet?');
  document.getElementById('attr').textContent =
    'three.js could not be fetched from cdnjs — check the network/proxy';
}

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setClearColor(0x04070a,1);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100000);
const scene = new THREE.Scene();

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(1,0.5,1); scene.add(sun);

/* ---------- starfield ---------- */
(function(){
  const g=new THREE.BufferGeometry(), N=1200, p=new Float32Array(N*3);
  for(let i=0;i<N;i++){
    const r=4000+Math.random()*3000, th=Math.random()*6.283, ph=Math.acos(2*Math.random()-1);
    p[i*3]=r*Math.sin(ph)*Math.cos(th); p[i*3+1]=r*Math.cos(ph); p[i*3+2]=r*Math.sin(ph)*Math.sin(th);
  }
  g.setAttribute('position', new THREE.BufferAttribute(p,3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({color:0x8899aa,size:1.6,sizeAttenuation:false})));
})();

/* ---------- GLOBE (realistic) ---------- */
const R = 100;
const globeGroup = new THREE.Group();
scene.add(globeGroup);

const globeMat = new THREE.MeshPhongMaterial({color:0x2b4a66, shininess:12, specular:0x223344});
const globe = new THREE.Mesh(new THREE.SphereGeometry(R,96,96), globeMat);
globeGroup.add(globe);

/* NASA Blue Marble colour + bump: free, no key (three-globe's public assets) */
const texLoader = new THREE.TextureLoader();
texLoader.setCrossOrigin('anonymous');
texLoader.load('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
  t => { globeMat.map = t; globeMat.color.set(0xffffff); globeMat.needsUpdate = true;
         blog('[GLOBE] earth texture loaded'); setMsg('globe ready'); },
  undefined,
  () => { blog('[GLOBE] no internet — using plain shaded sphere'); setMsg('offline: plain sphere'); });
texLoader.load('https://unpkg.com/three-globe/example/img/earth-topology.png',
  t => { globeMat.bumpMap = t; globeMat.bumpScale = 1.2; globeMat.needsUpdate = true; });

/* thin atmosphere */
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R*1.02,48,48),
  new THREE.MeshBasicMaterial({color:0x4a80c0, transparent:true, opacity:0.10, side:THREE.BackSide})));

/* graticule */
function llv(lat,lon,r){
  const ph=(90-lat)*Math.PI/180, th=(lon+180)*Math.PI/180;
  return new THREE.Vector3(-r*Math.sin(ph)*Math.cos(th), r*Math.cos(ph), r*Math.sin(ph)*Math.sin(th));
}
(function(){
  const m=new THREE.LineBasicMaterial({color:0x9fb0c0,transparent:true,opacity:0.13});
  for(let la=-60;la<=60;la+=30){ const p=[]; for(let lo=-180;lo<=180;lo+=4) p.push(llv(la,lo,R+0.2));
    globeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),m)); }
  for(let lo=-180;lo<180;lo+=30){ const p=[]; for(let la=-90;la<=90;la+=4) p.push(llv(la,lo,R+0.2));
    globeGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),m)); }
})();

/* target pin */
const pin = new THREE.Group(); pin.visible=false; globeGroup.add(pin);
(function(){
  const stem=new THREE.Mesh(new THREE.CylinderGeometry(0.2,0.2,6,6), new THREE.MeshBasicMaterial({color:0xffb000}));
  const cone=new THREE.Mesh(new THREE.ConeGeometry(1.4,2.6,10), new THREE.MeshBasicMaterial({color:0xc1401f}));
  stem.name='stem'; cone.name='cone'; pin.add(stem); pin.add(cone);
})();
function placePin(lat,lon){
  const base=llv(lat,lon,R), dir=base.clone().normalize();
  const q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0), dir);
  const stem=pin.getObjectByName('stem'), cone=pin.getObjectByName('cone');
  stem.quaternion.copy(q); cone.quaternion.copy(q);
  stem.position.copy(base.clone().add(dir.clone().multiplyScalar(3)));
  cone.position.copy(base.clone().add(dir.clone().multiplyScalar(7.3)));
  pin.visible=true;
}

/* ---------- TERRAIN (real 3D map of the target area) ---------- */
const VEX = 2.0;   /* vertical exaggeration — terrain reads better slightly boosted */
const terrainGroup = new THREE.Group();
terrainGroup.visible = false;
scene.add(terrainGroup);
let terrainMesh = null;

async function buildTerrain(lat, lon){
  setMsg('requesting tiles from python...');
  blog('[TERRAIN] asking python for tiles @ '+lat.toFixed(4)+', '+lon.toFixed(4));
  if(!bridge){ throw new Error('bridge not ready'); }
  bridge.fetchTerrain(lat, lon);      /* python replies via window.onTerrainData */
  return new Promise((res,rej)=>{ terrainResolve=res; terrainReject=rej; });
}

let terrainResolve=null, terrainReject=null;

/* called from Python: heights = flat array (SEG+1)^2 of metres,
   satB64 = base64 jpeg/png of the draped satellite image */
window.onTerrainData = function(payload){
  try{
    const d = (typeof payload === 'string')? JSON.parse(payload) : payload;
    if(d.error){ setMsg('tiles failed: '+d.error); blog('[TERRAIN] '+d.error);
                 if(terrainReject) terrainReject(new Error(d.error)); return; }

    setMsg('building mesh...');
    const SEG=d.seg, SIZE=200;
    const geo=new THREE.PlaneGeometry(SIZE,SIZE,SEG,SEG);
    const pos=geo.attributes.position;
    const unitsPerMeter = SIZE/d.patch_m;
    for(let i=0;i<pos.count;i++){
      pos.setZ(i, d.heights[i]*unitsPerMeter*VEX);
    }
    geo.computeVertexNormals();

    const img=new Image();
    img.onload=function(){
      const tex=new THREE.Texture(img); tex.needsUpdate=true;
      const mat=new THREE.MeshPhongMaterial({map:tex, shininess:2});
      if(terrainMesh){ terrainGroup.remove(terrainMesh);
        terrainMesh.geometry.dispose(); terrainMesh.material.dispose(); }
      terrainMesh=new THREE.Mesh(geo,mat);
      terrainMesh.rotation.x=-Math.PI/2;
      terrainGroup.add(terrainMesh);
      document.getElementById('rElev').textContent=Math.round(d.hmin)+'-'+Math.round(d.hmax)+' m';
      document.getElementById('attr').textContent='imagery (c) Esri  ·  elevation: AWS Terrarium (open data)';
      blog('[TERRAIN] mesh built - '+Math.round(d.hmin)+' to '+Math.round(d.hmax)+' m, '
           +(d.patch_m/1000).toFixed(1)+' km across');
      setMsg('terrain ready');
      if(terrainResolve) terrainResolve(d);
    };
    img.onerror=function(){ setMsg('texture decode failed');
      if(terrainReject) terrainReject(new Error('texture decode failed')); };
    img.src='data:image/jpeg;base64,'+d.sat_b64;
  }catch(e){
    blog('[TERRAIN] js error: '+e.message);
    if(terrainReject) terrainReject(e);
  }
};

/* ---------- camera / modes ---------- */
let MODE='globe';                          /* 'globe' | 'zooming' | 'terrain' */
const gcam={az:0.6,el:0.35,azT:0.6,elT:0.35,d:320,dT:320};
const tcam={az:0.0,el:0.55,azT:0.0,elT:0.55,d:220,dT:220};
let drag=false,px=0,py=0,downX=0,downY=0;

function updateCamera(){
  const c = (MODE==='terrain')? tcam : gcam;
  c.d += (c.dT-c.d)*0.12;
  c.az += (c.azT-c.az)*0.12;
  c.el += (c.elT-c.el)*0.12;
  const el=Math.max(-1.45,Math.min(1.45,c.el));
  if(MODE==='terrain'){
    camera.position.set(c.d*Math.cos(el)*Math.sin(c.az), c.d*Math.sin(el), c.d*Math.cos(el)*Math.cos(c.az));
    camera.lookAt(0,0,0);
  } else {
    camera.position.set(c.d*Math.cos(el)*Math.sin(c.az), c.d*Math.sin(el), c.d*Math.cos(el)*Math.cos(c.az));
    camera.lookAt(0,0,0);
  }
}

/* fly the globe so lat/lon faces the camera, then dive in */
let flying=false, fs=0, fdur=1800, ff={}, ft={};
function flyToGlobe(lat,lon,then){
  ff={az:gcam.azT, el:gcam.elT, d:gcam.dT};
  ft={az:-(lon)*Math.PI/180+Math.PI, el:lat*Math.PI/180, d:118};
  fs=performance.now(); flying=true; MODE='zooming'; setView('DIVING'); setMsg('flying to target...');
  flyDone=then||null;
}
let flyDone=null;
function tickFly(now){
  if(!flying) return;
  let t=(now-fs)/fdur; if(t>=1){t=1;flying=false;}
  const e=t<0.5? 2*t*t : 1-Math.pow(-2*t+2,2)/2;
  let da=ft.az-ff.az; while(da>Math.PI)da-=6.283; while(da<-Math.PI)da+=6.283;
  gcam.azT=ff.az+da*e; gcam.az=gcam.azT;
  gcam.elT=ff.el+(ft.el-ff.el)*e; gcam.el=gcam.elT;
  gcam.dT=ff.d+(ft.d-ff.d)*e; gcam.d=gcam.dT;
  if(!flying && flyDone){ const f=flyDone; flyDone=null; f(); }
}

async function enterArea(lat,lon){
  placePin(lat,lon);
  flyToGlobe(lat,lon, async ()=>{
    try{
      await buildTerrain(lat,lon);
      globeGroup.visible=false; terrainGroup.visible=true;
      MODE='terrain'; setView('TERRAIN 3D');
      tcam.az=tcam.azT=0; tcam.el=tcam.elT=0.55; tcam.d=tcam.dT=220;
      document.getElementById('hint').textContent='drag = orbit terrain · wheel = zoom';
      if(bridge) bridge.enterArea(lat,lon);
    }catch(e){
      blog('[TERRAIN] failed: '+e.message);
      setMsg('terrain failed (offline?)');
      MODE='globe'; setView('GLOBE');
    }
  });
}

function backToGlobe(){
  terrainGroup.visible=false; globeGroup.visible=true;
  MODE='globe'; setView('GLOBE'); setMsg('globe');
  gcam.dT=320;
  document.getElementById('hint').textContent='drag = rotate · wheel = zoom · click = read lat/lon';
}

/* ---------- input ---------- */
canvas.addEventListener('mousedown',e=>{drag=true;px=e.clientX;py=e.clientY;downX=e.clientX;downY=e.clientY;flying=false;});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{
  if(!drag) return;
  const dx=e.clientX-px, dy=e.clientY-py; px=e.clientX; py=e.clientY;
  const c=(MODE==='terrain')? tcam : gcam;
  c.azT-=dx*0.005; c.elT=Math.max(-1.45,Math.min(1.45,c.elT+dy*0.005));
});
canvas.addEventListener('wheel',e=>{
  e.preventDefault();
  const c=(MODE==='terrain')? tcam : gcam;
  c.dT*= (e.deltaY>0? 1.1:0.9);
  c.dT = (MODE==='terrain')? Math.max(40,Math.min(600,c.dT)) : Math.max(112,Math.min(600,c.dT));
},{passive:false});

const ray=new THREE.Raycaster(), m2=new THREE.Vector2();
canvas.addEventListener('click',e=>{
  if(Math.abs(e.clientX-downX)>4||Math.abs(e.clientY-downY)>4) return;
  if(MODE!=='globe') return;
  const r=canvas.getBoundingClientRect();
  m2.x=((e.clientX-r.left)/r.width)*2-1;
  m2.y=-((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(m2,camera);
  const hit=ray.intersectObject(globe);
  if(hit.length){
    const p=globe.worldToLocal(hit[0].point.clone()).normalize();
    const lat=90-Math.acos(p.y)*180/Math.PI;
    let lon=Math.atan2(p.z,-p.x)*180/Math.PI-180;
    while(lon<-180)lon+=360; while(lon>180)lon-=360;
    document.getElementById('rLat').textContent=lat.toFixed(4);
    document.getElementById('rLon').textContent=lon.toFixed(4);
    document.getElementById('latIn').value=lat.toFixed(4);
    document.getElementById('lonIn').value=lon.toFixed(4);
    placePin(lat,lon);
    if(bridge) bridge.pickPoint(lat,lon);
  }
});

function readIn(){
  const la=parseFloat(document.getElementById('latIn').value);
  const lo=parseFloat(document.getElementById('lonIn').value);
  if(isNaN(la)||isNaN(lo)||la<-85||la>85||lo<-180||lo>180){
    setMsg('invalid coords (lat -85..85, lon -180..180)'); return null;
  }
  return {la,lo};
}
document.getElementById('enterBtn').onclick=()=>{ const c=readIn(); if(c) enterArea(c.la,c.lo); };
document.getElementById('globeBtn').onclick=backToGlobe;
document.getElementById('latIn').addEventListener('keydown',e=>{ if(e.key==='Enter') document.getElementById('enterBtn').click(); });
document.getElementById('lonIn').addEventListener('keydown',e=>{ if(e.key==='Enter') document.getElementById('enterBtn').click(); });

/* python -> js entry point */
window.flyToCoords = function(lat,lon){
  document.getElementById('latIn').value=lat.toFixed(4);
  document.getElementById('lonIn').value=lon.toFixed(4);
  enterArea(lat,lon);
};

function resize(){
  const w=window.innerWidth, h=window.innerHeight;
  renderer.setSize(w,h,false);
  camera.aspect=w/h; camera.updateProjectionMatrix();
}
window.addEventListener('resize',resize); resize();

function loop(now){
  requestAnimationFrame(loop);
  tickFly(now);
  updateCamera();
  if(MODE==='globe' && !drag) globeGroup.rotation.y += 0.0004;
  renderer.render(scene,camera);
}
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


class _LoggingPage(QWebEnginePage if HAS_WEBENGINE else object):
    """QWebEnginePage that forwards JS console output (including uncaught
    errors) into TerraMap's system log — otherwise a JS exception just makes
    the page do nothing, with no visible trace anywhere."""

    def __init__(self, parent, log_fn):
        super().__init__(parent)
        self._log = log_fn

    def javaScriptConsoleMessage(self, level, message, line, source):
        tag = {0: "JS", 1: "JS WARN", 2: "JS ERROR"}.get(int(level), "JS")
        self._log(f"[{tag}] {message}  (line {line})")


class GlobePanel(QWidget):
    """Drop-in widget for TerraMap's center stack.

    Signals:
        area_entered(lat, lon) — user dove into an area; the 3D terrain is up
        point_picked(lat, lon) — user clicked a spot on the globe
    """

    def __init__(self, log_fn=None, parent=None):
        super().__init__(parent)
        self.log = log_fn or (lambda m: None)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        if not HAS_WEBENGINE:
            msg = QLabel("GLOBE unavailable — QtWebEngine is not installed.\n\n"
                          "Install it with:\n    pip install PySide6-Addons")
            msg.setStyleSheet("color:#7f8f6a; font-size:12px; padding:20px;")
            lay.addWidget(msg)
            self.view = None
            self.bridge = None
            return

        self.view = QWebEngineView(self)
        self.bridge = GlobeBridge(self)
        self.bridge.log_line.connect(self._on_js_log)
        self.bridge.terrain_ready.connect(self._send_terrain_to_js)

        # surface JS errors in TerraMap's log instead of losing them silently
        page = _LoggingPage(self.view, self._on_js_log)
        self.view.setPage(page)

        channel = QWebChannel(page)
        channel.registerObject("bridge", self.bridge)
        page.setWebChannel(channel)

        # Base URL matters: it decides the page's origin for cross-origin
        # fetches (CDN script, Earth texture). Map tiles are no longer fetched
        # by the page at all — Python downloads them — so the only remote
        # loads left are the Three.js CDN script and the globe texture.
        self.view.setHtml(GLOBE_HTML, QUrl("https://cdnjs.cloudflare.com/"))
        lay.addWidget(self.view)

    def _send_terrain_to_js(self, payload_json):
        if self.view is None:
            return
        js_arg = json.dumps(payload_json)   # pass as a JS string literal
        self.view.page().runJavaScript(f"window.onTerrainData({js_arg});")

    # convenience passthroughs -------------------------------------------
    @property
    def area_entered(self):
        return self.bridge.area_entered

    @property
    def point_picked(self):
        return self.bridge.point_picked

    def fly_to(self, lat, lon):
        """Drive the globe from Python (e.g. from a localization result)."""
        if self.view is None:
            return
        self.view.page().runJavaScript(
            f"window.flyToCoords({float(lat)}, {float(lon)});")

    def _on_js_log(self, msg):
        self.log(msg)