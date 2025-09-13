import os, time, threading, statistics, signal, sys, json, subprocess
from collections import deque
from flask import Flask, Response, render_template_string, jsonify, request, make_response
from ultralytics import YOLO
import cv2

# ---------- Optional deps (GPS & HR) ----------
try:
    import serial
except Exception:
    serial = None
try:
    import pynmea2
except Exception:
    pynmea2 = None
try:
    from heartrate_monitor import HeartRateMonitor
except Exception:
    HeartRateMonitor = None

# ================== ENV ==================
MODEL_PATH = os.getenv("MODEL_PATH", "best_pothole.pt")
PORT       = int(os.getenv("PORT", "5000"))
CAM_INDEX  = int(os.getenv("CAMERA_INDEX", "-1"))   # -1 = auto
WIDTH      = int(os.getenv("WIDTH", "640"))
HEIGHT     = int(os.getenv("HEIGHT", "480"))
IMGSZ      = int(os.getenv("IMGSZ", "416"))
CONF       = float(os.getenv("CONF", "0.30"))
PROCESS_EVERY_N = int(os.getenv("PROCESS_EVERY_N", "1"))  # 1 tiap frame

# Ultrasonik
TRIG_PIN   = int(os.getenv("TRIG_PIN", "23"))  # BCM
ECHO_PIN   = int(os.getenv("ECHO_PIN", "24"))  # BCM
DIST_WARN1 = float(os.getenv("DIST_WARN1", "1.5"))  # m (CAUTION)
DIST_WARN2 = float(os.getenv("DIST_WARN2", "0.5"))  # m (DANGER)

# GPS (default pakai symlink otomatis /dev/serial0)
GPS_PORT   = os.getenv("GPS_PORT", "/dev/serial0")
GPS_BAUD   = int(os.getenv("GPS_BAUD", "9600"))

# Arah
LEFT_THRESH  = float(os.getenv("LEFT_THRESH",  "0.40"))  # < 40% = kiri
RIGHT_THRESH = float(os.getenv("RIGHT_THRESH", "0.60"))  # > 60% = kanan
MIN_PERSIST_FRM = int(os.getenv("MIN_PERSIST_FRM", "3"))

# Audio
AUDIO_DIR       = os.getenv("AUDIO_DIR", "sounds")
AUDIO_METHOD    = os.getenv("AUDIO_METHOD", "aplay")  # aplay|pygame
AUDIO_COOLDOWN  = float(os.getenv("AUDIO_COOLDOWN", "2.5"))
# ========================================

# ========= Globals & State =========
start_ts = time.time()
fps_val = 0.0
fps_alpha = 0.2
last_jpg = None
DETECT_ENABLED = True

_last_dir = None
_persist_count = 0
_last_audio_t = 0.0
_last_audio_kind = None
audio_lock = threading.Lock()

# --------- Ultrasonic (HC-SR04) ----------
ULTRA_READY = False
distance_lock = threading.Lock()
distance_m = None  # meter
try:
    import RPi.GPIO as GPIO
    import time as _t
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)
    GPIO.output(TRIG_PIN, False)
    _t.sleep(0.1)
    ULTRA_READY = True
    print(f"[OK ] GPIO ready TRIG={TRIG_PIN} ECHO={ECHO_PIN} (BCM)")
except Exception as e:
    print("[WARN] RPi.GPIO tidak tersedia/gagal init:", e)
    GPIO = None

def read_distance_once(timeout=0.06):
    if not ULTRA_READY:
        return None
    GPIO.output(TRIG_PIN, True); _t.sleep(10e-6); GPIO.output(TRIG_PIN, False)

    t0 = time.perf_counter()
    while GPIO.input(ECHO_PIN) == 0:
        if time.perf_counter() - t0 > timeout:
            return None
    start = time.perf_counter()
    while GPIO.input(ECHO_PIN) == 1:
        if time.perf_counter() - start > timeout:
            return None
    end = time.perf_counter()
    return (end - start) * 343.0 / 2.0

def ultrasonic_worker():
    global distance_m
    buf = deque(maxlen=5)
    while True:
        try:
            d = read_distance_once()
            if d is not None and 0.02 <= d <= 4.0:
                buf.append(d)
                with distance_lock:
                    distance_m = statistics.median(buf)
        except Exception:
            pass
        time.sleep(0.08)

if ULTRA_READY:
    threading.Thread(target=ultrasonic_worker, daemon=True).start()

# --------- GPS (NMEA) ----------
gps_lock = threading.Lock()
gps_data = {"lat": None, "lon": None, "speed_kmh": None, "time_utc": None, "valid": False}
GPS_READY = False
gps_ser = None

def gps_worker():
    global GPS_READY, gps_ser
    if serial is None or pynmea2 is None:
        print("[INFO] GPS skipped: pyserial/pynmea2 not available")
        return
    try:
        gps_ser = serial.Serial(GPS_PORT, baudrate=GPS_BAUD, timeout=1)
        GPS_READY = True
        print(f"[OK ] GPS serial opened on {GPS_PORT}@{GPS_BAUD}")
    except Exception as e:
        print(f"[WARN] Could not open GPS serial: {e}")
        return
    while True:
        try:
            line = gps_ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            if line.startswith(('$GPRMC', '$GNRMC')):
                try:
                    msg = pynmea2.parse(line)
                    lat, lon = (msg.latitude, msg.longitude)
                    # knots -> km/h
                    try:
                        spd_kmh = float(msg.spd_over_grnd or 0.0) * 1.852
                    except Exception:
                        spd_kmh = None
                    time_utc = None
                    try:
                        if getattr(msg, "datestamp", None) and getattr(msg, "timestamp", None):
                            time_utc = msg.datestamp.strftime("%Y-%m-%d") + " " + msg.timestamp.strftime("%H:%M:%S")
                    except Exception:
                        pass
                    with gps_lock:
                        gps_data.update({
                            "lat": lat if lat != 0 else None,
                            "lon": lon if lon != 0 else None,
                            "speed_kmh": spd_kmh,
                            "time_utc": time_utc,
                            "valid": (getattr(msg, "status", None) == "A"),
                        })
                except Exception:
                    pass
        except Exception:
            time.sleep(0.2)

threading.Thread(target=gps_worker, daemon=True).start()

# --------- Heart Rate (MAX30102) ----------
hr_lock = threading.Lock()
hr_metrics = {"bpm": None, "spo2": None, "ready": False}
hrm = None

def _probe_hr_attr(obj, *names):
    for n in names:
        try:
            v = getattr(obj, n, None)
            if v is None:
                continue
            if callable(v):
                v = v()
            return float(v)
        except Exception:
            continue
    return None

def hr_worker():
    global hrm
    if HeartRateMonitor is None:
        print("[INFO] HR skipped: HeartRateMonitor module not available")
        return
    try:
        hrm = HeartRateMonitor(print_raw=False, print_result=False)
        hrm.start_sensor()
        with hr_lock:
            hr_metrics["ready"] = True
        print("[OK ] MAX30102 HeartRateMonitor started")
    except Exception as e:
        print("[WARN] Failed to start HeartRateMonitor:", e)
        return

    while True:
        try:
            bpm  = _probe_hr_attr(hrm, "bpm", "BPM", "heart_rate", "HR")
            spo2 = _probe_hr_attr(hrm, "spo2", "SpO2", "SPO2")
            with hr_lock:
                if bpm  is not None: hr_metrics["bpm"]  = bpm
                if spo2 is not None: hr_metrics["spo2"] = spo2
        except Exception:
            pass
        time.sleep(0.3)

threading.Thread(target=hr_worker, daemon=True).start()

# --------- Kamera (USB) ----------
def open_cam():
    import glob, re
    devs = []
    for path in sorted(glob.glob("/dev/video*")):
        m = re.match(r"/dev/video(\d+)$", path)
        if m: devs.append(int(m.group(1)))
    candidates = ([CAM_INDEX] if CAM_INDEX >= 0 else (devs if devs else [0,1,2,10,11,12,21,22,23,31]))
    print(f"[INFO] Kandidat kamera: {candidates}")
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY, 0]
    for i in candidates:
        for be in backends:
            cap = cv2.VideoCapture(i, be)
            if not cap.isOpened():
                if cap: cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            for fcc in ['MJPG','YUYV','H264']:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fcc))
                time.sleep(0.05)
                ret, frm = cap.read()
                if ret and frm is not None:
                    print(f"[OK ] Kamera: /dev/video{i} {WIDTH}x{HEIGHT} backend={be} fourcc={fcc}")
                    return cap, i
            cap.release()
    raise RuntimeError("Tidak ada kamera yang bisa dibuka")

try:
    cap, chosen_idx = open_cam()
except Exception as e:
    print("[ERR] Gagal buka kamera:", e)
    sys.exit(1)

# --------- YOLO ---------
try:
    print(f"[INFO] Load model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("[ERR] Gagal load model:", e)
    sys.exit(1)

# --------- Audio helpers ---------
def _find_audio(kind:str):
    # cari file di AUDIO_DIR, dukung .wav .mp3 .ogg .wap (jika kamu memang pakai .wap)
    for ext in ("wav","WAV","mp3","MP3","ogg","OGG","wap","WAP"):
        p = os.path.join(AUDIO_DIR, f"{kind}.{ext}")
        if os.path.exists(p): return p
    return None

def _play_aplay(path:str):
    try:
        subprocess.Popen(["aplay","-q",path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("[WARN] aplay gagal:", e)

def _play_pygame(path:str):
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except Exception as e:
        print("[WARN] pygame audio gagal:", e)

def play_audio(kind:str):
    global _last_audio_t, _last_audio_kind
    path = _find_audio(kind)
    if not path:
        print(f"[WARN] file audio '{kind}' tidak ditemukan di {AUDIO_DIR}")
        return
    now = time.time()
    with audio_lock:
        if (now - _last_audio_t) < AUDIO_COOLDOWN and kind == _last_audio_kind:
            return
        _last_audio_t   = now
        _last_audio_kind= kind
    if AUDIO_METHOD.lower() == "pygame":
        threading.Thread(target=_play_pygame, args=(path,), daemon=True).start()
    else:
        threading.Thread(target=_play_aplay,  args=(path,), daemon=True).start()

# ------------------ Flask Web ------------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pothole Guard - Panduan & Monitor</title>

    <!-- Tailwind CSS for modern styling -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- LeafletJS for the map on the monitor page -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <!-- Google Fonts for better typography -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        body { font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background-color: #0f172a; color: #e2e8f0; }
        :root { --green: #10b981; --amber: #f59e0b; --red: #ef4444; --bg: #0b1020; --fg: #e5e7eb; --box: #111827; --line: #1f2937; }
        #page2 * { box-sizing: border-box }
        #page2 header { padding: 14px 16px; background: #0f172a; display: flex; flex-wrap: wrap; gap: 8px 24px; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line) }
        #page2 .title { font-weight: 800; letter-spacing: .3px }
        #page2 .muted { opacity: .8; font-size: 13px; }
        #page2 .wrap { max-width: 1200px; margin: 18px auto; padding: 0 16px; }
        #page2 .grid { display: grid; gap: 16px; grid-template-columns: 1fr; }
        @media(min-width: 1100px) { #page2 .grid { grid-template-columns: 1.2fr .8fr; } }
        #page2 .card { background: var(--box); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
        #page2 .card h3 { margin: 0; padding: 12px 14px; border-bottom: 1px solid var(--line); font-size: 15px; opacity: .85; }
        #page2 .video { width: 100%; display: block; background: #0b1225; aspect-ratio: 16/9; object-fit: contain; }
        #page2 .metric { padding: 14px; display: flex; gap: 10px; align-items: center; }
        #page2 .badge { padding: 6px 10px; border-radius: 999px; font-weight: 700; }
        #page2 .ok { background: rgba(16,185,129,.15); color: var(--green); border: 1px solid rgba(16,185,129,.4); }
        #page2 .warn { background: rgba(245,158,11,.15); color: var(--amber); border: 1px solid rgba(245,158,11,.4); }
        #page2 .danger { background: rgba(239,68,68,.15); color: var(--red); border: 1px solid rgba(239,68,68,.4); }
        #page2 .kv { padding: 0 14px 14px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px; opacity: .95; }
        #page2 .kv div { background: #0b1225; border: 1px solid var(--line); padding: 10px 12px; border-radius: 12px; }
        #page2 .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 0 14px 14px; }
        #page2 .btn { padding: 8px 12px; border-radius: 10px; border: 1px solid var(--line); background: #0b1225; color: var(--fg); cursor: pointer; }
        #page2 .btn:active { transform: translateY(1px); }
        #page2 .field { display: flex; align-items: center; gap: 6px; border: 1px solid var(--line); background: #0b1225; padding: 8px 10px; border-radius: 10px; }
        #page2 .field input { width: 72px; background: transparent; border: none; color: var(--fg); outline: none; }
        #page2 a { color: #60a5fa; text-decoration: none }
        #page2 footer { opacity: .6; font-size: 12px; text-align: center; padding: 16px; }
        #page2 #map { width: 100%; height: 320px; background: #0b1225; border-radius: 0 0 16px 16px; }
        @media(min-width: 1100px) { #page2 #map { height: 360px; } }
    </style>

    <script>
      // expose server thresholds to JS
      window.WARN1 = {{warn1}}; // CAUTION distance (m)
      window.WARN2 = {{warn2}}; // DANGER distance (m)
    </script>
</head>
<body class="antialiased">

    <!-- Page 1: Panduan Pengguna (User Guide) -->
    <div id="page1">
        <div class="min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
            <div class="w-full max-w-3xl bg-slate-800/50 backdrop-blur-sm rounded-2xl shadow-2xl p-6 sm:p-8 md:p-10 border border-slate-700">
                <header class="text-center mb-8">
                    <h1 class="text-3xl sm:text-4xl font-extrabold text-sky-400 flex items-center justify-center gap-3">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open-check"><path d="M8 3H2v15h7c1.7 0 3 1.3 3 3V7c0-2.2-1.8-4-4-4Z"/><path d="M16 3h6v15h-7c-1.7 0-3 1.3-3 3V7c0-2.2 1.8-4 4-4Z"/><path d="m9 12 2 2 4-4"/></svg>
                        Panduan Pothole Guard
                    </h1>
                    <p class="mt-2 text-slate-400">Untuk Pengguna Tunanetra (Kalung + Pinggang)</p>
                </header>

                <div class="space-y-6 text-slate-300 leading-relaxed">
                    <div>
                        <h2 class="font-bold text-lg text-sky-300 border-b border-slate-700 pb-2 mb-3">Tujuan Utama</h2>
                        <p>Membantu pengguna tunanetra menghindari lubang/halangan di depan dengan suara peringatan. Alat ini adalah pendamping tambahan, bukan pengganti tongkat atau pendamping manusia.</p>
                    </div>

                    <details class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                        <summary class="font-bold text-lg text-sky-300 cursor-pointer">1. Isi & Bentuk Perangkat</summary>
                        <div class="mt-3 pl-4 border-l-2 border-sky-500 text-slate-400 space-y-2">
                           <p><strong>Box A (Kalung, di dada):</strong> Berisi kamera dan sensor jarak, menghadap ke depan.</p>
                           <p><strong>Box B (Ikat pinggang):</strong> Berisi unit utama, baterai/powerbank, dan modul lain. Disembunyikan di pinggang.</p>
                           <p>Suara peringatan keluar dari speaker/earphone. Disarankan gunakan earphone <em>bone-conduction</em> agar telinga tetap mendengar lingkungan.</p>
                        </div>
                    </details>

                    <details class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                        <summary class="font-bold text-lg text-sky-300 cursor-pointer">2. Cara Memakai</summary>
                        <div class="mt-3 pl-4 border-l-2 border-sky-500 text-slate-400 space-y-2">
                           <p><strong>Kalung (Box A):</strong> Pasang di dada, posisi tengah, sedikit menunduk ke arah jalan (sekitar 10–15°). Pastikan lensa tidak tertutup pakaian.</p>
                           <p><strong>Ikat pinggang (Box B):</strong> Kencangkan agar tidak bergeser saat berjalan. Rapikan kabel.</p>
                           <p><strong>Earphone/Speaker:</strong> Pasang yang nyaman dan aman; pastikan volume cukup terdengar.</p>
                        </div>
                    </details>

                    <details class="bg-slate-900/50 p-4 rounded-lg border border-slate-700" open>
                        <summary class="font-bold text-lg text-sky-300 cursor-pointer">4. Arti Suara Peringatan</summary>
                        <div class="mt-3 pl-4 border-l-2 border-sky-500 text-slate-400 space-y-2">
                           <p><strong>“Kiri”</strong> → ada lubang/halangan di sisi kiri.</p>
                           <p><strong>“Kanan”</strong> → ada lubang/halangan di sisi kanan.</p>
                           <p><strong>“Depan”</strong> → bahaya dekat tepat di depan; langkah lebih hati-hati atau hentikan sebentar.</p>
                        </div>
                    </details>

                    <details class="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                        <summary class="font-bold text-lg text-sky-300 cursor-pointer">Lainnya (Menyalakan, Daya, FAQ)</summary>
                        <div class="mt-3 pl-4 border-l-2 border-sky-500 text-slate-400 space-y-4">
                            <p><strong>Menyalakan:</strong> Tekan tombol daya pada Box B, tunggu bunyi “siap” (30–60 detik). Bunyi 1x (nyala), 2x (siap), 3x (baterai lemah).</p>
                            <p><strong>Berjalan:</strong> Gunakan bersama tongkat. Berjalan lurus dengan dada menghadap arah jalan.</p>
                            <p><strong>Pengisian Daya:</strong> Isi daya powerbank pada Box B melalui port USB.</p>
                            <p><strong>Keselamatan:</strong> Ini alat bantu, bukan pengganti tongkat/pendamping. Selalu waspada terhadap lingkungan sekitar.</p>
                        </div>
                    </details>
                </div>

                <div class="mt-10 text-center">
                    <button onclick="showPage('page2')" class="bg-sky-600 hover:bg-sky-500 text-white font-bold py-3 px-8 rounded-full transition-all duration-300 shadow-lg shadow-sky-900/50 transform hover:scale-105">
                        Buka Monitor Alat
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Page 2: Monitor Alat (Device Dashboard) -->
    <div id="page2" class="hidden">
        <header>
            <div>
                <div class="title">Pothole Monitor</div>
                <div class="muted">YOLO · Ultrasonic · GPS · Heart Rate · Directional Audio</div>
            </div>
            <div class="flex items-center gap-4">
                 <div class="muted">Model {{model}} · imgsz {{imgsz}} · conf {{conf}} · Camera /dev/video{{cam}}</div>
                 <button onclick="showPage('page1')" class="text-sm bg-slate-700 hover:bg-slate-600 text-slate-2 00 font-semibold py-2 px-4 rounded-lg transition-colors duration-200">
                    ← Kembali ke Panduan
                 </button>
            </div>
        </header>

        <div class="wrap">
            <div class="grid">
                <div class="card">
                    <h3>Realtime Camera</h3>
                    <img class="video" src="/video" alt="Realtime camera feed placeholder" />
                    <div class="row">
                        <button class="btn" id="toggleBtn">Toggle Deteksi</button>
                        <button class="btn" onclick="snapshot()">Snapshot</button>
                        <div class="field">CONF <input id="conf" type="number" step="0.01" min="0" max="1" value="{{conf}}"></div>
                        <div class="field">IMGSZ <input id="imgsz" type="number" step="32" min="128" max="1280" value="{{imgsz}}"></div>
                        <div class="field">Every N <input id="processn" type="number" step="1" min="1" max="10" value="{{processn}}"></div>
                        <span class="muted" id="status"></span>
                    </div>
                </div>

                <div>
                    <div class="card" style="margin-bottom:16px">
                        <h3>Status & Metrics</h3>
                        <div class="metric">
                            <span id="badge" class="badge ok">OK</span>
                            <div style="font-size:38px; font-weight:800"><span id="dist">-</span> <span style="opacity:.6;font-size:18px">m</span></div>
                        </div>
                        <div class="kv">
                            <div>FPS: <b id="fps">-</b></div>
                            <div>Threshold: <b><span id="t_warn2">-</span> / <span id="t_warn1">-</span> m</b></div>
                            <div>Uptime: <b id="uptime">-</b></div>
                            <div>Detect: <b id="detect">-</b></div>
                            <div>Direction: <b id="dir">-</b></div>
                            <div>Audio: <b id="aud">-</b></div>
                            <div>Detak Jantung: <b id="hr">-</b></div>
                        </div>
                    </div>

                    <div class="card">
                        <h3>GPS Map</h3>
                        <div id="map"></div>
                        <div class="row" style="margin-top:8px">
                            <button class="btn" id="followBtn">Follow: ON</button>
                            <a id="gmapsBtn" class="btn" target="_blank" href="#">Buka di Google Maps</a>
                            <span class="muted" id="gpsStatus">menunggu data GPS…</span>
                        </div>
                        <div class="kv" style="margin-top:6px">
                            <div>Lat: <b id="lat">-</b></div>
                            <div>Lon: <b id="lon">-</b></div>
                            <div>Speed: <b id="spd">-</b></div>
                            <div>UTC: <b id="utc">-</b></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>Stop: Ctrl+C (atau systemd). Built for Raspberry Pi.</footer>
    </div>

    <script>
        // --- Page Navigation ---
        function showPage(pageId) {
            document.getElementById('page1').classList.add('hidden');
            document.getElementById('page2').classList.add('hidden');
            document.getElementById(pageId).classList.remove('hidden');
            window.scrollTo(0, 0);
            // initialize thresholds display
            const tw1 = document.getElementById('t_warn1');
            const tw2 = document.getElementById('t_warn2');
            if (tw1 && tw2) { tw1.textContent = (window.WARN1 ?? '-'); tw2.textContent = (window.WARN2 ?? '-'); }
        }

        // --- Monitor Page Logic ---
        async function getJSON(u, opt) {
            try {
                const r = await fetch(u, opt || { cache: 'no-store' });
                if (!r.ok) return null;
                return await r.json();
            } catch (error) {
                return null;
            }
        }

        // --- Leaflet Map Setup ---
        let map, marker, pathLine, follow = true;
        let mapInitialized = false;

        function ensureMap() {
            if (mapInitialized) return;
            if (document.getElementById('map') && document.getElementById('map').offsetParent !== null) {
                map = L.map('map', { zoomControl: true });
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 20,
                    attribution: '© OpenStreetMap'
                }).addTo(map);
                marker = L.marker([0, 0]).addTo(map).bindPopup('Menunggu GPS…');
                pathLine = L.polyline([], { weight: 4, opacity: 0.9, color: '#3b82f6' }).addTo(map);
                map.setView([0, 0], 2);

                const fb = document.getElementById('followBtn');
                fb.onclick = () => {
                    follow = !follow;
                    fb.textContent = 'Follow: ' + (follow ? 'ON' : 'OFF');
                };
                mapInitialized = true;
            }
        }

        async function tick() {
            if (!document.getElementById('page2').classList.contains('hidden')) {
                const j = await getJSON('/metrics');
                if (!j) { setTimeout(tick, 1000); return; }

                // General metrics
                document.getElementById('fps').textContent = j.fps ? j.fps.toFixed(1) : '-';
                document.getElementById('uptime').textContent = j.uptime_human || '-';
                document.getElementById('detect').textContent = j.detect_enabled ? 'ON' : 'OFF';
                document.getElementById('dir').textContent = j.direction || '-';
                document.getElementById('aud').textContent = j.last_audio || '-';

                // Heart rate (FIXED: use j.hr.bpm and j.hr.spo2)
                const hrBpm = j.hr && j.hr.bpm;
                const hrSpo2 = j.hr && j.hr.spo2;
                let hrTxt = '-';
                if (hrBpm != null && !Number.isNaN(hrBpm)) {
                    hrTxt = Math.round(hrBpm) + ' bpm';
                    if (hrSpo2 != null && !Number.isNaN(hrSpo2)) hrTxt += ', SpO₂ ' + Math.round(hrSpo2) + '%';
                }
                document.getElementById('hr').textContent = hrTxt;

                // Distance badge
                const d = j.distance_m;
                const distEl = document.getElementById('dist');
                const badge = document.getElementById('badge');
                if (d === null || d === undefined) {
                    distEl.textContent = '-';
                    badge.textContent = 'NO DATA';
                    badge.className = 'badge warn';
                } else {
                    distEl.textContent = d.toFixed(2);
                    const warn2 = (window.WARN2 ?? 1.0);
                    const warn1 = (window.WARN1 ?? 2.0);
                    if (d < warn2) { badge.textContent = 'DANGER'; badge.className = 'badge danger'; }
                    else if (d < warn1) { badge.textContent = 'CAUTION'; badge.className = 'badge warn'; }
                    else { badge.textContent = 'OK'; badge.className = 'badge ok'; }
                }

                // --- Update GPS & Map ---
                ensureMap();
                if (mapInitialized) {
                    const statusEl = document.getElementById('gpsStatus');
                    const latEl = document.getElementById('lat');
                    const lonEl = document.getElementById('lon');
                    const spdEl = document.getElementById('spd');
                    const utcEl = document.getElementById('utc'); // FIXED typo
                    const gmapsBtn = document.getElementById('gmapsBtn');

                    const glat = j.gps?.lat,
                          glon = j.gps?.lon;
                    const gvalid = j.gps?.valid;

                    if (glat != null && glon != null) {
                        const ll = [glat, glon];
                        marker.setLatLng(ll).setPopupContent(
                            `Lat ${glat.toFixed(6)}, Lon ${glon.toFixed(6)}<br><a target="_blank" href="https://maps.google.com/?q=${glat},${glon}">Buka di Google Maps</a>`
                        );
                        pathLine.addLatLng(ll);

                        if (follow) {
                            const targetZoom = Math.max(map.getZoom(), 16);
                            map.setView(ll, targetZoom, { animate: true });
                        }

                        latEl.textContent = glat.toFixed(6);
                        lonEl.textContent = glon.toFixed(6);
                        spdEl.textContent = j.gps?.speed_kmh ? (j.gps.speed_kmh.toFixed(1) + ' km/h') : '-';
                        utcEl.textContent = j.gps?.time_utc || '-';
                        gmapsBtn.href = `https://maps.google.com/?q=${glat},${glon}`;
                        statusEl.textContent = gvalid ? 'GPS fix OK' : 'GPS belum fix';
                    } else {
                        statusEl.textContent = 'menunggu data GPS…';
                        latEl.textContent = lonEl.textContent = '-';
                        spdEl.textContent = utcEl.textContent = '-';
                        gmapsBtn.href = '#';
                    }
                }
            }
            setTimeout(tick, 600);
        }
        tick();

        document.getElementById('toggleBtn').onclick = async () => {
            const r = await fetch('/toggle', { method: 'POST' });
            if (!r) return;
            const j = await r.json();
            document.getElementById('status').textContent = 'Detect ' + (j.detect_enabled ? 'ON' : 'OFF');
        };

        async function applySettings(){
            const body = {
                conf: parseFloat(document.getElementById('conf').value),
                imgsz: parseInt(document.getElementById('imgsz').value),
                process_n: parseInt(document.getElementById('processn').value),
            };
            const r = await fetch('/set', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
            if (!r) return;
            const j = await r.json();
            document.getElementById('status').textContent = j.msg || 'applied';
        }
        ['conf', 'imgsz', 'processn'].forEach(id => {
            document.getElementById(id).addEventListener('change', applySettings);
        });

        async function snapshot() {
            const r = await fetch('/snapshot', { cache: 'no-store' });
            if (!r) return;
            const blob = await r.blob();
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'snapshot.jpg';
            a.click();
            URL.revokeObjectURL(a.href);
        }
    </script>
</body>
</html>
"""

def bgr_color(name):
    return {
        "gray": (128,128,128),
        "red": (0,0,255),
        "amber": (0,165,255),
        "green": (0,200,0),
        "black": (0,0,0),
        "yellow": (0,255,255)
    }[name]

def decide_direction_from_boxes(boxes, w):
    """boxes: list of (x1,y1,x2,y2,conf); return 'kiri'|'kanan'|None"""
    if not boxes: return None
    # pilih bbox paling 'dekat' (y2 terbesar) — heuristik sederhana
    bx = max(boxes, key=lambda b: b[3])
    cx = (bx[0]+bx[2]) / 2.0
    nx = cx / max(1.0, float(w))
    if nx < LEFT_THRESH:  return "kiri"
    if nx > RIGHT_THRESH: return "kanan"
    return None

# --------- Video generator ----------
def gen_frames():
    global fps_val, last_jpg, _last_dir, _persist_count, _last_audio_kind
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        t0 = time.time()

        H, W = frame.shape[:2]
        out = frame
        boxes = []

        if DETECT_ENABLED and (frame_id % max(1, PROCESS_EVERY_N) == 0):
            try:
                results = model(out, imgsz=IMGSZ, conf=CONF, verbose=False)
                out = results[0].plot()
                b = results[0].boxes
                if b is not None and b.xyxy is not None and b.conf is not None:
                    for (x1,y1,x2,y2), conf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                        boxes.append((float(x1),float(y1),float(x2),float(y2),float(conf)))
            except Exception:
                out = frame

        # Garis bantu arah
        lx = int(LEFT_THRESH  * W)
        rx = int(RIGHT_THRESH * W)
        cv2.line(out, (lx,0), (lx,H), (60,60,60), 1)
        cv2.line(out, (rx,0), (rx,H), (60,60,60), 1)

        # Keputusan arah
        direction = decide_direction_from_boxes(boxes, W)

        # Overlay jarak
        with distance_lock:
            d = distance_m
        label = "Distance: -- m" if d is None else f"Distance: {d:.2f} m"
        if d is None:
            color = bgr_color("gray")
        elif d < DIST_WARN2:
            color = bgr_color("red")
        elif d < DIST_WARN1:
            color = bgr_color("amber")
        else:
            color = bgr_color("green")
        cv2.rectangle(out, (8,8), (300,48), bgr_color("black"), -1)
        cv2.putText(out, label, (14,38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Tampilkan arah (kalau ada)
        if direction:
            cv2.putText(out, f"Arah: {direction.upper()}",
                        (14,72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color("yellow"), 2, cv2.LINE_AA)

        # Audio logic: prioritas danger-depan
        danger = (d is not None and d < DIST_WARN2)
        if danger:
            play_audio("depan")
            cv2.putText(out, "DANGER", (14,106), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr_color("red"), 2, cv2.LINE_AA)
        else:
            # butuh konsistensi beberapa frame agar tidak terlalu sensitif
            if direction == _last_dir and direction is not None:
                _persist_count += 1
            else:
                _persist_count = 1
                _last_dir = direction
            if direction and _persist_count >= MIN_PERSIST_FRM:
                play_audio(direction)  # 'kiri' atau 'kanan'
                _persist_count = 0

        # FPS
        dt = time.time() - t0
        inst = 1.0 / max(dt, 1e-6)
        fps_val = fps_alpha*inst + (1.0-fps_alpha)*fps_val

        ok2, jpg = cv2.imencode('.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok2:
            last_jpg = jpg.tobytes()
            yield (b'--frame\r\nX-Accel-Buffering: no\r\nContent-Type: image/jpeg\r\n\r\n' + last_jpg + b'\r\n')
        frame_id += 1

# --------- Routes ----------
@app.route("/")
def index():
    return render_template_string(HTML, model=os.path.basename(MODEL_PATH),
                                  imgsz=IMGSZ, conf=CONF, cam=chosen_idx,
                                  warn1=DIST_WARN1, warn2=DIST_WARN2, processn=PROCESS_EVERY_N)

@app.route("/video")
def video():
    resp = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route("/snapshot")
def snapshot():
    global last_jpg
    if last_jpg is None:
        return "no frame yet", 503
    r = make_response(last_jpg)
    r.headers['Content-Type'] = 'image/jpeg'
    r.headers['Content-Disposition'] = 'attachment; filename="snapshot.jpg"'
    r.headers['Cache-Control'] = 'no-store'
    return r

@app.route("/metrics")
def metrics():
    with distance_lock:
        d = distance_m
    with gps_lock:
        g = dict(gps_data)
    with hr_lock:
        h = dict(hr_metrics)
    uptime = time.time() - start_ts
    with audio_lock:
        last_aud = _last_audio_kind
    return jsonify({
        "distance_m": (None if d is None else float(d)),
        "fps": float(fps_val),
        "camera": int(chosen_idx),
        "uptime_sec": int(uptime),
        "uptime_human": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
        "detect_enabled": DETECT_ENABLED,
        "gps": g,
        "hr": h,                       # <-- HR object (bpm, spo2, ready)
        "conf": CONF,
        "imgsz": IMGSZ,
        "process_n": PROCESS_EVERY_N,
        "ultrasonic_ready": ULTRA_READY,
        "gps_ready": GPS_READY,
        "hr_ready": h.get("ready", False),
        "model": os.path.basename(MODEL_PATH),
        "direction": _last_dir,
        "last_audio": last_aud
    })

@app.route("/toggle", methods=["POST"])
def toggle():
    global DETECT_ENABLED
    DETECT_ENABLED = not DETECT_ENABLED
    return jsonify({"detect_enabled": DETECT_ENABLED})

@app.route("/set", methods=["POST"])
def set_params():
    global CONF, IMGSZ, PROCESS_EVERY_N
    try:
        j = request.get_json(silent=True) or {}
        if "conf" in j:
            c = float(j["conf"])
            if 0 <= c <= 1: CONF = c
        if "imgsz" in j:
            s = int(j["imgsz"])
            if 128 <= s <= 1280: IMGSZ = s
        if "process_n" in j:
            n = int(j["process_n"])
            if 1 <= n <= 10: PROCESS_EVERY_N = n
        return jsonify({"ok": True, "msg": "updated", "conf": CONF, "imgsz": IMGSZ, "process_n": PROCESS_EVERY_N})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/healthz")
def healthz():
    return "ok", 200


def cleanup(*_):
    try:
        if cap: cap.release()
    except Exception:
        pass
    if ULTRA_READY and GPIO:
        try: GPIO.cleanup()
        except Exception: pass
    try:
        if gps_ser: gps_ser.close()
    except Exception:
        pass
    try:
        if hrm and hasattr(hrm, "stop_sensor"): hrm.stop_sensor()
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT,  cleanup)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
