#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, cv2, re, time, yaml, queue, threading, requests, logging, subprocess, shlex, json, sqlite3, shutil, signal
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from logging.handlers import RotatingFileHandler

"""
ANPR: ffmpeg->ROI->ALPR(docker exec) + SQLite + HTTP mini API.
- Stały kontener ALPR (alprd) z obrazem pawloxdocker/alpr:latest
- Współdzielone ROI przez hostowy katalog (HOST_ALPR_TMP) -> /data w ALPR
- Pełne logowanie + zrzuty klatek (roi/full/blur/noresult/fullstream)
- Auto-reload config.yaml i whitelist.txt (BEZ restartu procesu)
"""

# -------------------- CONFIG (auto-reload + walidacja) --------------------

def load_cfg_file(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class CfgReloader:
    def __init__(self, path="config.yaml", reload_every=2):
        self.path = path
        self.reload_every = reload_every
        self._cfg = load_cfg_file(path)
        self._last_mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        self._last_check = 0
        self._plate_regex_pat = self._cfg.get("PLATE_REGEX", "^[A-Z0-9]{4,10}$")
        self._plate_regex = re.compile(self._plate_regex_pat)
        self._force = False  # ustawiane przez SIGHUP

    def force_reload_now(self):
        self._force = True

    def _validate(self, c: dict) -> dict:
        out = dict(c)
        out["PROCESS_EVERY_N_FRAME"] = max(1, int(c.get("PROCESS_EVERY_N_FRAME", 3)))
        out["CONFIRM_FRAMES"] = max(1, int(c.get("CONFIRM_FRAMES", 2)))
        out["CONFIRM_WINDOW_SEC"] = float(c.get("CONFIRM_WINDOW_SEC", 2.0))
        out["COOLDOWN_SECONDS"] = max(1.0, float(c.get("COOLDOWN_SECONDS", 20.0)))
        out["SAVE_MIN_INTERVAL_SEC"] = float(c.get("SAVE_MIN_INTERVAL_SEC", 120.0))
        out["SAVE_MIN_HASH_DIFF"] = int(c.get("SAVE_MIN_HASH_DIFF", 8))
        out["ALPR_CONFIDENCE_MIN"] = float(c.get("ALPR_CONFIDENCE_MIN", 85.0))
        out["ALPR_COUNTRY"] = c.get("ALPR_COUNTRY", "eu")
        out["ALPR_DETECT_REGION"] = bool(c.get("ALPR_DETECT_REGION", True))
        out["ALPR_EXTRA_ARGS"] = str(c.get("ALPR_EXTRA_ARGS", "") or "").strip()
        out["ALPR_IMAGE_FORMAT"] = (c.get("ALPR_IMAGE_FORMAT", "jpg") or "jpg").lower()
        if out["ALPR_IMAGE_FORMAT"] not in ("jpg", "jpeg", "png"):
            out["ALPR_IMAGE_FORMAT"] = "jpg"
        out["DOCKER_IMAGE"] = c.get("DOCKER_IMAGE", "pawloxdocker/alpr:latest")
        out["DOCKER_USE_SUDO"] = bool(c.get("DOCKER_USE_SUDO", False))
        out["ALPR_CONTAINER_NAME"] = c.get("ALPR_CONTAINER_NAME", "alprd")
        out["HEARTBEAT_SEC"] = int(c.get("HEARTBEAT_SEC", 30))
        out["HEALTHCHECK"] = bool(c.get("HEALTHCHECK", False))
        out["CROP_PAD"] = int(c.get("CROP_PAD", 6))
        out["PLATE_REGEX"] = c.get("PLATE_REGEX", "^[A-Z0-9]{4,10}$")
        out["RTSP_URL"] = c.get("RTSP_URL")
        out["FFMPEG_WIDTH"] = int(c.get("FFMPEG_WIDTH", 1280))
        out["FFMPEG_FPS"] = int(c.get("FFMPEG_FPS", 5))
        out["FFMPEG_QUALITY"] = int(c.get("FFMPEG_QUALITY", 5))
        out["GATE_URL"] = c.get("GATE_URL")
        out["LOG_PATH"] = c.get("LOG_PATH", "./anpr.log")
        out["WHITELIST_FILE"] = c.get("WHITELIST_FILE", "./whitelist.txt")
        out["WHITELIST_RELOAD_SEC"] = int(c.get("WHITELIST_RELOAD_SEC", 15))
        # statyczny ROI
        out["ROI_REL_W"] = float(c.get("ROI_REL_W", 0.65))
        out["ROI_REL_H"] = float(c.get("ROI_REL_H", 0.45))
        # debugowe zapisy
        out["SAVE_ALL_TEST_FRAMES"] = bool(c.get("SAVE_ALL_TEST_FRAMES", True))
        out["TEST_FRAMES_DIR"] = c.get("TEST_FRAMES_DIR", "./test_frames")
        # sampling pełnych klatek co N sek (0 = wyłączone)
        out["FULLSTREAM_DUMP_EVERY_SEC"] = float(c.get("FULLSTREAM_DUMP_EVERY_SEC", 0.0))
        return out

    def get(self):
        now = time.time()
        need = self._force or (now - self._last_check >= self.reload_every)
        if need:
            self._force = False
            self._last_check = now
            try:
                mt = os.path.getmtime(self.path)
                if mt != self._last_mtime:
                    new_cfg = load_cfg_file(self.path)
                    self._cfg = self._validate(new_cfg)
                    self._last_mtime = mt
                    pat = self._cfg.get("PLATE_REGEX", "^[A-Z0-9]{4,10}$")
                    if pat != self._plate_regex_pat:
                        self._plate_regex_pat = pat
                        self._plate_regex = re.compile(pat)
                    log_system("Przeładowano config.yaml")
            except FileNotFoundError:
                pass
        self._cfg = self._validate(self._cfg)
        return self._cfg

    def plate_regex(self):
        return self._plate_regex

# -------------------- LOGI --------------------

def init_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3); fh.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(ch); logger.addHandler(fh)

def log_open(plate_disp, owner_label):
    if owner_label: logging.info(f"{plate_disp} – OTWIERANIE BRAMY – {owner_label}")
    else: logging.info(f"{plate_disp} – OTWIERANIE BRAMY")

def log_denied(plate_disp, is_new):
    if is_new: logging.info(f"{plate_disp} – BRAK UPRAWNIEŃ – WYKRYTO NOWĄ TABLICĘ")
    else: logging.info(f"{plate_disp} – BRAK UPRAWNIEŃ")

def log_ignored(plate_disp, secs): logging.info(f"{plate_disp} – IGNOROWANE (ponowne wykrycie w ciągu {secs}s)")

def log_system(msg): logging.info(f"SYSTEM – {msg}")

def log_read(plate_disp, conf): logging.info(f"ODCZYT – {plate_disp} (conf {conf:.1f})")

# -------------------- UTIL --------------------

def normalize_plate_display(txt):
    t = txt.upper()
    t = t.replace("Ó","O").replace("Ø","O").replace("Ö","O")
    t = t.replace("¡","I").replace("|","I")
    t = re.sub(r"[\-_.:/\\]", "", t)
    t = re.sub(r"\s+", "", t).strip()
    return t

_EQ = {'0':'O','O':'0','B':'8','8':'B','I':'1','1':'I'}

def _equiv(s):
    return ''.join(_EQ.get(c,c) for c in s)

def near_plate(a: str, b: str, max_dist: int = 1) -> bool:
    if not a or not b or len(a) != len(b):
        return False
    a2, b2 = _equiv(a), _equiv(b)
    prev = list(range(len(b2)+1))
    for i, ca in enumerate(a2, 1):
        cur = [i]
        for j, cb in enumerate(b2, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j-1] + 1,
                prev[j-1] + (ca != cb)
            ))
        prev = cur
    return prev[-1] <= max_dist

def phash(img_bgr) -> int:
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(g)
        low = dct[:8, :8]
        med = np.median(low[1:])
        bits = (low > med).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return 0

def hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def ssim_gray(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a; mu_b2 = mu_b * mu_b; mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    s = num / (den + 1e-9)
    return float(np.clip(s.mean(), -1, 1))

# -------------------- Whitelist (plik) --------------------

class FileWhitelist:
    def __init__(self, path, reload_every=15):
        self.path = path; self.reload_every = reload_every
        self._plates = {}; self._mt = 0; self._last_check = 0
        self._load(force=True)

    def _load(self, force=False):
        now = time.time()
        if not force and (now - self._last_check) < self.reload_every:
            return
        self._last_check = now
        try:
            mt = os.path.getmtime(self.path)
            if force or mt != self._mt:
                self._mt = mt; new_map = {}
                with open(self.path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"): continue
                        parts = [p.strip() for p in line.split(",", 1)]
                        plate = normalize_plate_display(parts[0])
                        label = parts[1] if len(parts) > 1 and parts[1] else None
                        if plate: new_map[plate] = label
                self._plates = new_map
        except FileNotFoundError:
            self._plates = {}

    def is_whitelisted(self, plate_display):
        self._load()
        p = normalize_plate_display(plate_display)
        return (p in self._plates, self._plates.get(p))

# -------------------- VIDEO VIA FFMPEG (hot-reload) --------------------

class FFMpegPipeGrabber(threading.Thread):
    def __init__(self, initial_cfg: dict, out_q, cfg_rel: "CfgReloader"):
        super().__init__(daemon=True)
        self.cfg_rel = cfg_rel
        self.q = out_q
        self.stop_flag = threading.Event()
        self.proc = None
        # aktualne parametry
        self.rtsp_url = initial_cfg.get("RTSP_URL")
        self.width = int(initial_cfg.get("FFMPEG_WIDTH", 1280))
        self.fps = int(initial_cfg.get("FFMPEG_FPS", 5))
        self.quality = int(initial_cfg.get("FFMPEG_QUALITY", 5))
        # DEBUG/STATY
        self.bytes_in = 0
        self.frames_decoded = 0
        self.last_frame_ts = 0.0
        self._stderr_tail = deque(maxlen=50)
        self._stderr_thread = None
        self._last_dump = 0.0

    def _build_cmd(self, rtsp_url, width, fps, quality):
        vf_scale = f",scale={width}:-1"
        return (
            "ffmpeg -nostdin -rtsp_transport tcp -hwaccel none "
            f"-i {shlex.quote(rtsp_url)} -an -r {fps} "
            f"-vf format=yuvj420p{vf_scale} -c:v mjpeg -q:v {quality} "
            "-f image2pipe -"
        )

    def _read_stderr(self, pipe):
        try:
            for raw in iter(pipe.readline, b""):
                line = raw.decode("utf-8", "ignore").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except Exception:
            pass

    def stats(self):
        age = time.time() - self.last_frame_ts if self.last_frame_ts > 0 else None
        return {
            "frames_decoded": self.frames_decoded,
            "bytes_in": self.bytes_in,
            "last_frame_age_sec": age,
            "queue_load": f"{self.q.qsize()}/{self.q.maxsize}",
            "proc_alive": (self.proc is not None and self.proc.poll() is None),
            "ffmpeg_tail": list(self._stderr_tail)[-5:],
            "rtsp_url": self.rtsp_url,
            "fps": self.fps, "width": self.width, "quality": self.quality,
        }

    def _ffprobe_check(self, rtsp_url):
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-rw_timeout", "15000000",
                "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                "-of", "csv=p=0", rtsp_url
            ]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if r.returncode != 0 or not (r.stdout or "").strip():
                log_system(f"FFprobe: problem z RTSP (rc={r.returncode}) STDERR='{(r.stderr or '').strip()[:200]}'")
                return False
            log_system(f"FFprobe OK: video={r.stdout.strip()}")
            return True
        except FileNotFoundError:
            log_system("FFprobe nie znaleziony w PATH – pomijam sanity-check.")
            return True
        except subprocess.TimeoutExpired:
            log_system("FFprobe timeout – kontynuuję mimo to.")
            return False
        except Exception as e:
            log_system(f"FFprobe wyjątek: {e}")
            return False

    def _start_ffmpeg(self):
        cmd = self._build_cmd(self.rtsp_url, self.width, self.fps, self.quality)
        log_system(f"FFmpeg start: {cmd}")
        self.proc = subprocess.Popen(
            shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        self._stderr_thread = threading.Thread(target=self._read_stderr, args=(self.proc.stderr,), daemon=True)
        self._stderr_thread.start()

    def _stop_ffmpeg(self):
        try:
            if self.proc and (self.proc.poll() is None):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        finally:
            self.proc = None

    def _maybe_apply_new_cfg(self):
        cfg = self.cfg_rel.get()
        new_url = cfg.get("RTSP_URL")
        new_w = int(cfg.get("FFMPEG_WIDTH", 1280))
        new_fps = int(cfg.get("FFMPEG_FPS", 5))
        new_q = int(cfg.get("FFMPEG_QUALITY", 5))
        if (new_url != self.rtsp_url) or (new_w != self.width) or (new_fps != self.fps) or (new_q != self.quality):
            log_system(f"FFmpeg: wykryto zmianę configu -> restart strumienia")
            self.rtsp_url, self.width, self.fps, self.quality = new_url, new_w, new_fps, new_q
            self._stop_ffmpeg()
            # sanity check na nowy URL (nie blokuje jeśli fail)
            self._ffprobe_check(self.rtsp_url)
            self._start_ffmpeg()

    def run(self):
        # pierwszy start
        self._ffprobe_check(self.rtsp_url)
        self._start_ffmpeg()

        buffer = bytearray()
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        last_warn = 0.0
        first_frame_logged = False

        while not self.stop_flag.is_set():
            # hot-reload parametrów ffmpeg
            self._maybe_apply_new_cfg()

            if self.proc is None or self.proc.poll() is not None:
                log_system(f"FFmpeg nie działa – próba ponownego startu")
                self._start_ffmpeg()
                first_frame_logged = False
                buffer.clear()

            try:
                chunk = self.proc.stdout.read(4096) if self.proc and self.proc.stdout else None
            except Exception:
                chunk = None

            if not chunk:
                now = time.time()
                if now - last_warn > 5.0:
                    age = self.stats()["last_frame_age_sec"]
                    age_txt = f"{age:.1f}s" if age is not None else "brak"
                    log_system(f"FFmpeg: brak nowych danych. last_frame_age={age_txt}")
                    last_warn = now
                time.sleep(0.05)
                continue

            self.bytes_in += len(chunk)
            buffer.extend(chunk)

            while True:
                i = buffer.find(SOI)
                j = buffer.find(EOI, i + 2)
                if i != -1 and j != -1:
                    jpg = bytes(buffer[i:j + 2]); del buffer[:j + 2]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.frames_decoded += 1
                        self.last_frame_ts = time.time()
                        if not first_frame_logged:
                            H, W = img.shape[:2]
                            log_system(f"FFmpeg: pierwsza klatka OK ({W}x{H}), frames_decoded={self.frames_decoded}")
                            first_frame_logged = True
                        # opcjonalny sampling fullstream
                        try:
                            cfg = self.cfg_rel.get() if self.cfg_rel else {}
                            every = float(cfg.get("FULLSTREAM_DUMP_EVERY_SEC", 0.0))
                            if every > 0 and (time.time() - self._last_dump) >= every:
                                ensure_dir("./test_frames/fullstream")
                                cv2.imwrite(os.path.join("./test_frames/fullstream", f"{int(time.time()*1000)}.jpg"),
                                            img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                self._last_dump = time.time()
                        except Exception:
                            pass
                        try:
                            self.q.put(img, timeout=0.01)
                        except queue.Full:
                            logging.info("[VIDEO] kolejka pełna – klatka odrzucona")
                    else:
                        logging.info("[VIDEO] nieudany odczyt klatki (decode JPEG failed)")
                else:
                    break

            if self.last_frame_ts > 0 and (time.time() - self.last_frame_ts) > 10.0:
                if time.time() - last_warn > 5.0:
                    log_system("FFmpeg: >10s bez klatek — możliwy brak połączenia z kamerą lub zacięcie strumienia.")
                    last_warn = time.time()

    def stop(self):
        self.stop_flag.set()
        self._stop_ffmpeg()

# -------------------- ALPR (Docker – kontener stały + exec; hot-reload) --------------------

ALPR_TMP_DIR = os.environ.get("ALPR_TMP_DIR", "/shared/alprtmp")   # ścieżka w kontenerze app (bind mount)
HOST_ALPR_TMP = os.environ.get("HOST_ALPR_TMP")                    # hostowa ścieżka; przekazywana z systemd

def ensure_dir(p):
    try: os.makedirs(p, exist_ok=True)
    except: pass

ensure_dir(ALPR_TMP_DIR)

class AlprExec:
    def __init__(self, initial_cfg: dict):
        self.image = initial_cfg.get("DOCKER_IMAGE","pawloxdocker/alpr:latest")
        self.container = initial_cfg.get("ALPR_CONTAINER_NAME","alprd")
        self.use_sudo = bool(initial_cfg.get("DOCKER_USE_SUDO", False))
        self.country_default = initial_cfg.get("ALPR_COUNTRY","eu")
        self._ensure_container()

    def refresh_from_cfg(self, cfg: dict):
        changed = False
        if cfg.get("DOCKER_IMAGE","") != self.image:
            self.image = cfg.get("DOCKER_IMAGE")
            changed = True
        if cfg.get("ALPR_CONTAINER_NAME","") != self.container:
            self.container = cfg.get("ALPR_CONTAINER_NAME")
            changed = True
        self.use_sudo = bool(cfg.get("DOCKER_USE_SUDO", self.use_sudo))
        self.country_default = cfg.get("ALPR_COUNTRY", self.country_default)
        if changed:
            log_system("ALPR: wykryto zmianę obrazu/kontenera – ensure_container()")
            self._ensure_container()

    def _sudo(self):
        return ["sudo"] if self.use_sudo else []

    def _ensure_container(self):
        host_tmp = HOST_ALPR_TMP or ALPR_TMP_DIR
        check = subprocess.run(self._sudo()+["docker","inspect","-f","{{.State.Running}}", self.container],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if check.returncode != 0:
            cmd = self._sudo()+[
                "docker","run","-d","--name", self.container, "--restart","unless-stopped",
                "-v", f"{host_tmp}:/data",
                self.image, "tail","-f","/dev/null"  # stabilny „zajmij-życie”
            ]
            subprocess.check_call(cmd)
            log_system(f"Uruchomiono kontener {self.container} z obrazem {self.image}")
        else:
            running = (check.stdout.strip().lower() == "true")
            if not running:
                subprocess.check_call(self._sudo()+["docker","start", self.container])
                log_system(f"Wznowiono kontener {self.container}")

    def run_on_file(self, img_path: str, cfg: dict, timeout: float = 5.0):
        """
        Uruchamia alpr na pliku, używając dynamicznych opcji z cfg.
        Z retry i fallbackiem 'pl'->'eu' oraz weryfikacją rozmiaru pliku w kontenerze.
        """
        self.refresh_from_cfg(cfg)
        base = os.path.basename(img_path)

        country = cfg.get("ALPR_COUNTRY", self.country_default)
        detect_region = bool(cfg.get("ALPR_DETECT_REGION", True))
        extra = str(cfg.get("ALPR_EXTRA_ARGS","") or "").strip()

        def _alpr_cmd(ctry):
            cmd = self._sudo()+["docker","exec", self.container, "alpr","-j","-c", ctry]
            cmd += ["--detect_region", "1" if detect_region else "0"]
            if extra:
                cmd += shlex.split(extra)
            cmd += [f"/data/{base}"]
            return cmd

        def _size_cmd():
            return self._sudo()+["docker","exec", self.container, "bash","-lc",
                                 f"test -s /data/{shlex.quote(base)} && stat -c%s /data/{shlex.quote(base)} || echo 0"]

        # Poczekaj aż plik „pojawi się” w kontenerze i będzie >0 B (do ~200 ms)
        size_ok = False
        for _ in range(10):
            try:
                rsz = subprocess.run(_size_cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=0.3)
                sz = int((rsz.stdout or "0").strip() or "0")
                if sz > 0:
                    size_ok = True
                    break
            except Exception:
                pass
            time.sleep(0.02)

        if not size_ok:
            logging.info(f"[ALPR] plik jeszcze niewidoczny lub pusty w kontenerze: {base}")

        def _run_once(ctry):
            return subprocess.run(_alpr_cmd(ctry), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)

        tries = 3
        data = None
        last_err = ""
        for _ in range(tries):
            r = _run_once(country)
            if r.returncode == 0:
                data = json.loads(r.stdout or "{}")
                break
            last_err = (r.stderr or "")
            if "Missing config for the country" in last_err or "does not exist" in last_err:
                logging.info(f"[ALPR] brak configu dla '{country}', fallback 'eu'")
                country = "eu"
                continue
            if "Unknown file type" in last_err or "Unsupported image type" in last_err:
                time.sleep(0.04)
                continue
            time.sleep(0.04)

        if data is None:
            logging.info(f"[ALPR] rc!=0 stderr='{last_err.strip()[:200]}'")
            return []

        out = []
        for res in data.get("results", []):
            plate = normalize_plate_display(res.get("plate",""))
            conf = float(res.get("confidence", 0.0))
            if plate:
                out.append((plate, conf))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

# -------------------- I/O: zapisy, SQLite --------------------

ODCZYTY_DIR = "odczyty"
ensure_dir(ODCZYTY_DIR)

DB_PATH = os.path.join("db", "reads.db")
ensure_dir(os.path.dirname(DB_PATH))
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS reads(
  ts REAL,
  plate TEXT,
  conf REAL,
  whitelisted INTEGER,
  label TEXT,
  image_path TEXT
)
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_reads_ts ON reads(ts)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_reads_plate ON reads(plate)")
conn.commit()

def insert_read(ts, plate, conf, ok, label, img):
    conn.execute("INSERT INTO reads VALUES(?,?,?,?,?,?)", (ts, plate, conf, int(ok), label or "", img or ""))
    conn.commit()

def append_daily_csv(ts, plate, conf, whitelisted, label, img_path, base_dir="."):
    day = time.strftime("%Y%m%d", time.localtime(ts))
    path = os.path.join(base_dir, f"reads_{day}.csv")
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))},{plate},{conf:.1f},{1 if whitelisted else 0},{(label or '').replace(',', ' ')},{img_path}\n"
    first = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if first:
            f.write("timestamp,plate,confidence,whitelisted,label,image_path\n")
        f.write(line)

def save_detection_image_fullframe(full_frame, plate, ts):
    ensure_dir(ODCZYTY_DIR)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    safe_plate = re.sub(r"[^A-Z0-9]", "", plate)
    fname = f"{stamp}_{safe_plate}.jpg"
    path = os.path.join(ODCZYTY_DIR, fname)
    cv2.imwrite(path, full_frame)
    return path

# -------------------- Pipeline: Prefilter i Workers --------------------

class Prefilter(threading.Thread):
    """Statyczny ROI (środek kadru), bez detekcji ruchu.
    Wejście: q_frames (pełne klatki), Wyjście: q_rois (ścieżka do ROI.jpg/.png, + meta).
    """
    def __init__(self, cfg_rel: CfgReloader, q_frames: queue.Queue, q_rois: queue.Queue):
        super().__init__(daemon=True)
        self.cfg_rel = cfg_rel
        self.q_frames = q_frames
        self.q_rois = q_rois
        self.stop_flag = threading.Event()
        self.frame_id = 0

    def _atomic_write(self, path_tmp, path_final, img, ext):
        # zapis do pliku tymczasowego
        if ext in ("jpg", "jpeg"):
            cv2.imwrite(path_tmp, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        else:
            cv2.imwrite(path_tmp, img)
        # fsync, aby dopchnąć na dysk/wolumin
        try:
            fd = os.open(path_tmp, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except Exception:
            pass
        # atomowe przeniesienie
        os.replace(path_tmp, path_final)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                frame = self.q_frames.get(timeout=0.2)
            except queue.Empty:
                continue
            self.frame_id += 1

            cfg = self.cfg_rel.get()
            every = int(cfg.get("PROCESS_EVERY_N_FRAME", 3))
            if self.frame_id % every != 0:
                continue

            # Statyczny ROI: środkowy prostokąt wg ROI_REL_W/H
            H, W = frame.shape[:2]
            roi_rel_w = float(cfg.get("ROI_REL_W", 0.65))
            roi_rel_h = float(cfg.get("ROI_REL_H", 0.45))
            roi_w = max(16, int(W * roi_rel_w))
            roi_h = max(16, int(H * roi_rel_h))
            x1 = max(0, W//2 - roi_w//2)
            y1 = max(0, H//2 - roi_h//2)
            x2 = min(W, x1 + roi_w)
            y2 = min(H, y1 + roi_h)
            roi = frame[y1:y2, x1:x2]

            # Ostrość: nie odrzucamy, tylko logujemy skrajnie niską
            lap = cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if lap < 1.0:
                logging.info(f"[ROI] bardzo niska ostrość (Laplacian={lap:.1f}), ale przepuszczam dalej")

            # Zapis ROI do pliku współdzielonego przez Docker (dla exec) ATOMOWO
            ts = time.time()
            ext = (cfg.get("ALPR_IMAGE_FORMAT", "jpg") or "jpg").lower()
            if ext not in ("jpg", "jpeg", "png"): ext = "jpg"
            roi_name = f"roi_{int(ts*1000)}_{self.frame_id}.{ext}"
            roi_tmp = os.path.join(ALPR_TMP_DIR, "." + roi_name + ".tmp")
            roi_path = os.path.join(ALPR_TMP_DIR, roi_name)
            self._atomic_write(roi_tmp, roi_path, roi, ext)

            # debug: zapisz ROI i full
            if cfg.get("SAVE_ALL_TEST_FRAMES", True):
                base_dir = cfg.get("TEST_FRAMES_DIR", "./test_frames")
                roi_dir = os.path.join(base_dir, "roi")
                full_dir = os.path.join(base_dir, "full")
                blur_dir = os.path.join(base_dir, "blur")
                ensure_dir(roi_dir); ensure_dir(full_dir); ensure_dir(blur_dir)
                ts_ms = int(ts*1000)
                try:
                    shutil.copy2(roi_path, os.path.join(roi_dir, f"{ts_ms}.{ext}"))
                    cv2.imwrite(os.path.join(full_dir, f"{ts_ms}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if lap < 20.0:
                        if ext in ("jpg","jpeg"):
                            cv2.imwrite(os.path.join(blur_dir, f"{ts_ms}.{ext}"), roi, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        else:
                            cv2.imwrite(os.path.join(blur_dir, f"{ts_ms}.{ext}"), roi)
                except Exception:
                    pass

            # Przekaż pełną klatkę wraz z ROI do dalszego etapu
            try:
                self.q_rois.put((roi_path, frame, ts), timeout=0.01)
            except queue.Full:
                try: os.remove(roi_path)
                except: pass

    def stop(self):
        self.stop_flag.set()

class ALPRWorker(threading.Thread):
    def __init__(self, cfg_rel: CfgReloader, alpr: AlprExec, q_rois: queue.Queue, q_results: queue.Queue):
        super().__init__(daemon=True)
        self.cfg_rel = cfg_rel
        self.alpr = alpr
        self.q_rois = q_rois
        self.q_results = q_results
        self.stop_flag = threading.Event()
        self._last_empty_log = 0.0

    def run(self):
        while not self.stop_flag.is_set():
            try:
                roi_path, full_frame, ts = self.q_rois.get(timeout=0.2)
            except queue.Empty:
                continue

            cfg = self.cfg_rel.get()
            results = self.alpr.run_on_file(roi_path, cfg=cfg, timeout=5.0)

            try:
                os.remove(roi_path)
            except Exception:
                pass

            if not results:
                now = time.time()
                logging.info("[ALPR] brak wyników dla tego ROI")
                if cfg.get("SAVE_ALL_TEST_FRAMES", True):
                    base_dir = cfg.get("TEST_FRAMES_DIR", "./test_frames")
                    nr_dir = os.path.join(base_dir, "noresult"); ensure_dir(nr_dir)
                    cv2.imwrite(os.path.join(nr_dir, f"{int(now*1000)}.jpg"), full_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if now - self._last_empty_log > 15:
                    log_system("ALPR: brak wyników dla ostatniego ROI (statyczny ROI, możliwe złe kadrowanie/ostrość).")
                    self._last_empty_log = now
                continue

            try:
                self.q_results.put((results, full_frame, ts), timeout=0.01)
            except queue.Full:
                logging.info("[ALPR] kolejka wyników pełna – odrzucono pakiet")

    def stop(self):
        self.stop_flag.set()

# -------------------- Główna logika decyzji --------------------

class ANPRCore:
    def __init__(self, cfg_rel: CfgReloader, whitelist: FileWhitelist):
        self.cfg_rel = cfg_rel
        self.whitelist = whitelist
        self.last_seen_any = {}
        self.last_ignored_log = {}
        self.last_saved = {}
        self.last_phash = {}
        self.last_full_gray = {}
        self.confirm_window = defaultdict(lambda: deque(maxlen=16))
        self.unknown_unique = set()

    def handle_results(self, results, full_frame, ts):
        cfg = self.cfg_rel.get()
        plate_re = self.cfg_rel.plate_regex()
        alpr_min_conf = float(cfg.get("ALPR_CONFIDENCE_MIN", 85.0))
        confirm_frames = int(cfg.get("CONFIRM_FRAMES", 2))
        confirm_window_sec = float(cfg.get("CONFIRM_WINDOW_SEC", 2.0))
        cooldown = float(cfg.get("COOLDOWN_SECONDS", 20.0))
        save_min_interval = float(cfg.get("SAVE_MIN_INTERVAL_SEC", max(60.0, cooldown*2)))
        save_min_hash_diff = int(cfg.get("SAVE_MIN_HASH_DIFF", 8))

        plate_display, conf = results[0]
        if conf < alpr_min_conf:
            logging.info(f"[ALPR] odrzucono: conf {conf:.1f} < {alpr_min_conf:.1f} dla {plate_display}")
            return
        if not plate_re.match(plate_display):
            logging.info(f"[ALPR] odrzucono przez regex: {plate_display}")
            return
        if len(plate_display) < 4:
            logging.info(f"[ALPR] odrzucono przez długość (<4): {plate_display}")
            return

        self.confirm_window[plate_display].append(ts)
        recent = [t for t in self.confirm_window[plate_display] if ts - t < confirm_window_sec]
        if len(recent) < confirm_frames:
            logging.info(f"[CONFIRM] {plate_display} – {len(recent)}/{confirm_frames} w {confirm_window_sec:.1f}s")
            return

        last_any = self.last_seen_any.get(plate_display, 0.0)
        if ts - last_any < cooldown:
            last_ign = self.last_ignored_log.get(plate_display, 0.0)
            if last_ign < last_any:
                log_ignored(plate_display, int(cooldown))
                self.last_ignored_log[plate_display] = ts
            return

        for prev_plate, prev_ts in list(self.last_seen_any.items()):
            if ts - prev_ts < cooldown*2 and len(prev_plate) == len(plate_display):
                if near_plate(prev_plate, plate_display, max_dist=1):
                    plate_display = prev_plate
                    break

        log_read(plate_display, conf)
        self.last_seen_any[plate_display] = ts

        # zapis pełnej klatki z deduplikacją
        can_save = True
        last_save_ts = self.last_saved.get(plate_display, 0.0)
        if ts - last_save_ts < save_min_interval:
            cur_ph = phash(full_frame)
            last_ph = self.last_phash.get(plate_display, 0)
            ham = hamming64(cur_ph, last_ph)
            gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
            ssim_val = 0.0
            if plate_display in self.last_full_gray:
                a = gray.astype(np.float64); b = self.last_full_gray[plate_display].astype(np.float64)
                C1 = (0.01 * 255) ** 2; C2 = (0.03 * 255) ** 2
                mu_a = cv2.GaussianBlur(a, (11,11), 1.5); mu_b = cv2.GaussianBlur(b, (11,11), 1.5)
                mu_a2 = mu_a*mu_a; mu_b2 = mu_b*mu_b; mu_ab = mu_a*mu_b
                sigma_a2 = cv2.GaussianBlur(a*a, (11,11), 1.5) - mu_a2
                sigma_b2 = cv2.GaussianBlur(b*b, (11,11), 1.5) - mu_b2
                sigma_ab = cv2.GaussianBlur(a*b, (11,11), 1.5) - mu_ab
                num = (2*mu_ab + C1) * (2*sigma_ab + C2)
                den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
                ssim_val = float(np.clip((num/(den+1e-9)).mean(), -1, 1))
            if ham <= save_min_hash_diff and ssim_val >= 0.92:
                can_save = False

        if can_save:
            img_path = save_detection_image_fullframe(full_frame, plate_display, ts)
            self.last_saved[plate_display] = ts
            self.last_phash[plate_display] = phash(full_frame)
            self.last_full_gray[plate_display] = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        else:
            img_path = ""

        is_ok, label = self.whitelist.is_whitelisted(plate_display)

        insert_read(ts, plate_display, conf, is_ok, label, img_path)
        append_daily_csv(ts, plate_display, conf, is_ok, label, img_path, base_dir=".")

        if is_ok:
            log_open(plate_display, label)
            gate_url = cfg.get("GATE_URL")
            if gate_url:
                try:
                    requests.get(gate_url, timeout=4.0)
                except Exception:
                    log_system("Błąd otwierania bramy – brak odpowiedzi URL")
        else:
            is_new = plate_display not in self.unknown_unique
            if is_new:
                self.unknown_unique.add(plate_display)
            log_denied(plate_display, is_new)

# -------------------- Minimalny HTTP podgląd --------------------

class MiniHTTP(threading.Thread):
    """Prosty serwer HTTP na porcie 8088 z JSON /last?limit=50 oraz /stats"""
    def __init__(self, get_stats_fn=None):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.get_stats_fn = get_stats_fn

    def run(self):
        import socket
        from urllib.parse import urlparse, parse_qs

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", 8088))
        s.listen(5)
        log_system("MiniHTTP: nasłuch na 0.0.0.0:8088 (/last?limit=50, /stats)")

        while not self.stop_flag.is_set():
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
            except socket.timeout:
                continue
            except Exception:
                continue
            try:
                req = conn.recv(2048).decode("iso-8859-1", errors="ignore")
                line = req.split("\r\n",1)[0]
                parts = line.split()
                path = parts[1] if len(parts) >= 2 else "/"
                parsed = urlparse(path)
                if parsed.path == "/last":
                    qs = parse_qs(parsed.query)
                    limit = int(qs.get("limit", [50])[0])
                    rows = conn_sql("SELECT ts,plate,conf,whitelisted,label,image_path FROM reads ORDER BY ts DESC LIMIT ?", (limit,))
                    body = json.dumps([
                        {
                            "ts": r[0],
                            "timestamp": datetime.fromtimestamp(r[0]).strftime("%Y-%m-%d %H:%M:%S"),
                            "plate": r[1],
                            "conf": r[2],
                            "whitelisted": bool(r[3]),
                            "label": r[4],
                            "image_path": r[5],
                        } for r in rows
                    ], ensure_ascii=False)
                    resp = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json; charset=utf-8\r\n"
                        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                        "Connection: close\r\n\r\n" + body
                    )
                    conn.sendall(resp.encode("utf-8"))
                elif parsed.path == "/stats":
                    st = self.get_stats_fn() if self.get_stats_fn else {}
                    body = json.dumps(st, ensure_ascii=False)
                    resp = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: application/json; charset=utf-8\r\n"
                        f"Content-Length: {len(body.encode('utf-8'))}\r\n"
                        "Connection: close\r\n\r\n" + body
                    )
                    conn.sendall(resp.encode("utf-8"))
                else:
                    body = "OK"
                    resp = (
                        "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/plain; charset=utf-8\r\n"
                        f"Content-Length: {len(body)}\r\n"
                        "Connection: close\r\n\r\n" + body
                    )
                    conn.sendall(resp.encode("utf-8"))
            except Exception:
                pass
            finally:
                try: conn.close()
                except: pass

    def stop(self):
        self.stop_flag.set()

def conn_sql(query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    return rows

# -------------------- MAIN --------------------

HEARTBEAT_LAST = 0.0

def main():
    first_cfg = load_cfg_file("config.yaml")
    log_path = first_cfg.get("LOG_PATH", "./anpr.log")
    init_logger(log_path)
    log_system("Start ANPR (static ROI + debug + test frame dumps)")

    cfg_rel = CfgReloader("config.yaml", reload_every=2)

    # SIGHUP => natychmiastowy reload
    def _on_hup(*_):
        cfg_rel.force_reload_now()
        log_system("Wymuszono reload config.yaml (SIGHUP)")
    signal.signal(signal.SIGHUP, _on_hup)

    c = cfg_rel.get()

    wl = FileWhitelist(c.get("WHITELIST_FILE", "./whitelist.txt"), c.get("WHITELIST_RELOAD_SEC", 15))

    q_frames = queue.Queue(maxsize=8)
    grab = FFMpegPipeGrabber(c, q_frames, cfg_rel=cfg_rel)
    grab.start()

    alpr = AlprExec(c)

    q_rois = queue.Queue(maxsize=16)
    q_results = queue.Queue(maxsize=16)

    pre = Prefilter(cfg_rel, q_frames, q_rois)
    pre.start()

    workers = []
    for _ in range(2):
        w = ALPRWorker(cfg_rel, alpr, q_rois, q_results)
        w.start(); workers.append(w)

    core = ANPRCore(cfg_rel, wl)

    http_srv = MiniHTTP(get_stats_fn=lambda: {
        "grabber": grab.stats(),
        "queues": {
            "frames_q": f"{q_frames.qsize()}/{q_frames.maxsize}",
            "rois_q": f"{q_rois.qsize()}/{q_rois.maxsize if hasattr(q_rois,'maxsize') else 'NA'}",
            "results_q": f"{q_results.qsize()}/{q_results.maxsize if hasattr(q_results,'maxsize') else 'NA'}",
        },
        "save_all_test_frames": c.get("SAVE_ALL_TEST_FRAMES", True),
        "test_frames_dir": c.get("TEST_FRAMES_DIR", "./test_frames"),
        "time": time.time()
    })
    http_srv.start()

    # śledzenie zmian ważnych kluczy dla WHITELIST/ALPR
    wl_path = c.get("WHITELIST_FILE", "./whitelist.txt")
    wl_reload = int(c.get("WHITELIST_RELOAD_SEC", 15))

    try:
        while True:
            hc = cfg_rel.get()

            # heartbeat
            hb = int(hc.get("HEARTBEAT_SEC", 30))
            global HEARTBEAT_LAST
            now = time.time()
            if hb > 0 and now - HEARTBEAT_LAST >= hb:
                HEARTBEAT_LAST = now
                st = grab.stats()
                age = st.get("last_frame_age_sec")
                age_txt = f"{age:.1f}s" if isinstance(age, (int,float)) and age is not None else "brak"
                log_system(
                    f"heartbeat | frames={st['frames_decoded']} bytes={st['bytes_in']} "
                    f"last_frame_age={age_txt} q_frames={st['queue_load']} ffmpeg_alive={st['proc_alive']}"
                )

            # hot-swap whitelist jeśli zmieniono plik/częstotliwość
            if hc.get("WHITELIST_FILE") != wl_path or int(hc.get("WHITELIST_RELOAD_SEC",15)) != wl_reload:
                wl_path = hc.get("WHITELIST_FILE")
                wl_reload = int(hc.get("WHITELIST_RELOAD_SEC",15))
                core.whitelist = FileWhitelist(wl_path, wl_reload)
                log_system(f"Zmieniono źródło whitelist: {wl_path} (reload={wl_reload}s)")

            # ALPRExec śledzi zmiany obrazu/kontenera wewnątrz run_on_file(),
            # ale robimy też okresowe ensure:
            alpr.refresh_from_cfg(hc)

            try:
                results, full_frame, ts = q_results.get(timeout=0.5)
            except queue.Empty:
                continue
            core.handle_results(results, full_frame, ts)

    except KeyboardInterrupt:
        pass
    finally:
        try: http_srv.stop()
        except: pass
        try: pre.stop()
        except: pass
        for w in workers:
            try: w.stop()
            except: pass
        try: grab.stop()
        except: pass
        log_system("Stop ANPR (static ROI + debug)")

if __name__=="__main__":
    main()

