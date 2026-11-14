#!/usr/bin/env python3
"""
UVR5-UI — GPU-first hardened build (isolated-worker by default).

修訂摘要（繁體中文）：
  • 預設以「隔離 worker 子程序」執行分離，CUDA 錯誤不會污染父行程的 context。
  • Gradio 4/5 相容修正、cuBLAS 穩定性修正、yt-dlp JS runtime 自動探索。
  • 2026-09-18 修訂：合奏分頁的 URL 摺疊面板移到最上方並預設展開；
    修正低取樣率上傳被就地修改兩次的競態；移除 CPU 回退中閒置的支線；
    批次覆寫改為真正的「覆寫」（0 = 使用模型預設）；
    CLI 進度去重更精簡；暫存清理對項目消失更穩健。
  • 新增：可選的 GPU 解鎖 AMP 精度設定檔（UVR_GPU_UNLOCK_PROFILE=1）。
    Kernel module 必須另外安裝，本程式不會載入它、也不會改動驅動。
    明確的 UVR_AUTOCAST 覆寫仍優先於此設定檔。
  • 2026 本版最佳化：補充繁體中文文件、於啟動與 GPU 分頁顯示
    GPU 解鎖設定檔狀態、對已解鎖 SM 8.6 硬體在未啟用設定檔時給出提示。
  • [REV-FIX-CONTAINER] 修正 `_update()` 在 Gradio 5 建立元件實例而非
    回傳 update 的相容性 Bug；修正 `_empty_slot_outputs` 誤刪臨時目錄。
  • [REV-FIX-GPU] `free_gpu()` 改為先取 `GPU.lock` 再檢查 context，
    避免與 `_run_sep_with_timeout` 的 CUDA 呼叫競態，防止 context 被
    無故標記為 dead。
  • [REV-3060] 針對 SM 8.6 / 12GB 裝置調整重試等待與 VRAM clamp 註解；
    重試階梯的 pause 改為可中斷版本。
  • [REV-INTEGRATED] GPU 分頁在 FORCE_CPU 下只印硬體摘要、不呼叫 CUDA
    runtime；VRAM clamp 區分「模型預設」與「使用者顯式 override」，
    12GB 卡自動上限仍為 512，但顯式選 768 時只警告不降級。
  • [REV-UI-FEEDBACK] 於 `run_one_ensemble_model_stream` 建立 reporter 後
    立即設定 `panels_visible = True`，避免首次點 Separate 時狀態面板空白、
    使用者誤以為「按咗冇反應」。`_LINE_SPLIT` 維持 `[\r\n]+`，
    `render()` 維持 `"\n".join(lines)`。

搜尋 "[REV]" 標籤可找到所有變更位置；本版本補文件、修競態、
核心分離邏輯與原設計完全一致。
"""
from __future__ import annotations

import os
import sys
import re
import gc
import json
import copy
import glob
import uuid
import time
import shutil
import signal
import atexit
import logging
import platform
import tempfile
import threading
import warnings
import subprocess
import collections
import concurrent.futures
import urllib.parse
from contextlib import contextmanager
from argparse import ArgumentParser

warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")

# ──────────────────────────────────────────────────────────────────────
#  環境變數預設值（必須在 import torch 之前設定）
# ──────────────────────────────────────────────────────────────────────
os.environ.setdefault("GRADIO_HEARTBEAT_INTERVAL", "8")
os.environ.setdefault("FORWARDED_ALLOW_IPS", "*")

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    if os.environ.get("UVR_EXPANDABLE_SEGMENTS", "1") == "1":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,garbage_collection_threshold:0.8")
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
            "UVR_ALLOC_CONF", "garbage_collection_threshold:0.8")

if os.environ.get("UVR_CUDA_DEBUG", "0") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ── cuBLAS 穩定性修正：強制使用傳統 cuBLAS，關閉 cuBLASLt ─────────────
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"
# ─────────────────────────────────────────────────────────────────────


class _ClientDisconnectFilter(logging.Filter):
    _NEEDLE = "ClientDisconnect"

    def filter(self, record):
        if record.levelno < logging.ERROR:
            return True
        text = record.getMessage()
        if record.exc_info:
            text += " " + "".join(
                logging.Formatter().formatException(record.exc_info))
        if self._NEEDLE in text:
            logging.getLogger("uvicorn.error").warning(
                "Client disconnected mid-request (upload aborted by the "
                "browser/client) — this is benign and can be ignored.")
            return False
        return True


_log_client_disconnect = os.environ.get("UVR_LOG_CLIENT_DISCONNECT", "0") \
    .strip().lower() in ("1", "true", "yes", "on")
if not _log_client_disconnect:
    _uvicorn_err_logger = logging.getLogger("uvicorn.error")
    if not any(isinstance(f, _ClientDisconnectFilter)
               for f in _uvicorn_err_logger.filters):
        _uvicorn_err_logger.addFilter(_ClientDisconnectFilter())


def _envflag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


FORCE_CPU = _envflag("UVR_FORCE_CPU", "0")
ALLOW_CPU_FALLBACK = _envflag("UVR_ALLOW_CPU_FALLBACK", "1")
SUBPROC_FALLBACK = _envflag("UVR_SUBPROCESS_FALLBACK", "1")
ALWAYS_SUBPROCESS = _envflag("UVR_ALWAYS_SUBPROCESS", "1")
USE_AUTOCAST = _envflag("UVR_AUTOCAST", "0")
GPU_UNLOCK_PROFILE = _envflag("UVR_GPU_UNLOCK_PROFILE", "0")
AUTOCAST_EXPLICIT = "UVR_AUTOCAST" in os.environ
MODEL_CACHE_ENABLED = _envflag("UVR_MODEL_CACHE", "0")
CUDNN_BENCHMARK = _envflag("UVR_CUDNN_BENCHMARK", "0")
MODEL_IDLE_TIMEOUT = int(os.environ.get("UVR_MODEL_IDLE_TIMEOUT", "900"))
SUBPROC_TIMEOUT = int(os.environ.get("UVR_SUBPROC_TIMEOUT", "5400"))
INPROC_TIMEOUT = int(os.environ.get("UVR_INPROC_TIMEOUT", "1800"))
MODEL_LOAD_TIMEOUT = int(os.environ.get("UVR_MODEL_LOAD_TIMEOUT", "900"))
TARGET_SR = int(os.environ.get("UVR_TARGET_SR", "44100"))
IS_WINDOWS = platform.system() == "Windows"

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.")
    raise

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

try:
    torch.backends.cuda.preferred_blas_library = "cublas"
except (AttributeError, TypeError):
    pass

try:
    import yt_dlp
except ImportError:
    yt_dlp = None
    print("[warn] yt-dlp python module not found. Will rely on yt-dlp CLI.")

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio is not installed. Run: pip install gradio")
    raise

import numpy as np
import soundfile as sf
import librosa

try:
    import assets.themes.loadThemes as loadThemes
except Exception as e:
    print(f"[warn] loadThemes import failed: {e}")

    class _LoadThemesStub:
        @staticmethod
        def load_json():
            return None

        @staticmethod
        def get_list():
            return ["NoCrypt/miku"]

        @staticmethod
        def read_json():
            return "NoCrypt/miku"

        @staticmethod
        def select_theme(*a, **k):
            pass

        @staticmethod
        def select_mode(*a, **k):
            pass

    loadThemes = _LoadThemesStub()

try:
    from audio_separator.separator import Separator
except ImportError as e:
    print(f"FATAL: audio_separator not installed: {e}")
    raise

try:
    from assets.i18n.i18n import I18nAuto
except Exception as e:
    print(f"[warn] I18nAuto import failed: {e}")

    class I18nAuto:
        def __call__(self, s):
            return s

        def _get_available_languages(self):
            return []

    i18n = I18nAuto()
else:
    i18n = I18nAuto()


def _available_languages():
    try:
        langs = i18n._get_available_languages()
        return list(langs) if langs else []
    except Exception:
        return []


try:
    from assets.presence.discord_presence import RPCManager, track_presence
except Exception as e:
    print(f"[warn] discord_presence unavailable ({e}); using no-op stubs.")

    class RPCManager:
        @staticmethod
        def start_presence():
            pass

    def track_presence(_label):
        def deco(fn):
            return fn
        return deco


# ──────────────────────────────────────────────────────────────────────
#  Gradio 4.x/5.x 相容輔助（修正「No Response」問題）
# ──────────────────────────────────────────────────────────────────────
_HAS_GR_SKIP = hasattr(gr, "skip")

def _skip():
    """等同 Gradio 3.x 的空 gr.update()。"""
    return gr.skip() if _HAS_GR_SKIP else None

def _update(comp_cls, **kwargs):
    """等同 Gradio 3.x 的 gr.update(**kwargs)。

    [REV-FIX-CONTAINER] Gradio 5 沒有 `gr.update` 時，直接
    `comp_cls(**kwargs)` 會「建立一個新元件」而非更新既有元件；
    這會導致 slot 的 Radio / Audio 被替換成新實例，前端綁定失效、
    舊元件洩漏。正確做法是回傳帶 `__type__: "update"` 的 dict。
    """
    if hasattr(gr, "update"):
        return gr.update(**kwargs)
    if not kwargs:
        return _skip()
    return {"__type__": "update", **kwargs}


# ──────────────────────────────────────────────────────────────────────
#  小型行程輔助函式
# ──────────────────────────────────────────────────────────────────────
def run_cmd(cmd, timeout=60, check=False, env=None):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, check=check, env=env)


def _popen_kwargs():
    kw = {}
    if IS_WINDOWS:
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    return kw


def _kill_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# [REV-FIX-LINE-SPLIT] 同時支援 CR / LF / CRLF，防止 worker 進度被吞。
_LINE_SPLIT = re.compile(rb"[\r\n]+")
_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")


def _pump(stream, on_line):
    buf = b""
    try:
        fd = stream.fileno()
    except Exception:
        return

    def _emit(raw):
        try:
            s = _ANSI_RE.sub(b"", raw).decode("utf-8", "replace").strip()
        except Exception:
            return
        if not s:
            return
        try:
            on_line(s)
        except Exception:
            pass

    while True:
        try:
            chunk = os.read(fd, 65536)
        except (OSError, ValueError, BrokenPipeError):
            break
        if not chunk:
            break
        buf += chunk
        parts = _LINE_SPLIT.split(buf)
        buf = parts.pop()
        for p in parts:
            _emit(p)
    if buf:
        _emit(buf)


def stream_process(cmd, timeout, on_stdout=None, on_stderr=None, env=None):
    stdout_lines, stderr_lines = [], collections.deque(maxlen=600)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, bufsize=0, **_popen_kwargs())

    def _out(line):
        stdout_lines.append(line)
        if on_stdout:
            on_stdout(line)

    def _err(line):
        stderr_lines.append(line)
        if on_stderr:
            on_stderr(line)

    t_out = threading.Thread(target=_pump, args=(proc.stdout, _out), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, _err), daemon=True)
    t_out.start()
    t_err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            proc.wait(timeout=20)
        except Exception:
            pass
    finally:
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        for p in (proc.stdout, proc.stderr):
            try:
                p.close()
            except Exception:
                pass
    return proc.returncode, stdout_lines, list(stderr_lines), timed_out


def _check_rubberband_cli():
    try:
        r = run_cmd(["rubberband", "--help"], timeout=5)
        return r.returncode == 0 or "rubberband" in (r.stdout + r.stderr).lower()
    except Exception:
        return False


RUBBERBAND_OK = _check_rubberband_cli()
if not RUBBERBAND_OK:
    print("\n⚠️  Rubber Band CLI missing. Install: apt install rubberband-cli\n")

_ONNX_CUDA_AVAILABLE = True


def _probe_onnx_cuda():
    global _ONNX_CUDA_AVAILABLE
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            _ONNX_CUDA_AVAILABLE = False
            return
        import importlib.util
        spec = importlib.util.find_spec("onnxruntime")
        if spec and spec.origin:
            cuda_lib = os.path.join(os.path.dirname(spec.origin), "capi",
                                    "libonnxruntime_providers_cuda.so")
            if os.path.exists(cuda_lib):
                result = run_cmd(["ldd", cuda_lib], timeout=5)
                if "not found" in (result.stdout or ""):
                    _ONNX_CUDA_AVAILABLE = False
                    print("\n[warn] ONNX Runtime CUDA provider libraries missing.\n")
    except Exception:
        pass


_probe_onnx_cuda()

# ──────────────────────────────────────────────────────────────────────
#  路徑
# ──────────────────────────────────────────────────────────────────────
now_dir = os.getcwd()
sys.path.append(now_dir)
config_file = os.path.join(now_dir, "assets", "config.json")
models_file = os.path.join(now_dir, "assets", "models.json")
default_settings_file = os.path.join(now_dir, "assets", "default_settings.json")
custom_settings_file = os.path.join(now_dir, "assets", "custom_settings.json")
cookies_file = os.path.join(now_dir, "assets", "cookies.txt")
models_dir = os.path.abspath("./models")
os.makedirs(models_dir, exist_ok=True)
TEMP_ROOT = os.path.join(now_dir, "temp_outputs")
os.makedirs(TEMP_ROOT, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
#  GPU 探索
# ──────────────────────────────────────────────────────────────────────
GPU_PRESENT = bool(torch.cuda.is_available())
TOTAL_VRAM_GB = 0.0
GPU_NAME = "n/a"
GPU_CAPABILITY = None
if GPU_PRESENT:
    try:
        props = torch.cuda.get_device_properties(0)
        GPU_NAME = props.name
        TOTAL_VRAM_GB = props.total_memory / (1024 ** 3)
        try:
            GPU_CAPABILITY = tuple(torch.cuda.get_device_capability(0))
        except Exception:
            GPU_CAPABILITY = None
        _t = torch.randn(128, 128, device="cuda")
        _ = _t @ _t
        torch.cuda.synchronize()
        del _t, _
        torch.cuda.empty_cache()
        print(f"[init] CUDA OK: {GPU_NAME} ({TOTAL_VRAM_GB:.1f} GB)")
        if GPU_CAPABILITY:
            print(f"[init] Compute capability: SM {GPU_CAPABILITY[0]}.{GPU_CAPABILITY[1]}")
    except Exception as _e:
        print(f"[warn] CUDA sanity check failed at startup: {_e}")
        GPU_PRESENT = False
        GPU_CAPABILITY = None

if CUDNN_BENCHMARK and GPU_PRESENT:
    try:
        torch.backends.cudnn.benchmark = True
        print("[init] cudnn.benchmark enabled.")
    except Exception:
        pass

if FORCE_CPU:
    print("[init] UVR_FORCE_CPU=1 — all separations will run on CPU.")
elif not GPU_PRESENT:
    print("[init] No CUDA device visible to PyTorch — running on CPU.")

if FORCE_CPU or not GPU_PRESENT:
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    except Exception:
        pass

if os.path.isdir("env"):
    if IS_WINDOWS:
        python_location = r".\env\python.exe"
        separator_location = r".\env\Scripts\audio-separator.exe"
    elif platform.system() == "Linux":
        python_location = "env/bin/python"
        separator_location = "env/bin/audio-separator"
    else:
        python_location = None
        separator_location = "audio-separator"
else:
    python_location = None
    separator_location = "audio-separator"

# ──────────────────────────────────────────────────────────────────────
#  [REV-GPU-UNLOCK] GPU 解鎖設定檔輔助（只做狀態顯示，不改邏輯）
# ──────────────────────────────────────────────────────────────────────
def _gpu_unlock_target_capability() -> tuple:
    """回傳 gpu-unlock 設定檔鎖定的 compute capability。

    本版本鎖定 SM 8.6（Ampere RTX 30 系列：3060 / 3060 Ti / 3070 /
    3080 / 3090 等）。這是粗略的硬體適用性判斷，並不能辨識實際晶片，
    也不能證明硬體真的已解鎖。實際解鎖請以 Zagitta/gpu-unlock
    kernel module 的安裝與驗證為準。
    """
    return (8, 6)


def _gpu_unlock_profile_state_text() -> str:
    """產生 GPU 解鎖設定檔狀態的單行文字，供啟動輸出與 GPU 分頁使用。"""
    if not GPU_PRESENT:
        return "GPU 解鎖設定檔：未套用（無可見 CUDA 裝置）"
    if FORCE_CPU:
        return "GPU 解鎖設定檔：未套用（已強制使用 CPU）"
    target = _gpu_unlock_target_capability()
    cap_txt = (f"SM {GPU_CAPABILITY[0]}.{GPU_CAPABILITY[1]}"
               if GPU_CAPABILITY else "未知")
    if GPU_UNLOCK_PROFILE:
        state = "已啟用"
    else:
        state = "未啟用（預設關閉）"
    detail = (f"目標裝置 {target[0]}.{target[1]}｜目前裝置 {cap_txt}")
    note = ""
    if AUTOCAST_EXPLICIT:
        note = "｜明確 UVR_AUTOCAST 覆寫生效"
    return f"GPU 解鎖設定檔：{state}｜{detail}{note}"


def _should_suggest_gpu_unlock() -> bool:
    if not GPU_PRESENT or FORCE_CPU or GPU_UNLOCK_PROFILE:
        return False
    if GPU_CAPABILITY != _gpu_unlock_target_capability():
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
#  作業回報器（即時狀態）
# ──────────────────────────────────────────────────────────────────────
_PCT_RE = re.compile(r"(\d{1,3})\s?%")
_NOISE_RE = re.compile(
    r"(warnings?\.warn|UserWarning|FutureWarning|torch\.load|weights_only|"
    r"^\s*$|Downloading:? *$)", re.IGNORECASE)


class JobReporter:
    """回報單一作業的即時狀態（階段、百分比、日誌、錯誤）。"""

    def __init__(self, title=""):
        self.title = title
        self.lock = threading.Lock()
        self.stage = "queued"
        self.detail = ""
        self.pct = 0.0
        self.log = collections.deque(maxlen=12)
        self.full_log = collections.deque(maxlen=400)
        self.error = None
        self.warnings = []
        self.t0 = time.time()
        self.panels_visible = False

    def stage_set(self, stage, pct=None, detail=""):
        with self.lock:
            self.stage = stage
            if detail:
                self.detail = detail
            if pct is not None:
                self.pct = max(self.pct if pct >= self.pct else 0.0, float(pct))
        self.log_add(f"» {stage}")

    def pct_set(self, pct):
        with self.lock:
            self.pct = min(0.99, max(0.0, float(pct)))

    def log_add(self, line):
        line = str(line).rstrip()
        if not line or _NOISE_RE.search(line):
            return
        with self.lock:
            self.full_log.append(line)
            if not self.log or self.log[-1] != line:
                self.log.append(line[:220])

    def warn(self, line):
        with self.lock:
            self.warnings.append(str(line)[:300])
        self.log_add(f"⚠️  {line}")

    def fail(self, msg):
        with self.lock:
            self.error = str(msg)

    def snapshot(self):
        with self.lock:
            return (self.stage, self.detail, self.pct, list(self.log),
                    self.error, time.time() - self.t0)

    def render(self, final_ok=None):
        stage, detail, pct, log, error, elapsed = self.snapshot()
        if not self.panels_visible and final_ok is None:
            return ""
        bar_w = 24
        filled = int(round(pct * bar_w))
        bar = "█" * filled + "░" * (bar_w - filled)
        head = self.title or "Job"
        mins, secs = divmod(int(elapsed), 60)
        lines = [f"{head}"]
        if final_ok is True:
            lines.append(f"✅ finished in {mins:d}m{secs:02d}s")
        elif final_ok is False:
            lines.append(f"❌ failed after {mins:d}m{secs:02d}s")
        else:
            lines.append(f"⏳ {stage}{(' — ' + detail) if detail else ''}")
        lines.append(f"[{bar}] {int(pct * 100):3d}%   ({mins:d}m{secs:02d}s)")
        if error:
            lines.append("")
            lines.append("── error ──")
            lines.append(str(error)[:1800])
        if log:
            lines.append("")
            lines.append("── log ──")
            lines.extend(log[-8:])
        # [REV-FIX-RENDER] 狀態必須換行顯示，不能黏成一條。
        return "\n".join(lines)

    def tail(self, n=40):
        with self.lock:
            return "\n".join(list(self.full_log)[-n:])


# ──────────────────────────────────────────────────────────────────────
#  GPU 狀態 / 記憶體輔助
# ──────────────────────────────────────────────────────────────────────
_separator_lock = threading.RLock()
JOB_LOCK = threading.RLock()


def _acquire_sep_lock(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _separator_lock.acquire(blocking=False):
            return True
        time.sleep(0.1)
    return False


class _GpuState:
    def __init__(self):
        self.lock = threading.RLock()
        self.context_dead = False
        self.fatal_events = 0
        self.oom_events = 0
        self.last_error = ""
        self.last_recovery = 0.0


GPU = _GpuState()


def gpu_configured() -> bool:
    return GPU_PRESENT and not FORCE_CPU


def gpu_context_alive() -> bool:
    with GPU.lock:
        return gpu_configured() and not GPU.context_dead


def mark_context_dead(reason: str = ""):
    with GPU.lock:
        first = not GPU.context_dead
        GPU.context_dead = True
        GPU.fatal_events += 1
        GPU.last_error = str(reason)[:500]
    if first:
        print(f"[cuda] in-process context lost: {str(reason)[:300]}")


def note_oom(reason: str = ""):
    with GPU.lock:
        GPU.oom_events += 1
        GPU.last_error = str(reason)[:500]


def _raw_gpu_probe() -> bool:
    if not gpu_configured():
        return False
    try:
        a = torch.randn(512, 512, device="cuda")
        b = a @ a
        torch.cuda.synchronize()
        del a, b
        torch.cuda.empty_cache()
        return True
    except BaseException:
        return False


def gpu_healthy(allow_recovery: bool = True) -> bool:
    if not gpu_configured():
        return False
    with GPU.lock:
        dead = GPU.context_dead
    if dead and not allow_recovery:
        return False
    ok = _raw_gpu_probe()
    with GPU.lock:
        if ok and GPU.context_dead:
            GPU.context_dead = False
            GPU.last_recovery = time.time()
            print("[cuda] context recovered.")
        elif not ok and not GPU.context_dead:
            GPU.context_dead = True
            GPU.last_error = "preflight probe failed"
    return ok


# [REV-FIX-GPU] 分離進行中（_separator_lock 被持有）時不碰 CUDA API，
# 只做 gc.collect()，避免與 sep.separate() 競爭把 context 誤標為 dead。
def free_gpu(deep: bool = False):
    try:
        gc.collect()
        if not gpu_configured():
            return
        if _separator_lock._is_owned():
            return
        with GPU.lock:
            if GPU.context_dead:
                return
        try:
            avail = torch.cuda.is_available() and torch.cuda.is_initialized()
        except Exception:
            avail = False
        if avail:
            for fn in (torch.cuda.synchronize, torch.cuda.empty_cache):
                try:
                    fn()
                except BaseException:
                    mark_context_dead("free_gpu")
                    break
            if deep:
                for fn in (torch.cuda.ipc_collect,
                           torch.cuda.reset_peak_memory_stats,
                           torch.cuda.reset_accumulated_memory_stats):
                    try:
                        fn()
                    except Exception:
                        pass
        gc.collect()
    except Exception:
        pass


def vram_free_gb():
    if not gpu_configured():
        return None
    try:
        with GPU.lock:
            dead = GPU.context_dead
        if not dead and torch.cuda.is_available():
            free_b, _tot = torch.cuda.mem_get_info()
            return free_b / (1024 ** 3)
    except Exception:
        pass
    smi = _nvidia_smi()
    if smi:
        return max(0.0, (smi["total_mb"] - smi["used_mb"]) / 1024.0)
    return None


def _nvidia_smi():
    try:
        r = run_cmd(["nvidia-smi",
                     "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                     "--format=csv,noheader,nounits"], timeout=8)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        first = r.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        return {"name": parts[0], "used_mb": float(parts[1]), "total_mb": float(parts[2]),
                "util": parts[3], "temp": parts[4]}
    except Exception:
        return None


def wait_for_vram(min_gb, timeout=45, reporter=None):
    if not gpu_configured() or min_gb <= 0:
        return True
    deadline = time.time() + timeout
    warned = False
    unknown_since = None
    while time.time() < deadline:
        free = vram_free_gb()
        if free is None:
            if unknown_since is None:
                unknown_since = time.time()
            elif time.time() - unknown_since > 6:
                return True
            time.sleep(1)
            continue
        if free >= min_gb:
            return True
        if reporter and not warned:
            reporter.log_add(f"waiting for VRAM: {free:.2f} GB free, need ~{min_gb:.2f} GB")
            warned = True
        time.sleep(2)
    return False


def _is_cuda_err(e) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in (
        "cuda", "cublas", "cudnn", "illegal memory access", "an illegal instruction",
        "device-side assert", "out of memory", "no kernel image", "misaligned address"))


def _is_oom(e) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg or "cuda_error_out_of_memory" in msg
            or "cudaerrormemoryallocation" in msg) and "illegal" not in msg


@contextmanager
def _cpu_only_torch():
    orig_avail = torch.cuda.is_available
    orig_count = torch.cuda.device_count
    mps_mod = getattr(torch.backends, "mps", None)
    orig_mps = getattr(mps_mod, "is_available", None) if mps_mod is not None else None
    try:
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0
        if orig_mps is not None:
            mps_mod.is_available = lambda: False
        yield
    finally:
        torch.cuda.is_available = orig_avail
        torch.cuda.device_count = orig_count
        if orig_mps is not None:
            mps_mod.is_available = orig_mps


# ──────────────────────────────────────────────────────────────────────
#  gr.Progress 安全包裝
# ──────────────────────────────────────────────────────────────────────
def _envflag_cli(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


CLI_PROGRESS_MODE = _envflag_cli("UVR_LOG_ONLY", "0")
_CLI_MIN_DELTA = float(os.environ.get("UVR_CLI_MIN_DELTA", "0.04"))
_CLI_MIN_INTERVAL = float(os.environ.get("UVR_CLI_MIN_INTERVAL", "0.9"))


class _CLIProgress:
    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self._last_val = -1.0
        self._last_desc = None
        self._last_t = 0.0

    def __call__(self, value=0, desc=None, total=None, unit=None):
        if self._inner is None or CLI_PROGRESS_MODE:
            return
        try:
            v = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        d = str(desc) if desc is not None else None
        now = time.time()
        with self._lock:
            done = v >= 1.0
            if done:
                forward = True
            elif d is not None and d != self._last_desc:
                forward = True
            elif (abs(v - self._last_val) < _CLI_MIN_DELTA and
                    (now - self._last_t) < _CLI_MIN_INTERVAL):
                forward = False
            else:
                forward = True
            if forward:
                self._last_val, self._last_desc, self._last_t = v, d, now
        if forward:
            try:
                self._inner(value, desc=desc, total=total, unit=unit)
            except TypeError:
                try:
                    self._inner(value, desc=desc)
                except Exception:
                    pass
            except Exception:
                pass


class _NoopProgress:
    def __call__(self, *args, **kwargs):
        return None


def _cli_progress(progress):
    if progress is None or CLI_PROGRESS_MODE:
        return _NoopProgress()
    if isinstance(progress, _CLIProgress):
        return progress
    return _CLIProgress(progress)


@contextmanager
def heartbeat_progress(progress, interval=3, desc="Processing...", progress_value=0.05):
    stop = threading.Event()

    def _hb():
        while not stop.is_set():
            if stop.wait(interval):
                break
            try:
                progress(progress_value, desc=desc)
            except Exception:
                pass

    t = threading.Thread(target=_hb, daemon=True, name="heartbeat")
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=2)


# ──────────────────────────────────────────────────────────────────────
#  樣本率保證輔助
# ──────────────────────────────────────────────────────────────────────
def _ffmpeg_available():
    try:
        return run_cmd(["ffmpeg", "-version"], timeout=8).returncode == 0
    except Exception:
        return False


FFMPEG_OK = _ffmpeg_available()
if not FFMPEG_OK:
    print("⚠️  ffmpeg not found on PATH.")

_LOSSLESS_EXT = {".wav", ".flac", ".aiff", ".aif"}
_ENCODER_FOR_EXT = {
    ".wav": ["-c:a", "pcm_s24le"],
    ".flac": ["-c:a", "flac", "-compression_level", "5"],
    ".aiff": ["-c:a", "pcm_s16be"],
    ".aif": ["-c:a", "pcm_s16be"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    ".m4a": ["-c:a", "aac", "-b:a", "320k"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "8"],
    ".opus": ["-c:a", "libopus", "-b:a", "256k"],
    ".ac3": ["-c:a", "ac3", "-b:a", "640k"],
}


def probe_sr(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        return int(sf.info(path).samplerate)
    except Exception:
        pass
    if FFMPEG_OK:
        try:
            r = run_cmd(["ffprobe", "-v", "error", "-select_streams", "a:0",
                         "-show_entries", "stream=sample_rate",
                         "-of", "default=nw=1:nk=1", path], timeout=25)
            if r.returncode == 0 and r.stdout.strip():
                return int(r.stdout.strip().splitlines()[0])
        except Exception:
            pass
    return None


def ensure_min_samplerate(path, min_sr=TARGET_SR, reporter=None):
    if not path or not os.path.isfile(path):
        return path
    ext = os.path.splitext(path)[1].lower() or ".wav"
    sr = probe_sr(path)
    if sr is None:
        return path
    target = 48000 if ext == ".opus" else min_sr
    if sr >= target:
        return path
    msg = f"upsampling {os.path.basename(path)}: {sr} Hz → {target} Hz"
    print(f"[sr] {msg}")
    if reporter:
        reporter.log_add(msg)
    if FFMPEG_OK:
        enc = _ENCODER_FOR_EXT.get(ext, ["-c:a", "pcm_s24le"])
        tmp = f"{os.path.splitext(path)[0]}.__sr__{ext}"
        for af in (["-af", "aresample=resampler=soxr:precision=28"], []):
            cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path]
                   + af + ["-ar", str(target)] + enc + [tmp])
            try:
                r = run_cmd(cmd, timeout=1800)
            except Exception:
                r = None
            if r is not None and r.returncode == 0 and os.path.isfile(tmp) \
                    and os.path.getsize(tmp) > 0:
                try:
                    os.replace(tmp, path)
                    return path
                except OSError:
                    pass
                if os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
    if ext in _LOSSLESS_EXT:
        try:
            y, orig = librosa.load(path, sr=None, mono=False)
            y = librosa.resample(y, orig_sr=orig, target_sr=target, res_type="soxr_hq")
            data = y.T if y.ndim > 1 else y
            sf.write(path, data, target)
        except Exception as e:
            if reporter:
                reporter.warn(f"could not upsample {os.path.basename(path)}: {e}")
    return path


def ensure_min_samplerate_many(paths, reporter=None):
    out = []
    for p in paths or []:
        try:
            out.append(ensure_min_samplerate(p, TARGET_SR, reporter))
        except Exception as e:
            if reporter:
                reporter.warn(f"sample-rate check failed for {os.path.basename(str(p))}: {e}")
            out.append(p)
    return out


# ──────────────────────────────────────────────────────────────────────
#  分離器建構
# ──────────────────────────────────────────────────────────────────────
def _arch_kwargs(arch, kw):
    params = dict(kw)
    if arch in ("roformer", "mdx23c"):
        params.setdefault("pitch_shift", 0)
        return {"mdxc_params": params}
    if arch == "mdxnet":
        return {"mdx_params": params}
    if arch == "vrarch":
        return {"vr_params": params}
    if arch == "demucs":
        return {"demucs_params": params}
    return {}


def make_separator(common, optional, arch_kwargs):
    opt = dict(optional)
    while True:
        try:
            return Separator(**common, **opt, **arch_kwargs)
        except TypeError as e:
            if not opt:
                raise
            hit = next((k for k in opt if k in str(e)), None)
            opt.pop(hit if hit else list(opt)[-1])


def _resolve_gpu_unlock_autocast(torch_runtime, arch, legacy_autocast=False,
                                 profile_enabled=False, autocast_explicit=False,
                                 force_cpu=False):
    if not profile_enabled or autocast_explicit:
        return bool(legacy_autocast)
    if force_cpu or sys.platform != "linux":
        return False
    if arch not in ("roformer", "mdx23c"):
        return False
    try:
        if not torch_runtime.cuda.is_available():
            return False
        device = torch_runtime.cuda.current_device()
        capability = tuple(torch_runtime.cuda.get_device_capability(device))
        return capability == (8, 6)
    except Exception:
        return False


def _build_sep(arch, output_dir, out_format, norm_thresh, single_stem, **kw):
    common = dict(
        log_level=logging.WARNING,
        model_file_dir=models_dir,
        output_dir=output_dir,
        output_format=out_format,
        normalization_threshold=float(norm_thresh),
        output_single_stem=(None if arch == "demucs" else single_stem),
    )
    resolved_autocast = _resolve_gpu_unlock_autocast(
        torch, arch, USE_AUTOCAST, GPU_UNLOCK_PROFILE, AUTOCAST_EXPLICIT,
        force_cpu=FORCE_CPU)
    if GPU_UNLOCK_PROFILE and not AUTOCAST_EXPLICIT and resolved_autocast:
        print(f"[gpu-unlock] in-process build: arch={arch} use_autocast={resolved_autocast}")
    optional = {
        "use_autocast": resolved_autocast,
        "sample_rate": TARGET_SR,
    }
    return make_separator(common, optional, _arch_kwargs(arch, kw))


def _dispose_separator(sep):
    if sep is None:
        return
    try:
        mi = getattr(sep, "model_instance", None)
        if mi is not None:
            for attr in ("model_run", "model", "models", "demucs_model", "model_data",
                         "ort_session", "onnx_session", "infer_session", "session",
                         "primary_source", "secondary_source", "primary_source_map",
                         "secondary_source_map", "audio_file_path", "cached_sources_map"):
                try:
                    if hasattr(mi, attr):
                        setattr(mi, attr, None)
                except Exception:
                    pass
        try:
            sep.model_instance = None
        except Exception:
            pass
        for attr in ("torch_device", "torch_device_cuda", "torch_device_mps"):
            try:
                if hasattr(sep, attr):
                    setattr(sep, attr, None)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        del sep
    gc.collect()
    free_gpu(deep=True)


class _ModelCache:
    def __init__(self):
        self.key = None
        self.sep = None
        self.last_used = 0.0
        self.lock = threading.RLock()

    def clear(self, quiet=True):
        with self.lock:
            sep, self.sep, self.key = self.sep, None, None
        if sep is not None:
            if not quiet:
                print("[cache] releasing cached model")
            _dispose_separator(sep)
        else:
            free_gpu(deep=True)

    def reap_if_idle(self, timeout):
        if timeout <= 0:
            return
        if not self.lock.acquire(blocking=False):
            return
        try:
            if self.sep is not None and (time.time() - self.last_used) > timeout:
                if _separator_lock.acquire(blocking=False):
                    try:
                        print("[cache] idle timeout — freeing model memory")
                        self.clear()
                    finally:
                        _separator_lock.release()
        finally:
            self.lock.release()


MODEL_CACHE = _ModelCache()


def _retarget_separator(sep, output_dir, out_format, norm_thresh, single_stem, arch):
    try:
        sep.output_dir = output_dir
        sep.output_format = out_format
        sep.normalization_threshold = float(norm_thresh)
        sep.output_single_stem = (None if arch == "demucs" else single_stem)
        mi = getattr(sep, "model_instance", None)
        if mi is not None:
            if not hasattr(mi, "output_dir"):
                return False
            mi.output_dir = output_dir
            for attr, val in (("output_format", out_format),
                              ("normalization_threshold", float(norm_thresh)),
                              ("output_single_stem",
                               None if arch == "demucs" else single_stem)):
                if hasattr(mi, attr):
                    setattr(mi, attr, val)
            if getattr(mi, "output_dir", None) != output_dir:
                return False
        return True
    except Exception:
        return False


def _cache_key(arch, model_fn, out_format, norm_thresh, single_stem, kw, cpu):
    try:
        kws = json.dumps(kw, sort_keys=True, default=str)
    except Exception:
        kws = repr(sorted(kw.items()))
    return (arch, model_fn, out_format, round(float(norm_thresh), 4),
            single_stem or "", bool(cpu), kws)


def _load_model_with_timeout(sep, model_fn, timeout):
    def _load():
        sep.load_model(model_filename=model_fn)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load)
        try:
            fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Model loading timed out after {timeout}s.")


def _acquire_separator(arch, model_fn, output_dir, out_format, norm_thresh,
                       single_stem, kw, cpu=False):
    key = _cache_key(arch, model_fn, out_format, norm_thresh, single_stem, kw, cpu)
    with MODEL_CACHE.lock:
        if MODEL_CACHE_ENABLED and MODEL_CACHE.sep is not None and MODEL_CACHE.key == key:
            if _retarget_separator(MODEL_CACHE.sep, output_dir, out_format,
                                   norm_thresh, single_stem, arch):
                MODEL_CACHE.last_used = time.time()
                return MODEL_CACHE.sep, True
            MODEL_CACHE.clear()
        else:
            MODEL_CACHE.clear()
    if cpu:
        with _cpu_only_torch():
            sep = _build_sep(arch, output_dir, out_format, norm_thresh, single_stem, **kw)
            _load_model_with_timeout(sep, model_fn, MODEL_LOAD_TIMEOUT)
    else:
        sep = _build_sep(arch, output_dir, out_format, norm_thresh, single_stem, **kw)
        _load_model_with_timeout(sep, model_fn, MODEL_LOAD_TIMEOUT)
    if MODEL_CACHE_ENABLED:
        MODEL_CACHE.sep = sep
        MODEL_CACHE.key = key
        MODEL_CACHE.last_used = time.time()
    return sep, False


def _run_sep_with_timeout(sep, audio_path, cpu, timeout):
    holder = {"result": None, "error": None}
    done = threading.Event()

    def _run():
        try:
            with _separator_lock:
                with torch.no_grad():
                    if cpu:
                        with _cpu_only_torch():
                            holder["result"] = sep.separate(audio_path)
                    else:
                        holder["result"] = sep.separate(audio_path)
        except BaseException as e:
            holder["error"] = e
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True, name="uvr-sep")
    t.start()
    if not done.wait(timeout=timeout):
        raise TimeoutError(f"Separation timed out after {timeout}s.")
    if holder["error"] is not None:
        raise holder["error"]
    return holder["result"]


OUTPUT_FORMATS = ['wav', 'flac', 'mp3', 'ogg', 'opus', 'm4a', 'aiff', 'ac3']
output_format = OUTPUT_FORMATS


def _collect_outputs(output_dir, result, since_ts):
    files = []
    for f in (result or []):
        f = str(f)
        p = f if os.path.isabs(f) else os.path.join(output_dir, f)
        if os.path.isfile(p):
            files.append(os.path.abspath(p))
    if files:
        return files
    cands = []
    for ext in OUTPUT_FORMATS:
        for p in glob.glob(os.path.join(output_dir, f"*.{ext}")):
            try:
                if os.path.getmtime(p) >= (since_ts - 5):
                    cands.append(os.path.abspath(p))
            except OSError:
                pass
    return sorted(set(cands))


# ──────────────────────────────────────────────────────────────────────
#  隔離 worker（保留 raw string：r''' 是正確的）
# ──────────────────────────────────────────────────────────────────────
_WORKER_SRC = r'''
import sys, os, json, logging, traceback

MARK_STAGE = "###UVR_STAGE###"
MARK_RESULT = "###UVR_RESULT###"
MARK_ERROR = "###UVR_ERROR###"
EXIT_OOM = 42
EXIT_MODEL = 43

def emit(marker, payload):
    sys.stderr.write(marker + str(payload).replace("\n", " | ") + "\n")
    sys.stderr.flush()

def make_separator(Separator, common, optional, arch_kwargs):
    opt = dict(optional)
    while True:
        try:
            return Separator(**common, **opt, **arch_kwargs)
        except TypeError as e:
            if not opt:
                raise
            hit = None
            for k in list(opt):
                if k in str(e):
                    hit = k
                    break
            opt.pop(hit if hit else list(opt)[-1])

def _resolve_gpu_unlock_autocast(torch_runtime, arch, legacy_autocast=False,
                                 profile_enabled=False, autocast_explicit=False,
                                 force_cpu=False):
    if not profile_enabled or autocast_explicit:
        return bool(legacy_autocast)
    if force_cpu or sys.platform != "linux":
        return False
    if arch not in ("roformer", "mdx23c"):
        return False
    try:
        if not torch_runtime.cuda.is_available():
            return False
        device = torch_runtime.cuda.current_device()
        capability = tuple(torch_runtime.cuda.get_device_capability(device))
        return capability == (8, 6)
    except Exception:
        return False


def main():
    cfg = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    if cfg.get("force_cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if cfg.get("alloc_conf"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cfg["alloc_conf"])

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["DISABLE_ADDMM_CUDA_LT"] = "1"

    emit(MARK_STAGE, "importing runtime")
    import torch
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cuda.preferred_blas_library = "cublas"
    except (AttributeError, TypeError):
        pass

    from audio_separator.separator import Separator
    if not cfg.get("force_cpu"):
        emit(MARK_STAGE, "cuda available=%s" % torch.cuda.is_available())
    common = dict(
        log_level=logging.WARNING,
        model_file_dir=cfg["model_file_dir"],
        output_dir=cfg["output_dir"],
        output_format=cfg["output_format"],
        normalization_threshold=cfg["normalization_threshold"],
        output_single_stem=cfg["output_single_stem"],
    )
    arch = cfg["arch"]
    effective_autocast = _resolve_gpu_unlock_autocast(
        torch, arch, cfg.get("use_autocast", False),
        cfg.get("gpu_unlock_profile", False),
        cfg.get("autocast_explicit", False), cfg.get("force_cpu", False))
    if cfg.get("gpu_unlock_profile", False) and effective_autocast:
        emit(MARK_STAGE, "gpu-unlock AMP profile: use_autocast=%s; "
             "kernel unlock must be verified separately" % effective_autocast)
    optional = {"use_autocast": effective_autocast,
                "sample_rate": cfg.get("sample_rate", 44100)}
    p = dict(cfg["params"])
    if arch in ("roformer", "mdx23c"):
        p.setdefault("pitch_shift", 0)
        kw = {"mdxc_params": p}
    elif arch == "mdxnet":
        kw = {"mdx_params": p}
    elif arch == "vrarch":
        kw = {"vr_params": p}
    elif arch == "demucs":
        common["output_single_stem"] = None
        kw = {"demucs_params": p}
    else:
        kw = {}
    emit(MARK_STAGE, "building separator")
    sep = make_separator(Separator, common, optional, kw)
    emit(MARK_STAGE, "loading model")
    try:
        sep.load_model(model_filename=cfg["model_filename"])
    except BaseException as e:
        txt = "%s: %s" % (type(e).__name__, e)
        emit(MARK_ERROR, txt)
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        low = txt.lower()
        if "out of memory" in low:
            sys.exit(EXIT_OOM)
        sys.exit(EXIT_MODEL)
    emit(MARK_STAGE, "separating")
    try:
        with torch.no_grad():
            files = sep.separate(cfg["audio_path"])
    except BaseException as e:
        txt = "%s: %s" % (type(e).__name__, e)
        emit(MARK_ERROR, txt)
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        low = txt.lower()
        if "out of memory" in low and "illegal" not in low:
            sys.exit(EXIT_OOM)
        sys.exit(1)
    out = []
    for f in files or []:
        f = str(f)
        out.append(f if os.path.isabs(f) else os.path.join(cfg["output_dir"], f))
    emit(MARK_STAGE, "writing results")
    sys.stdout.write(MARK_RESULT + json.dumps(out) + "\n")
    sys.stdout.flush()

try:
    main()
except SystemExit:
    raise
except BaseException as e:
    emit(MARK_ERROR, "%s: %s" % (type(e).__name__, e))
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    sys.exit(1)
'''

_WORKER_PATH = os.path.join(TEMP_ROOT, "_uvr_worker.py")
try:
    with open(_WORKER_PATH, "w", encoding="utf-8") as _f:
        _f.write(_WORKER_SRC)
except OSError as _e:
    print(f"[warn] could not write worker script ({_e}); subprocess fallback disabled.")
    SUBPROC_FALLBACK = False
    ALWAYS_SUBPROCESS = False

EXIT_OOM = 42
EXIT_MODEL = 43
_STAGE_PCT = {
    "importing runtime": 0.15,
    "building separator": 0.20,
    "loading model": 0.25,
    "separating": 0.35,
    "writing results": 0.93,
}


def _explain_failure(text, returncode=None, timed_out=False):
    low = (text or "").lower()
    if timed_out:
        return ("The worker exceeded UVR_SUBPROC_TIMEOUT and was killed. "
                "Very long files need a higher timeout or a smaller segment size.")
    if returncode == EXIT_OOM or "out of memory" in low:
        return ("Out of memory. The next attempt uses a smaller segment size. "
                "Keep batch size at 1 and close other GPU applications.")
    if returncode in (-9, 137):
        return ("The worker was killed by the OS (SIGKILL) — this is almost always "
                "system RAM exhaustion, not VRAM. Use a shorter file or add swap.")
    if returncode == EXIT_MODEL or "load_model" in low or "checkpoint" in low:
        return ("The model could not be loaded. It may be a partial/corrupt download — "
                f"delete it from {models_dir} and let the app fetch it again.")
    if "illegal memory access" in low or "cublas" in low:
        return ("CUDA fault (CUBLAS / illegal memory access). Usually a torch build that "
                "does not match the driver's CUDA runtime, an unstable/overclocked GPU, "
                "or a segment size that is too large. Retrying smaller.")
    if "no kernel image" in low:
        return ("Your PyTorch build has no kernels for this GPU architecture — "
                "reinstall torch for your CUDA version.")
    if "connection" in low or "timed out" in low or "http" in low:
        return "Network problem while downloading the model. Check connectivity and retry."
    if "no space left" in low:
        return "Disk is full — free space in the models/ and temp_outputs/ folders."
    return ""


def _run_subprocess_sep(arch, model_fn, audio_path, output_dir, out_format,
                        norm_thresh, single_stem, kw, force_cpu=False,
                        reporter=None, timeout=None):
    cfg = {
        "arch": arch,
        "model_filename": model_fn,
        "model_file_dir": models_dir,
        "audio_path": os.path.abspath(audio_path),
        "output_dir": os.path.abspath(output_dir),
        "output_format": out_format,
        "normalization_threshold": float(norm_thresh),
        "output_single_stem": single_stem,
        "params": kw,
        "force_cpu": bool(force_cpu),
        "use_autocast": USE_AUTOCAST,
        "gpu_unlock_profile": GPU_UNLOCK_PROFILE,
        "autocast_explicit": AUTOCAST_EXPLICIT,
        "sample_rate": TARGET_SR,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    try:
        os.makedirs(cfg["output_dir"], exist_ok=True)
    except OSError:
        pass
    if GPU_UNLOCK_PROFILE and reporter and not AUTOCAST_EXPLICIT:
        reporter.log_add(
            f"[gpu-unlock] worker 請求 AMP 設定檔：arch={arch} "
            f"（kernel 解鎖需另外確認）")
    fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="uvrcfg_")
    result_holder = {"files": None}
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["DISABLE_ADDMM_CUDA_LT"] = "1"
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        base = 0.35
        span = 0.55

        def on_stdout(line):
            if line.startswith("###UVR_RESULT###"):
                try:
                    result_holder["files"] = json.loads(line[len("###UVR_RESULT###"):])
                except Exception:
                    pass
            elif reporter:
                reporter.log_add(line)

        def on_stderr(line):
            if not reporter:
                return
            reporter.panels_visible = True
            if line.startswith("###UVR_STAGE###"):
                st = line[len("###UVR_STAGE###"):].strip()
                reporter.stage_set(st, _STAGE_PCT.get(st))
                return
            if line.startswith("###UVR_ERROR###"):
                reporter.log_add("‼ " + line[len("###UVR_ERROR###"):].strip())
                return
            m = _PCT_RE.search(line)
            if m:
                try:
                    pct = int(m.group(1))
                    if 0 <= pct <= 100:
                        reporter.pct_set(base + span * (pct / 100.0))
                except Exception:
                    pass
            reporter.log_add(line)

        rc, _out, err, timed_out = stream_process(
            [sys.executable, _WORKER_PATH, cfg_path],
            timeout=timeout or SUBPROC_TIMEOUT,
            on_stdout=on_stdout, on_stderr=on_stderr, env=env)
        files = [f for f in (result_holder["files"] or []) if os.path.isfile(f)]
        if rc == 0 and files:
            return [os.path.abspath(f) for f in files]
        tail = "\n".join(l for l in err
                         if not l.startswith("###UVR_STAGE###"))[-2500:]
        hint = _explain_failure(tail, rc, timed_out)
        if timed_out:
            note = "timed out"
        elif rc == 0:
            note = "finished but produced no output files (input could not be read)"
        else:
            note = f"exit code {rc}"
        raise RuntimeError(f"isolated worker {note}.\n{tail}"
                           + (f"\n➤ {hint}" if hint else ""))
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


CUBLAS_HINT = (
    "\nCUDA error (CUBLAS / illegal memory access) usually means:\n"
    "  • segment size too large for the free VRAM → lower it (256 → 128 → 64);\n"
    "  • a PyTorch build compiled against a different CUDA runtime than the driver "
    "→ reinstall torch matching the CUDA version reported by nvidia-smi;\n"
    "  • an unstable/overclocked or thermally throttled GPU."
)

# ──────────────────────────────────────────────────────────────────────
#  嘗試規劃 + 重試引擎
# ──────────────────────────────────────────────────────────────────────
def _norm_single_stem(s):
    if s is None:
        return None
    s = str(s).strip()
    return s or None


def _seg_floor(arch):
    return 5 if arch == "demucs" else 32


def _scale_kw(arch, kw, seg_div=1, batch_cap=None):
    out = dict(kw)
    seg = out.get("segment_size")
    if seg_div > 1 and isinstance(seg, (int, float)):
        floor = _seg_floor(arch)
        newv = max(floor, int(int(seg) // seg_div))
        if arch != "demucs":
            newv = max(floor, (newv // 32) * 32 or floor)
        out["segment_size"] = newv
    if batch_cap is not None and isinstance(out.get("batch_size"), (int, float)):
        out["batch_size"] = max(1, min(int(out["batch_size"]), int(batch_cap)))
    return out


# [REV-INTEGRATED] 區分「模型預設 segment」與「使用者顯式 override」：
#   - 自動 clamp（override_model_segment_size=False）仍保持 512 上限；
#   - 使用者顯式 override（override_model_segment_size=True）時尊重其選擇，
#     只給警告而不默默降級，讓想吃盡 RTX 3060 12GB 的人可以手動開 768。
def _vram_sane_kw(arch, kw):
    if not gpu_configured() or TOTAL_VRAM_GB <= 0:
        return dict(kw), None
    if arch not in ("roformer", "mdx23c"):
        out = dict(kw)
        if isinstance(out.get("batch_size"), (int, float)) and TOTAL_VRAM_GB < 8:
            out["batch_size"] = 1
        return out, None
    out = dict(kw)
    seg = out.get("segment_size")
    note = None
    user_override = bool(out.get("override_model_segment_size", False))
    if isinstance(seg, (int, float)):
        cap = None
        if TOTAL_VRAM_GB < 6:
            cap = 128
        elif TOTAL_VRAM_GB < 8:
            cap = 256
        elif TOTAL_VRAM_GB < 12:
            cap = 512
        elif TOTAL_VRAM_GB < 16:
            cap = 512
        if cap and int(seg) > cap:
            if user_override:
                note = (f"segment size {int(seg)} exceeds the {cap}-safety cap for "
                        f"a {TOTAL_VRAM_GB:.0f} GB GPU; honoring the explicit override — "
                        f"expect possible OOM")
            else:
                out["segment_size"] = cap
                note = (f"segment size {int(seg)} clamped to {cap} for a "
                        f"{TOTAL_VRAM_GB:.0f} GB GPU")
    if isinstance(out.get("batch_size"), (int, float)) and TOTAL_VRAM_GB < 8:
        out["batch_size"] = 1
    return out, note


def _plan_attempts(arch, kw):
    plan = []
    if FORCE_CPU or not GPU_PRESENT:
        if SUBPROC_FALLBACK:
            plan.append(("subcpu", _scale_kw(arch, kw, 1, 2), 0.0))
            plan.append(("subcpu", _scale_kw(arch, kw, 2, 1), 0.5))
        else:
            plan.append(("cpu", _scale_kw(arch, kw, 1, 2), 0.0))
            plan.append(("cpu", _scale_kw(arch, kw, 2, 1), 0.5))
        return plan
    if ALWAYS_SUBPROCESS and SUBPROC_FALLBACK:
        plan += [("subgpu", dict(kw), 0.0),
                 ("subgpu", _scale_kw(arch, kw, 2, 1), 1.5),
                 ("subgpu", _scale_kw(arch, kw, 4, 1), 1.5)]
    elif gpu_context_alive():
        plan += [("gpu", dict(kw), 0.0),
                 ("gpu", _scale_kw(arch, kw, 2, 1), 1.5)]
        if SUBPROC_FALLBACK:
            plan += [("subgpu", _scale_kw(arch, kw, 2, 1), 1.0),
                     ("subgpu", _scale_kw(arch, kw, 4, 1), 1.0)]
    else:
        if SUBPROC_FALLBACK:
            plan += [("subgpu", dict(kw), 0.0),
                     ("subgpu", _scale_kw(arch, kw, 2, 1), 1.0),
                     ("subgpu", _scale_kw(arch, kw, 4, 1), 1.0)]
        else:
            plan.append(("gpu", _scale_kw(arch, kw, 2, 1), 1.0))
    if ALLOW_CPU_FALLBACK:
        if SUBPROC_FALLBACK:
            plan.append(("subcpu", _scale_kw(arch, kw, 2, 1), 1.0))
        else:
            plan.append(("cpu", _scale_kw(arch, kw, 2, 1), 1.0))
    return plan


_MODE_LABEL = {"gpu": "in-process GPU", "subgpu": "isolated GPU worker",
               "subcpu": "isolated CPU worker", "cpu": "in-process CPU"}


def _attempt_sig(mode, kw):
    return (mode, kw.get("segment_size"), kw.get("batch_size"),
            kw.get("window_size"), kw.get("shifts"))


def _disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return None


def _stage_input_audio(audio_path, reporter=None):
    if not audio_path or not os.path.isfile(audio_path):
        return audio_path
    try:
        audio_path.encode("ascii")
        return audio_path
    except UnicodeEncodeError:
        pass
    try:
        ext = os.path.splitext(audio_path)[1].lower() or ".wav"
        stage_dir = get_temp_output_dir()
        staged = os.path.join(stage_dir, "input_" + uuid.uuid4().hex[:10] + ext)
        shutil.copy2(audio_path, staged)
        if reporter:
            reporter.log_add("staged input to an ASCII-safe temp path")
        return staged
    except Exception as e:
        if reporter:
            reporter.warn(f"could not stage input ({e}); using original path")
        return audio_path


def _separate_with_retry(arch, model_fn, audio_path, output_dir, out_format,
                         norm_thresh, single_stem, arch_kw,
                         reporter=None, desc="Separation"):
    if not audio_path or not os.path.isfile(audio_path):
        raise RuntimeError(f"{desc}: input audio file not found ({audio_path}).")
    audio_path = _stage_input_audio(audio_path, reporter)
    single_stem = _norm_single_stem(single_stem)
    os.makedirs(output_dir, exist_ok=True)
    free_disk = _disk_free_gb(output_dir)
    if free_disk is not None and free_disk < 2.0 and reporter:
        reporter.warn(f"Low disk space in output dir: {free_disk:.2f} GB free")
    safe_kw, note = _vram_sane_kw(arch, arch_kw)
    if note and reporter:
        reporter.warn(note)
    attempts = _plan_attempts(arch, safe_kw)
    seen, last_err, errors = set(), None, []
    for idx, (mode, kw, pause) in enumerate(attempts):
        sig = _attempt_sig(mode, kw)
        if sig in seen:
            continue
        seen.add(sig)
        if mode == "gpu" and not gpu_configured():
            continue
        if pause:
            _deadline = time.time() + pause
            while time.time() < _deadline:
                time.sleep(min(0.5, max(0.0, _deadline - time.time())))
        free_gpu(deep=True)
        if mode == "gpu" and not gpu_healthy():
            if reporter:
                reporter.warn("in-process CUDA context unusable — using isolated worker")
            continue
        if mode in ("gpu", "subgpu"):
            wait_for_vram(min(2.0, TOTAL_VRAM_GB * 0.25), timeout=30, reporter=reporter)
        label = _MODE_LABEL[mode]
        seg_txt = kw.get("segment_size", "n/a")
        bs_txt = kw.get("batch_size", "n/a")
        if reporter:
            reporter.stage_set(
                f"attempt {idx + 1}: {label}",
                0.10,
                detail=f"segment={seg_txt}, batch={bs_txt}")
        started = time.time()
        try:
            if mode in ("subgpu", "subcpu"):
                files = _run_subprocess_sep(arch, model_fn, audio_path, output_dir,
                                            out_format, norm_thresh, single_stem, kw,
                                            force_cpu=(mode == "subcpu"),
                                            reporter=reporter)
            else:
                cpu = (mode == "cpu")
                if reporter:
                    reporter.stage_set("loading model", 0.22)
                sep, cached = _acquire_separator(arch, model_fn, output_dir, out_format,
                                                 norm_thresh, single_stem, kw, cpu=cpu)
                if reporter:
                    reporter.stage_set(
                        "separating" + (" (cached model)" if cached else ""), 0.38)
                result = _run_sep_with_timeout(sep, audio_path, cpu, INPROC_TIMEOUT)
                files = _collect_outputs(output_dir, result, started)
            if not files:
                MODEL_CACHE.clear()
                raise RuntimeError("the model produced no output files")
            with MODEL_CACHE.lock:
                MODEL_CACHE.last_used = time.time()
            if reporter:
                reporter.stage_set("checking sample rate", 0.95)
            files = ensure_min_samplerate_many(files, reporter)
            if reporter:
                reporter.stage_set("done", 0.99)
            return files
        except (TimeoutError, concurrent.futures.TimeoutError) as te:
            last_err = te
            errors.append(f"[{label}] timeout: {te}")
            MODEL_CACHE.clear()
            if mode == "gpu":
                mark_context_dead(str(te))
            if reporter:
                reporter.warn(f"{label} timed out — trying the next strategy")
            continue
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            last_err = e
            txt = str(e)
            errors.append(f"[{label}] {txt[:900]}")
            if mode in ("gpu", "cpu"):
                MODEL_CACHE.clear()
                if _is_oom(e):
                    note_oom(txt)
                if mode == "cpu":
                    break
                if mode == "gpu":
                    if _is_cuda_err(e):
                        mark_context_dead(txt)
                        if reporter:
                            reporter.warn("CUDA fault in-process — switching to isolated worker")
                    else:
                        if reporter:
                            reporter.warn(f"non-CUDA failure: {txt[:200]}")
            else:
                if "produced no output" in txt.lower():
                    break
                if reporter:
                    hint = _explain_failure(txt)
                    reporter.warn((hint or txt)[:300])
            continue
        finally:
            free_gpu(deep=True)
    detail = "\n".join(errors[-3:])
    msg = f"{desc} failed after {len(seen)} attempt(s).\n{detail}"
    low = str(last_err).lower()
    if "cublas" in low or "illegal memory access" in low:
        msg += CUBLAS_HINT
    elif "out of memory" in low:
        msg += ("\nOut of memory even at the smallest segment size. Try a shorter file, "
                "a lighter model, or set UVR_FORCE_CPU=1.")
    if reporter:
        reporter.fail(msg)
    raise RuntimeError(msg)


def _start_idle_reaper():
    def _loop():
        while True:
            time.sleep(60)
            try:
                MODEL_CACHE.reap_if_idle(MODEL_IDLE_TIMEOUT)
            except Exception:
                pass

    threading.Thread(target=_loop, daemon=True, name="vram-reaper").start()


_start_idle_reaper()


# ──────────────────────────────────────────────────────────────────────
#  優雅關閉
# ──────────────────────────────────────────────────────────────────────
def _graceful_shutdown(*_args):
    try:
        MODEL_CACHE.clear(quiet=True)
    except Exception:
        pass
    try:
        free_gpu(deep=True)
    except Exception:
        pass


atexit.register(_graceful_shutdown)


def _install_signal_handlers():
    def _handler(signum, frame):
        _graceful_shutdown()
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except Exception:
            pass

    for sig in (signal.SIGTERM,):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


_install_signal_handlers()


# ──────────────────────────────────────────────────────────────────────
#  暫存管理
# ──────────────────────────────────────────────────────────────────────
CLEANUP_DELAY = int(os.environ.get("UVR_TEMP_TTL", "10800"))
_active_folders = set()
_active_folders_lock = threading.Lock()


def _mark_active(folder):
    if not folder:
        return
    with _active_folders_lock:
        _active_folders.add(os.path.abspath(folder))


def _unmark_active(folder):
    if not folder:
        return
    with _active_folders_lock:
        _active_folders.discard(os.path.abspath(folder))
    try:
        os.utime(folder, None)
    except OSError:
        pass


def cleanup_old_outputs():
    now = time.time()
    try:
        entries = os.listdir(TEMP_ROOT)
    except Exception:
        return
    with _active_folders_lock:
        snap = set(_active_folders)
    for f in entries:
        fp = os.path.abspath(os.path.join(TEMP_ROOT, f))
        if fp in snap or not os.path.isdir(fp):
            continue
        try:
            newest = os.path.getmtime(fp)
            for root, _d, files in os.walk(fp):
                for name in files:
                    try:
                        newest = max(newest, os.path.getmtime(os.path.join(root, name)))
                    except OSError:
                        pass
            if now - newest > CLEANUP_DELAY:
                shutil.rmtree(fp, ignore_errors=True)
        except Exception:
            pass


def start_cleanup_thread():
    def _loop():
        while True:
            try:
                cleanup_old_outputs()
            except Exception:
                pass
            time.sleep(120)

    threading.Thread(target=_loop, daemon=True, name="temp-janitor").start()


start_cleanup_thread()


def get_temp_output_dir():
    folder = os.path.join(TEMP_ROOT, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    return folder


# ──────────────────────────────────────────────────────────────────────
#  媒體輸入處理
# ──────────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.flv', '.wmv',
                    '.m4v', '.ts', '.mpg', '.mpeg', '.3gp', '.ogv')
AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a',
                    '.aiff', '.aif', '.ac3', '.aac', '.wma')
ALL_MEDIA_EXTENSIONS = list(VIDEO_EXTENSIONS) + list(AUDIO_EXTENSIONS)
FFMPEG_TIMEOUT = 60 * 45


def extract_audio_lossless(src, output_dir):
    if not FFMPEG_OK:
        raise gr.Error("ffmpeg not found — required to extract audio from video.")
    base = os.path.splitext(os.path.basename(src))[0]
    safe = re.sub(r"[^\w.-]+", "_", base)[:120] or "media"
    out = os.path.join(output_dir, f"{safe}_audio.flac")
    src_sr = probe_sr(src) or TARGET_SR
    target_sr = max(TARGET_SR, int(src_sr))
    attempts = [
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vn", "-map", "0:a:0", "-af", "aresample=resampler=soxr:precision=28",
         "-ar", str(target_sr), "-c:a", "flac", "-compression_level", "5", out],
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vn", "-ar", str(target_sr), "-c:a", "flac", out],
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-vn", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
         os.path.join(output_dir, f"{safe}_audio.wav")],
    ]
    last_err = ""
    for cmd in attempts:
        try:
            r = run_cmd(cmd, timeout=FFMPEG_TIMEOUT)
        except Exception as e:
            last_err = str(e)
            continue
        produced = cmd[-1]
        if r.returncode == 0 and os.path.isfile(produced) and os.path.getsize(produced) > 0:
            return ensure_min_samplerate(produced)
        last_err = (r.stderr or "")[-800:]
    raise gr.Error(f"Could not extract audio from the file.\n{last_err or 'unknown ffmpeg error'}")


def ensure_wav(input_path, reporter=None):
    if not input_path or not os.path.isfile(input_path):
        return input_path
    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return extract_audio_lossless(input_path, get_temp_output_dir())
    sr = probe_sr(input_path)
    if sr is not None and sr < TARGET_SR and FFMPEG_OK:
        out_dir = get_temp_output_dir()
        safe = re.sub(r"[^\w.-]+", "_",
                      os.path.splitext(os.path.basename(input_path))[0])[:120] or "audio"
        out = os.path.join(out_dir, f"{safe}_{TARGET_SR}.flac")
        r = run_cmd(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", input_path, "-vn",
                     "-af", "aresample=resampler=soxr:precision=28",
                     "-ar", str(TARGET_SR), "-c:a", "flac", out], timeout=FFMPEG_TIMEOUT)
        if r.returncode == 0 and os.path.isfile(out):
            if reporter:
                reporter.log_add(f"input upsampled {sr} Hz → {TARGET_SR} Hz")
            return out
    return input_path


def _cleanup_temp_wav(processing_path, original_path):
    if processing_path and processing_path != original_path and os.path.exists(processing_path):
        try:
            os.remove(processing_path)
        except OSError:
            pass


def _as_path(file_obj):
    if not file_obj:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, (list, tuple)) and file_obj:
        return _as_path(file_obj[0])
    return getattr(file_obj, "name", None)


def _describe_audio(path):
    try:
        info = sf.info(path)
        dur = info.frames / float(info.samplerate or 1)
        mins, secs = divmod(int(dur), 60)
        return f"{info.samplerate} Hz · {info.channels} ch · {mins}m{secs:02d}s"
    except Exception:
        sr = probe_sr(path)
        return f"{sr} Hz" if sr else "unknown format"


def smart_media_ingest(file_obj):
    fp = _as_path(file_obj)
    if not fp:
        return None, None, "No input loaded."
    if not os.path.isfile(fp):
        return None, None, "❌ File not found."
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext in VIDEO_EXTENSIONS:
            out = extract_audio_lossless(fp, get_temp_output_dir())
            return out, out, (f"🎬 Video decoded → `{os.path.basename(out)}` "
                              f"({_describe_audio(out)})")
        if ext in AUDIO_EXTENSIONS:
            src_sr = probe_sr(fp)
            if src_sr is not None and src_sr < TARGET_SR:
                out = ensure_wav(fp)
            else:
                out = fp
            return out, out, f"🎵 Audio loaded: `{os.path.basename(out)}` ({_describe_audio(out)})"
        if FFMPEG_OK:
            out = extract_audio_lossless(fp, get_temp_output_dir())
            return out, out, f"📦 Container decoded → `{os.path.basename(out)}` ({_describe_audio(out)})"
        return None, None, f"❌ Unsupported file type: {ext}"
    except gr.Error as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        return None, None, f"❌ Could not load input: {e}"


# ──────────────────────────────────────────────────────────────────────
#  URL 下載
# ──────────────────────────────────────────────────────────────────────
YTDLP_TIMEOUT = 3600
_META_SEP = "|||UVR5META|||"
_PRINT_FMT = f"after_move:%(title)s{_META_SEP}%(id)s{_META_SEP}%(filepath)s"


def _validate_url(url):
    if not url or not str(url).strip():
        raise gr.Error("Please enter a URL first.")
    url = str(url).strip()
    p = urllib.parse.urlparse(url)
    if p.scheme.lower() not in ("http", "https"):
        raise gr.Error("Only http/https links are supported.")
    if not p.hostname:
        raise gr.Error("Invalid URL.")
    return url


def _cookies_usable():
    if not os.path.isfile(cookies_file):
        return False
    try:
        if os.path.getsize(cookies_file) < 20:
            return False
        with open(cookies_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#") and not s.startswith("#HttpOnly_"):
                    continue
                if s.count("\t") >= 6:
                    return True
    except OSError:
        pass
    return False


def _ytdlp_commands():
    cmds, seen = [], set()
    for exe in (shutil.which("yt-dlp"), shutil.which("yt-dlp.exe")):
        if exe and exe not in seen:
            cmds.append([exe])
            seen.add(exe)
    if yt_dlp is not None:
        cmds.append([sys.executable, "-m", "yt_dlp"])
    return cmds


def _parse_meta(lines):
    for line in reversed(lines):
        line = line.strip()
        if _META_SEP not in line:
            continue
        try:
            title, vid, fp = line.rsplit(_META_SEP, 2)
        except ValueError:
            continue
        fp = fp.strip().strip('"').strip("'")
        if fp:
            return title, vid.strip(), fp
    return None


def _find_audio(output_dir, stdout_lines):
    meta = _parse_meta(stdout_lines)
    if meta:
        _, _, fp = meta
        cands = [fp] if os.path.isabs(fp) else [fp, os.path.join(output_dir, fp)]
        for c in cands:
            c = os.path.abspath(c)
            if os.path.isfile(c) and os.path.splitext(c)[1].lower() in AUDIO_EXTENSIONS:
                return c
    newest = None
    for ext in AUDIO_EXTENSIONS:
        for f in glob.glob(os.path.join(output_dir, f"*{ext}")):
            if newest is None or os.path.getmtime(f) > os.path.getmtime(newest):
                newest = f
    return os.path.abspath(newest) if newest else None


def _find_js_runtimes():
    args = []
    node_path = shutil.which("node") or shutil.which("nodejs")
    if node_path:
        args.extend(["--js-runtimes", f"node:{node_path}"])
    deno_path = shutil.which("deno")
    if deno_path:
        args.extend(["--js-runtimes", f"deno:{deno_path}"])
    bun_path = shutil.which("bun")
    if bun_path:
        args.extend(["--js-runtimes", f"bun:{bun_path}"])
    qjs_path = shutil.which("qjs") or shutil.which("quickjs")
    if qjs_path:
        args.extend(["--js-runtimes", f"quickjs:{qjs_path}"])
    return args


def _build_dl_cmd(base, url, out_dir, cookies=False):
    cmd = list(base) + [
        "--ignore-config", "--no-playlist", "--newline", "--no-mtime",
        "--retries", "10", "--fragment-retries", "10", "--extractor-retries", "5",
        "--socket-timeout", "30",
        "--js-runtimes", "nodejs,node,deno,bun,quickjs",
        "--extractor-args", "youtube:player_client=android,web",
        "--remote-components", "ejs:github",
        "-f", "bestaudio/best", "-x", "--audio-format", "flac", "--audio-quality", "0",
        "--paths", os.path.abspath(out_dir), "-o", "ytdlp_raw_%(id)s.%(ext)s",
        "--print", _PRINT_FMT,
    ]
    if IS_WINDOWS:
        cmd.append("--windows-filenames")
    if cookies and _cookies_usable():
        cmd += ["--cookies", os.path.abspath(cookies_file)]
    cmd += ["--", url]
    return cmd


def _sanitize_stem(title, max_chars=120, fallback="audio"):
    if not title:
        return fallback
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', title).strip()
    if not s:
        return fallback
    if len(s) > max_chars:
        t = s[:max_chars]
        sp = max(t.rfind(' '), t.rfind('-'), t.rfind('_'))
        s = t[:sp] if sp > max_chars // 2 else t
    return s


def _rename_readable(src, title, vid):
    try:
        src = os.path.abspath(src)
        if not os.path.isfile(src):
            return src
        d, ext = os.path.dirname(src), os.path.splitext(src)[1] or ".flac"
        stem = _sanitize_stem(title, fallback=vid or "audio")
        base = f"{stem} [{vid}]" if vid else stem
        cand = os.path.join(d, f"{base}{ext}")
        n = 1
        while os.path.exists(cand) and os.path.abspath(cand) != src:
            cand = os.path.join(d, f"{base} ({n}){ext}")
            n += 1
        if n > 500:
            return src
        if os.path.abspath(cand) != src:
            os.replace(src, cand)
        return cand
    except OSError:
        return src


def _ytdlp_hint(msg):
    lo = (msg or "").lower()
    if "sign in" in lo or "not a bot" in lo:
        return f"\nThe site wants authentication. Export browser cookies to '{cookies_file}'."
    if "po token" in lo:
        return "\nYouTube wants a PO Token. Update yt-dlp (pip install -U yt-dlp)."
    if any(k in lo for k in ("failed to extract", "player response", "signature", "nsig")):
        return "\nyt-dlp is probably outdated: pip install -U yt-dlp"
    if "age" in lo and "restrict" in lo:
        return "\nAge-restricted video. Cookies may help."
    if "private" in lo or "members-only" in lo:
        return "\nPrivate or members-only video."
    if "unavailable" in lo or "copyright" in lo:
        return "\nMedia removed or region-blocked."
    if "drm" in lo:
        return "\nDRM-protected media cannot be downloaded."
    if "ffmpeg" in lo or "ffprobe" in lo:
        return "\nInstall ffmpeg and ffprobe."
    return ""


_DL_PCT = re.compile(r"\[download\]\s+(\d{1,3}(?:\.\d+)?)%")


def download_audio(url, output_dir=None, progress=None):
    url = _validate_url(url)
    if not FFMPEG_OK:
        raise gr.Error("ffmpeg not found. Install ffmpeg for URL downloads.")
    commands = _ytdlp_commands()
    if not commands:
        raise gr.Error("yt-dlp not found. Install: pip install -U yt-dlp")
    out = os.path.abspath(output_dir) if output_dir else get_temp_output_dir()
    os.makedirs(out, exist_ok=True)
    _mark_active(out)
    cookie_modes = [False] + ([True] if _cookies_usable() else [])
    errors = []
    try:
        for base in commands:
            for use_ck in cookie_modes:
                lbl = os.path.basename(base[0]) if base[0] != sys.executable else "python -m yt_dlp"
                ck_lbl = "cookies" if use_ck else "no cookies"
                state = {"pct": 0.0}

                def on_line(line):
                    m = _DL_PCT.search(line)
                    if m and progress is not None:
                        try:
                            state["pct"] = float(m.group(1)) / 100.0
                            progress(min(0.95, 0.05 + 0.85 * state["pct"]),
                                     desc=f"Downloading {int(state['pct']*100)}% ({lbl})")
                        except Exception:
                            pass

                if progress is not None:
                    progress(0.03, desc=f"Starting download ({lbl}, {ck_lbl})…")
                cmd = _build_dl_cmd(base, url, out, use_ck)
                try:
                    rc, sout, serr, timed_out = stream_process(
                        cmd, timeout=YTDLP_TIMEOUT, on_stdout=on_line, on_stderr=on_line)
                except FileNotFoundError:
                    errors.append(f"{lbl}: executable not found.")
                    continue
                except Exception as e:
                    errors.append(f"{lbl} ({ck_lbl}): {e}")
                    continue
                if timed_out:
                    errors.append(f"{lbl} ({ck_lbl}): timed out.")
                    continue
                final = _find_audio(out, sout)
                if rc == 0 and final:
                    meta = _parse_meta(sout)
                    if meta:
                        title, vid = meta[0], meta[1]
                    else:
                        title, vid = os.path.splitext(os.path.basename(final))[0], ""
                    final = _rename_readable(final, title, vid)
                    if progress is not None:
                        progress(0.97, desc="Checking sample rate…")
                    final = ensure_min_samplerate(final)
                    if progress is not None:
                        progress(1.0, desc="Download complete")
                    return final
                diag = "\n".join(serr[-40:]) or f"exit {rc}"
                errors.append(f"{lbl} ({ck_lbl}):\n{diag[-3000:]}")
        combined = "\n".join(errors)[-6000:]
        raise gr.Error(f"Download failed.\n{combined}{_ytdlp_hint(combined)}")
    finally:
        _unmark_active(out)


def download_audio_to_slot(url, progress=gr.Progress()):
    progress = _cli_progress(progress)
    try:
        path = download_audio(url, progress=progress)
    except gr.Error as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        return None, None, f"❌ Download failed: {e}"
    return path, path, (f"🔗 Downloaded `{os.path.basename(path)}` "
                        f"({_describe_audio(path)})")


# ──────────────────────────────────────────────────────────────────────
#  變調
# ──────────────────────────────────────────────────────────────────────
def _format_pitch_steps(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0"
    sign = "+" if n >= 0 else "-"
    a = abs(n)
    if abs(a - round(a)) < 1e-6:
        return f"{sign}{int(round(a))}"
    return f"{sign}{a:.2f}".rstrip("0").rstrip(".")


def pitch_shift(input_file, n_steps, formant_preserve=True, pitch_quality=5,
                progress=gr.Progress()):
    progress = _cli_progress(progress)
    if not input_file:
        raise gr.Error("Please select an audio file first.")
    if not os.path.isfile(input_file):
        raise gr.Error(f"File not found: {input_file}")
    if not _check_rubberband_cli():
        raise gr.Error("Rubber Band CLI not installed (apt install rubberband-cli).")
    try:
        n_f = float(n_steps)
    except (TypeError, ValueError):
        raise gr.Error(f"Invalid pitch value: {n_steps}")
    progress(0.05, desc="Preparing input…")
    processing_path = ensure_wav(input_file)
    is_temp = (processing_path != input_file)
    tmp_in = tmp_out = None
    out_dir = None
    y = y_s = None
    try:
        y, sr = librosa.load(processing_path, sr=None, mono=False)
        if sr < TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
            sr = TARGET_SR
        y = y.reshape(-1, 1) if y.ndim == 1 else y.T
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp_in = t.name
        sf.write(tmp_in, y, sr, subtype="FLOAT")
        del y
        y = None
        gc.collect()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
            tmp_out = t.name
        cmd = ["rubberband", "--pitch", f"{n_f:.6f}"]
        if pitch_quality >= 4:
            cmd += ["--pitch-hq", "--fine"]
        elif pitch_quality == 3:
            cmd.append("--pitch-hq")
        if formant_preserve:
            cmd.append("--formant")
        cmd += [tmp_in, tmp_out]
        progress(0.35, desc="Rubber Band processing…")
        try:
            with heartbeat_progress(progress, 3, "Rubber Band processing…", 0.6):
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            raise gr.Error(f"Pitch shift failed: {e.stderr or e.stdout}")
        except subprocess.TimeoutExpired:
            raise gr.Error("Pitch shift timed out.")
        except FileNotFoundError:
            raise gr.Error("Rubber Band CLI not found.")
        progress(0.9, desc="Writing output…")
        y_s, _ = sf.read(tmp_out, always_2d=False)
        stem = os.path.splitext(os.path.basename(input_file))[0]
        out_dir = get_temp_output_dir()
        _mark_active(out_dir)
        out_path = os.path.join(out_dir, f"{stem}_pitch_{_format_pitch_steps(n_f)}.wav")
        sf.write(out_path, y_s, sr, subtype="PCM_24")
        ensure_min_samplerate(out_path)
        progress(1.0, desc="Done")
        return out_path
    finally:
        del y, y_s
        gc.collect()
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if is_temp:
            _cleanup_temp_wav(processing_path, input_file)
        _unmark_active(out_dir)


# ──────────────────────────────────────────────────────────────────────
#  模型目錄
# ──────────────────────────────────────────────────────────────────────
roformer_models = {
    'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'MelBand Roformer | Karaoke by Gabox': 'mel_band_roformer_karaoke_gabox.ckpt',
    'MelBand Roformer | Karaoke V2 by Gabox': 'mel_band_roformer_karaoke_gabox_v2.ckpt',
    'MelBand Roformer | Karaoke by becruily': 'mel_band_roformer_karaoke_becruily.ckpt',
    'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1053': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'Mel-Roformer-Viperx-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'MelBand Roformer | Denoise-Debleed by Gabox': 'mel_band_roformer_denoise_debleed_gabox.ckpt',
    'MelBand Roformer | Vocals by Kimberley Jensen': 'vocals_mel_band_roformer.ckpt',
    'MelBand Roformer Kim | FT by unwa': 'mel_band_roformer_kim_ft_unwa.ckpt',
    'MelBand Roformer Kim | FT 2 by unwa': 'mel_band_roformer_kim_ft2_unwa.ckpt',
    'MelBand Roformer Kim | FT 2 Bleedless by unwa': 'mel_band_roformer_kim_ft2_bleedless_unwa.ckpt',
    'MelBand Roformer Kim | FT 3 by unwa': 'mel_band_roformer_kim_ft3_unwa.ckpt',
    'MelBand Roformer Kim | Inst V1 by Unwa': 'melband_roformer_inst_v1.ckpt',
    'MelBand Roformer Kim | Inst V1 Plus by Unwa': 'melband_roformer_inst_v1_plus.ckpt',
    'MelBand Roformer Kim | Inst V1 (E) by Unwa': 'melband_roformer_inst_v1e.ckpt',
    'MelBand Roformer Kim | Inst V1 (E) Plus by Unwa': 'melband_roformer_inst_v1e_plus.ckpt',
    'MelBand Roformer Kim | Inst V2 by Unwa': 'melband_roformer_inst_v2.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
    'MelBand Roformer | Vocals by becruily': 'mel_band_roformer_vocals_becruily.ckpt',
    'MelBand Roformer | Instrumental by becruily': 'mel_band_roformer_instrumental_becruily.ckpt',
    'MelBand Roformer | Vocals Fullness by Aname': 'mel_band_roformer_vocal_fullness_aname.ckpt',
    'BS Roformer | Vocals by Gabox': 'bs_roformer_vocals_gabox.ckpt',
    'MelBand Roformer | Vocals by Gabox': 'mel_band_roformer_vocals_gabox.ckpt',
    'MelBand Roformer | Vocals V2 by Gabox': 'mel_band_roformer_vocals_v2_gabox.ckpt',
    'MelBand Roformer | Vocals FV1 by Gabox': 'mel_band_roformer_vocals_fv1_gabox.ckpt',
    'MelBand Roformer | Vocals FV2 by Gabox': 'mel_band_roformer_vocals_fv2_gabox.ckpt',
    'MelBand Roformer | Vocals FV3 by Gabox': 'mel_band_roformer_vocals_fv3_gabox.ckpt',
    'MelBand Roformer | Vocals FV4 by Gabox': 'mel_band_roformer_vocals_fv4_gabox.ckpt',
    'MelBand Roformer | Vocals FV5 by Gabox': 'mel_band_roformer_vocals_fv5_gabox.ckpt',
    'MelBand Roformer | Vocals FV6 by Gabox': 'mel_band_roformer_vocals_fv6_gabox.ckpt',
    'MelBand Roformer | Instrumental by Gabox': 'mel_band_roformer_instrumental_gabox.ckpt',
    'MelBand Roformer | Instrumental 2 by Gabox': 'mel_band_roformer_instrumental_2_gabox.ckpt',
    'MelBand Roformer | Instrumental 3 by Gabox': 'mel_band_roformer_instrumental_3_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V1 by Gabox': 'mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V2 by Gabox': 'mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt',
    'MelBand Roformer | Instrumental Bleedless V3 by Gabox': 'mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V1 by Gabox': 'mel_band_roformer_instrumental_fullness_v1_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V2 by Gabox': 'mel_band_roformer_instrumental_fullness_v2_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness V3 by Gabox': 'mel_band_roformer_instrumental_fullness_v3_gabox.ckpt',
    'MelBand Roformer | Instrumental Fullness Noisy V4 by Gabox': 'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt',
    'MelBand Roformer | INSTV5 by Gabox': 'mel_band_roformer_instrumental_instv5_gabox.ckpt',
    'MelBand Roformer | INSTV5N by Gabox': 'mel_band_roformer_instrumental_instv5n_gabox.ckpt',
    'MelBand Roformer | INSTV6 by Gabox': 'mel_band_roformer_instrumental_instv6_gabox.ckpt',
    'MelBand Roformer | INSTV6N by Gabox': 'mel_band_roformer_instrumental_instv6n_gabox.ckpt',
    'MelBand Roformer | INSTV7 by Gabox': 'mel_band_roformer_instrumental_instv7_gabox.ckpt',
    'MelBand Roformer | INSTV7N by Gabox': 'mel_band_roformer_instrumental_instv7n_gabox.ckpt',
    'MelBand Roformer | INSTV8 by Gabox': 'mel_band_roformer_instrumental_instv8_gabox.ckpt',
    'MelBand Roformer | INSTV8N by Gabox': 'mel_band_roformer_instrumental_instv8n_gabox.ckpt',
    'MelBand Roformer | Instrumental FV7z by Gabox': 'mel_band_roformer_instrumental_fv7z_gabox.ckpt',
    'MelBand Roformer | Instrumental FV8 by Gabox': 'mel_band_roformer_instrumental_fv8_gabox.ckpt',
    'MelBand Roformer | Instrumental FVX by Gabox': 'mel_band_roformer_instrumental_fvx_gabox.ckpt',
    'MelBand Roformer | De-Reverb by anvuew': 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
    'MelBand Roformer | De-Reverb Less Aggressive by anvuew': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
    'MelBand Roformer | De-Reverb Mono by anvuew': 'dereverb_mel_band_roformer_mono_anvuew.ckpt',
    'MelBand Roformer | De-Reverb Big by Sucial': 'dereverb_big_mbr_ep_362.ckpt',
    'MelBand Roformer | De-Reverb Super Big by Sucial': 'dereverb_super_big_mbr_ep_346.ckpt',
    'MelBand Roformer | De-Reverb-Echo by Sucial': 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
    'MelBand Roformer | De-Reverb-Echo V2 by Sucial': 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt',
    'MelBand Roformer | De-Reverb-Echo Fused by Sucial': 'dereverb_echo_mbr_fused.ckpt',
    'MelBand Roformer Kim | SYHFT by SYH99999': 'MelBandRoformerSYHFT.ckpt',
    'MelBand Roformer Kim | SYHFT V2 by SYH99999': 'MelBandRoformerSYHFTV2.ckpt',
    'MelBand Roformer Kim | SYHFT V2.5 by SYH99999': 'MelBandRoformerSYHFTV2.5.ckpt',
    'MelBand Roformer Kim | SYHFT V3 by SYH99999': 'MelBandRoformerSYHFTV3Epsilon.ckpt',
    'MelBand Roformer Kim | Big SYHFT V1 by SYH99999': 'MelBandRoformerBigSYHFTV1.ckpt',
    'MelBand Roformer Kim | Big Beta 4 FT by unwa': 'melband_roformer_big_beta4.ckpt',
    'MelBand Roformer Kim | Big Beta 5e FT by unwa': 'melband_roformer_big_beta5e.ckpt',
    'MelBand Roformer | Big Beta 6 by unwa': 'melband_roformer_big_beta6.ckpt',
    'MelBand Roformer | Big Beta 6X by unwa': 'melband_roformer_big_beta6x.ckpt',
    'BS Roformer | Vocals Revive by Unwa': 'bs_roformer_vocals_revive_unwa.ckpt',
    'BS Roformer | Vocals Revive V2 by Unwa': 'bs_roformer_vocals_revive_v2_unwa.ckpt',
    'BS Roformer | Vocals Revive V3e by Unwa': 'bs_roformer_vocals_revive_v3e_unwa.ckpt',
    'BS Roformer | Chorus Male-Female by Sucial': 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
    'BS Roformer | Male-Female by aufr33': 'bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt',
    'MelBand Roformer | Aspiration by Sucial': 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
    'MelBand Roformer | Aspiration Less Aggressive by Sucial': 'aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt',
    'MelBand Roformer | Bleed Suppressor V1 by unwa-97chris': 'mel_band_roformer_bleed_suppressor_v1.ckpt',
    'BS Roformer | Vocals Resurrection by unwa': 'bs_roformer_vocals_resurrection_unwa.ckpt',
    'BS Roformer | Instrumental Resurrection by unwa': 'bs_roformer_instrumental_resurrection_unwa.ckpt',
    'BS Roformer SW by jarredou': 'BS-Roformer-SW.ckpt',
}

mdx23c_models = ['MDX23C_D1581.ckpt', 'MDX23C-8KFFT-InstVoc_HQ.ckpt', 'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
                 'MDX23C-De-Reverb-aufr33-jarredou.ckpt', 'MDX23C-DrumSep-aufr33-jarredou.ckpt']

mdxnet_models = [
    'UVR_MDXNET_KARA.onnx', 'UVR_MDXNET_KARA_2.onnx',
    'UVR-MDX-NET-Inst_full_292.onnx', 'UVR-MDX-NET_Inst_187_beta.onnx', 'UVR-MDX-NET_Inst_82_beta.onnx',
    'UVR-MDX-NET_Inst_90_beta.onnx', 'UVR-MDX-NET_Main_340.onnx', 'UVR-MDX-NET_Main_390.onnx',
    'UVR-MDX-NET_Main_406.onnx', 'UVR-MDX-NET_Main_427.onnx', 'UVR-MDX-NET_Main_438.onnx',
    'UVR-MDX-NET-Inst_HQ_1.onnx', 'UVR-MDX-NET-Inst_HQ_2.onnx', 'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET-Inst_HQ_4.onnx', 'UVR-MDX-NET-Inst_HQ_5.onnx', 'UVR_MDXNET_Main.onnx',
    'UVR-MDX-NET-Inst_Main.onnx', 'UVR_MDXNET_1_9703.onnx', 'UVR_MDXNET_2_9682.onnx',
    'UVR_MDXNET_3_9662.onnx', 'UVR-MDX-NET-Inst_1.onnx', 'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET-Inst_3.onnx', 'UVR_MDXNET_9482.onnx', 'UVR-MDX-NET-Voc_FT.onnx',
    'Kim_Vocal_1.onnx', 'Kim_Vocal_2.onnx', 'Kim_Inst.onnx', 'Reverb_HQ_By_FoxJoy.onnx',
    'UVR-MDX-NET_Crowd_HQ_1.onnx',
    'kuielab_a_vocals.onnx', 'kuielab_a_other.onnx', 'kuielab_a_bass.onnx', 'kuielab_a_drums.onnx',
    'kuielab_b_vocals.onnx', 'kuielab_b_other.onnx', 'kuielab_b_bass.onnx', 'kuielab_b_drums.onnx']

vrarch_models = [
    '5_HP-Karaoke-UVR.pth', '6_HP-Karaoke-UVR.pth',
    '1_HP-UVR.pth', '2_HP-UVR.pth', '3_HP-Vocal-UVR.pth', '4_HP-Vocal-UVR.pth',
    '7_HP2-UVR.pth', '8_HP2-UVR.pth', '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1.pth', '11_SP-UVR-2B-32000-2.pth', '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1.pth', '14_SP-UVR-4B-44100-2.pth', '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2.pth', '17_HP-Wind_Inst-UVR.pth', 'UVR-De-Echo-Aggressive.pth',
    'UVR-De-Echo-Normal.pth', 'UVR-DeEcho-DeReverb.pth', 'UVR-De-Reverb-aufr33-jarredou.pth',
    'UVR-DeNoise-Lite.pth', 'UVR-DeNoise.pth', 'UVR-BVE-4B_SN-44100-1.pth', 'UVR-BVE-4B_SN-44100-2.pth',
    'MGM_HIGHEND_v4.pth', 'MGM_LOWEND_A_v4.pth', 'MGM_LOWEND_B_v4.pth', 'MGM_MAIN_v4.pth']

demucs_models = ['htdemucs_ft.yaml', 'htdemucs_6s.yaml', 'htdemucs.yaml', 'hdemucs_mmi.yaml']

extensions = tuple(ALL_MEDIA_EXTENSIONS)

ARCH_MAP = {}
for fn in mdx23c_models:
    ARCH_MAP[fn] = "mdx23c"
for fn in mdxnet_models:
    ARCH_MAP[fn] = "mdxnet"
for fn in vrarch_models:
    ARCH_MAP[fn] = "vrarch"
for fn in demucs_models:
    ARCH_MAP[fn] = "demucs"
for _k, fn in roformer_models.items():
    ARCH_MAP[fn] = "roformer"


def _build_categorized_ensemble_choices():
    choices = []

    def _bucket(k):
        kl = k.lower()
        if "karaoke" in kl:
            return "🎤 Karaoke"
        if "de-reverb" in kl or "dereverb" in kl:
            return "🔊 De-Reverb"
        if "denoise" in kl or "noise" in kl:
            return "🎧 Denoise"
        if "crowd" in kl:
            return "👥 Crowd"
        if "aspiration" in kl:
            return "💨 Aspiration"
        if "male-female" in kl or "chorus" in kl:
            return "🎶 Male/Female"
        if "bleed" in kl:
            return "🩸 Bleed"
        if ("inst" in kl and "voc" not in kl) or "instrumental" in kl:
            return "🎸 Instrumental"
        if "vocal" in kl or "voc" in kl:
            return "🎙️ Vocals"
        return "📦 Other"

    grouped = {}
    for dn in roformer_models:
        grouped.setdefault(_bucket(dn), []).append(dn)
    for b in ["🎤 Karaoke", "🎙️ Vocals", "🎸 Instrumental", "🔊 De-Reverb", "🎧 Denoise",
              "👥 Crowd", "💨 Aspiration", "🎶 Male/Female", "🩸 Bleed", "📦 Other"]:
        for n in sorted(grouped.get(b, [])):
            choices.append((f"📦 Roformer ➡ {b} ➡ {n}", n))
    for fn in sorted(mdx23c_models):
        choices.append((f"🎛️ MDX23C ➡ {fn}", fn))
    for fn in sorted(mdxnet_models):
        choices.append((f"⚡ MDX-NET ➡ {fn}", fn))
    for fn in sorted(vrarch_models):
        choices.append((f"🎚️ VR Arch ➡ {fn}", fn))
    for fn in sorted(demucs_models):
        choices.append((f"🎼 Demucs ➡ {fn}", fn))
    return choices


ensemble_dropdown_choices = _build_categorized_ensemble_choices()

# ──────────────────────────────────────────────────────────────────────
#  設定 / 其他 UI 輔助
# ──────────────────────────────────────────────────────────────────────
def load_config_presence():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            return json.load(f).get("discord_presence", False)
    except Exception:
        return False


def initialize_presence():
    try:
        if load_config_presence():
            RPCManager.start_presence()
    except Exception:
        pass


initialize_presence()


def leaderboard(list_filter):
    try:
        if not list_filter:
            return "<i>Please select a filter first.</i>"
        if python_location and os.path.exists(python_location) and os.path.exists(separator_location):
            cmd = [python_location, separator_location, "-l", f"--list_filter={list_filter}"]
        else:
            cmd = [separator_location, "-l", f"--list_filter={list_filter}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return f"<pre>Error: {(r.stderr or r.stdout)[-2000:]}</pre>"
        rows = []
        for i, line in enumerate(re.findall(r"^(?!-+)(.+)$", r.stdout.strip(), re.MULTILINE)):
            cells = re.split(r"\s{2,}", line.strip())
            st = "font-weight:bold;font-size:1.2em;" if i == 0 else ""
            rows.append(f"<tr style='{st}'>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        return "<table border='1' style='border-collapse:collapse;'>" + "".join(rows) + "</table>"
    except subprocess.TimeoutExpired:
        return "<i>Timed out.</i>"
    except Exception as e:
        return f"<i>Error: {e}</i>"


def get_language_settings():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            c = json.load(f)
        if c.get("lang", {}).get("override"):
            return c["lang"]["selected_lang"]
    except Exception:
        pass
    return "Language automatically detected by system"


def save_lang_settings(sel):
    try:
        with open(config_file, "r", encoding="utf8") as f:
            c = json.load(f)
    except Exception:
        c = {}
    c.setdefault("lang", {})
    c["lang"]["override"] = (sel != "Language automatically detected by system")
    c["lang"]["selected_lang"] = sel
    try:
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w", encoding="utf8") as f:
            json.dump(c, f, indent=2)
        gr.Info(i18n("Language saved. Restart to apply."))
    except OSError as e:
        gr.Warning(f"Could not save language: {e}")


def alternative_model_downloader(method, key, source, output_dir=models_dir,
                                 progress=gr.Progress()):
    progress = _cli_progress(progress)
    logs = []
    if not method:
        return "Select a method."
    if not key:
        return "Select a model."
    if not source:
        return "Select a source."
    try:
        with open(models_file, 'r', encoding='utf-8') as f:
            md = json.load(f)
    except Exception as e:
        return f"Cannot read models file: {e}"
    if key not in md:
        return f"Model '{key}' not found in models.json."
    entry = md[key]
    if isinstance(entry, list):
        urls = entry
    elif isinstance(entry, dict):
        urls = entry.get(source) or entry.get('Github') or entry.get('HuggingFace') or []
    else:
        urls = []
    if not urls:
        return f"No URLs for '{key}'."
    os.makedirs(output_dir, exist_ok=True)
    total = len(urls)
    progress(0, desc="Starting...")
    for i, url in enumerate(urls):
        fn = os.path.basename(urllib.parse.urlparse(url).path)
        if not fn:
            logs.append(f"Skip (no filename): {url}")
            continue
        full = os.path.join(output_dir, fn)
        if os.path.exists(full) and os.path.getsize(full) > 0:
            logs.append(f"{fn} already exists.")
            continue
        progress((i + 0.1) / total, desc=f"Downloading {fn} ({i+1}/{total})")
        if method == 'wget':
            cmd = ['wget', '--progress=bar:force', '-O', full, url]
        elif method == 'curl':
            cmd = ['curl', '-L', '-#', '-o', full, url]
        else:
            logs.append(f"Unsupported method: {method}")
            continue

        def on_line(line, _i=i, _fn=fn):
            try:
                if method == 'wget' and '%' in line:
                    pct = int(line.strip().split('%')[0].split()[-1])
                    progress((_i + pct / 100) / total, desc=f"{_fn} ({pct}%)")
                elif method == 'curl' and '#' in line:
                    fp = min(line.count('#') / 50, 1.0)
                    progress((_i + fp) / total, desc=f"{_fn} ({int(fp*100)}%)")
            except Exception:
                pass

        try:
            rc, _o, err, timed_out = stream_process(cmd, timeout=7200, on_stderr=on_line)
            if rc != 0 or timed_out:
                logs.append(f"Error downloading {fn}: " + ("timeout" if timed_out
                            else "\n".join(err[-5:])))
                if os.path.exists(full):
                    try:
                        os.remove(full)
                    except OSError:
                        pass
            else:
                logs.append(f"{fn} downloaded!")
            progress((i + 1) / total, desc=f"{i+1}/{total} done")
        except FileNotFoundError:
            logs.append(f"'{method}' is not installed.")
            break
        except Exception as e:
            logs.append(f"Error: {e}")
    progress(1.0, desc="Done")
    return "\n".join(logs)


def read_main_config():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_settings_from_file(fp):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def get_initial_settings():
    mc = read_main_config() or {}
    s = copy.deepcopy(load_settings_from_file(default_settings_file) or {})
    if mc.get('load_custom_settings'):
        cs = load_settings_from_file(custom_settings_file)
        if cs:
            for sec, params in cs.items():
                if sec in s and isinstance(s[sec], dict) and isinstance(params, dict):
                    s[sec].update(params)
                else:
                    s[sec] = params
    return s


initial_settings = get_initial_settings()

# ──────────────────────────────────────────────────────────────────────
#  合奏核心
# ──────────────────────────────────────────────────────────────────────
def resolve_ensemble_model(sel):
    if not sel:
        return None, None
    if sel in roformer_models:
        return roformer_models[sel], "roformer"
    return sel, ARCH_MAP.get(sel, "unknown")


def _classify_stem(path):
    b = os.path.basename(path).lower()
    if "instrumental" in b or "(inst)" in b or "_inst" in b:
        return "Instrumental", "instrumental"
    if "vocals" in b or "vocal" in b:
        return "Vocals", "vocals"
    if "drums" in b:
        return "Drums", "other"
    if "bass" in b:
        return "Bass", "other"
    if "other" in b:
        return "Other", "other"
    return os.path.splitext(os.path.basename(path))[0][:40], "other"


def _pick_stems(stems, max_stems=6):
    if not stems:
        return []
    labelled = [(p, _classify_stem(p)) for p in stems]
    inst = [p for p, (l, k) in labelled if k == "instrumental"]
    voc = [p for p, (l, k) in labelled if k == "vocals"]
    other = [p for p, (l, k) in labelled if k not in ("instrumental", "vocals")]
    chosen = []
    for p in inst + voc + other:
        if p not in chosen:
            chosen.append(p)
        if len(chosen) >= max_stems:
            break
    return chosen[:max_stems]


_ARCH_TO_SECTION = {"roformer": "Roformer", "mdx23c": "MDX23C", "mdxnet": "MDX-NET",
                    "vrarch": "VR Arch", "demucs": "Demucs"}


def _ens_settings(arch_key, seg_override=0, batch_override=0, overlap_override=0):
    r = initial_settings.get(arch_key, {}) or {}
    common = {"batch_size": int(batch_override) if batch_override else r.get("batch_size", 1),
              "normalization_threshold": r.get("normalization_threshold", 0.9)}
    if arch_key in ("Roformer", "MDX23C"):
        s = {**common,
             "segment_size": int(seg_override) if seg_override else r.get("segment_size", 256),
             "override_segment_size": bool(r.get("override_segment_size", False)),
             "overlap": int(overlap_override) if overlap_override else r.get("overlap", 8)}
        if seg_override:
            s["override_segment_size"] = True
        return s
    if arch_key == "MDX-NET":
        return {**common, "hop_length": r.get("hop_length", 1024),
                "segment_size": int(seg_override) if seg_override else r.get("segment_size", 256),
                "denoise": r.get("denoise", True), "overlap": r.get("overlap", 0.25)}
    if arch_key == "VR Arch":
        return {**common, "window_size": r.get("window_size", 512),
                "aggression": r.get("aggression", 5), "tta": r.get("tta", True),
                "post_process": r.get("post_process", False),
                "post_process_threshold": r.get("post_process_threshold", 0.2),
                "high_end_process": r.get("high_end_process", False)}
    if arch_key == "Demucs":
        return {**common, "shifts": r.get("shifts", 2),
                "segment_size": r.get("segment_size", 40),
                "segments_enabled": r.get("segments_enabled", True),
                "overlap": r.get("overlap", 0.25)}
    return common


def _arch_runtime_kw(arch, settings):
    if arch in ("roformer", "mdx23c"):
        return {"segment_size": settings.get("segment_size", 256),
                "override_model_segment_size": settings.get("override_segment_size", False),
                "batch_size": settings.get("batch_size", 1),
                "overlap": settings.get("overlap", 8)}
    if arch == "mdxnet":
        return {"hop_length": settings.get("hop_length", 1024),
                "segment_size": settings.get("segment_size", 256),
                "overlap": settings.get("overlap", 0.25),
                "batch_size": settings.get("batch_size", 1),
                "enable_denoise": settings.get("denoise", True)}
    if arch == "vrarch":
        return {"batch_size": settings.get("batch_size", 1),
                "window_size": settings.get("window_size", 512),
                "aggression": settings.get("aggression", 5),
                "enable_tta": settings.get("tta", True),
                "enable_post_process": settings.get("post_process", False),
                "post_process_threshold": settings.get("post_process_threshold", 0.2),
                "high_end_process": settings.get("high_end_process", False)}
    if arch == "demucs":
        return {"batch_size": settings.get("batch_size", 1),
                "segment_size": settings.get("segment_size", 40),
                "shifts": settings.get("shifts", 2),
                "overlap": settings.get("overlap", 0.25),
                "segments_enabled": settings.get("segments_enabled", True)}
    return {}


def _empty_slot_outputs(status_text):
    return (
        None,
        _update(gr.Radio, choices=[], value=None, visible=False),
        *[_update(gr.Audio, visible=False, value=None, label=f"Stem {i+1}") for i in range(6)],
        _update(gr.Markdown, value=status_text)
    )


def _noop_slot_outputs(status_text):
    return (
        _skip(),
        _skip(),
        *[_skip() for _ in range(6)],
        _update(gr.Markdown, value=status_text)
    )


@track_presence("Ensemble slot separation")
def run_one_ensemble_model_stream(slot_index, audio, selected_model, weight,
                                  seg_override, batch_override, out_fmt,
                                  progress=gr.Progress()):
    progress = _cli_progress(progress)
    tag = f"🧩 Slot {slot_index + 1}"
    if not audio or not os.path.isfile(audio):
        yield _empty_slot_outputs(f"{tag} ❌ Load an audio or video file at the top first.")
        return
    if not selected_model:
        yield _empty_slot_outputs(f"{tag} ❌ Pick a model.")
        return
    if weight is None or float(weight) <= 0:
        yield _empty_slot_outputs(f"{tag} ❌ Weight must be greater than 0.")
        return
    fname, arch = resolve_ensemble_model(selected_model)
    if not fname or arch == "unknown":
        yield _empty_slot_outputs(f"{tag} ❌ Unknown model '{selected_model}'.")
        return
    rep = JobReporter(title=f"{tag} · {selected_model}")
    # [REV-UI-FEEDBACK] 立即開啟狀態面板，避免首次點 Separate 時面板空白。
    rep.panels_visible = True
    rep.log_add(f"input: {os.path.basename(audio)} ({_describe_audio(audio)})")
    mode_txt = ("isolated worker (default)" if ALWAYS_SUBPROCESS and SUBPROC_FALLBACK
                else "in-process")
    rep.log_add(f"execution: {'CPU' if FORCE_CPU or not GPU_PRESENT else 'GPU'} · {mode_txt}")
    if GPU_UNLOCK_PROFILE and arch in ("roformer", "mdx23c"):
        rep.log_add("gpu-unlock AMP profile: 已請求（kernel 解鎖需另行驗證）")
    if not os.path.exists(os.path.join(models_dir, fname)):
        rep.log_add(f"model not cached — it will be downloaded ({fname})")
    settings = _ens_settings(_ARCH_TO_SECTION.get(arch, "Roformer"),
                             seg_override, batch_override)
    arch_kw = _arch_runtime_kw(arch, settings)
    norm = settings.get("normalization_threshold", 0.9)
    fmt = out_fmt if out_fmt in OUTPUT_FORMATS else "wav"
    holder = {"files": None, "error": None}

    def _work():
        processing_path = None
        base_out = None
        try:
            with JOB_LOCK:
                rep.stage_set("preparing input", 0.05)
                processing_path = ensure_wav(audio, rep)
                base_out = get_temp_output_dir()
                _mark_active(base_out)
                model_out = os.path.join(base_out, f"slot_{slot_index}")
                os.makedirs(model_out, exist_ok=True)
                free_gpu(deep=True)
                holder["files"] = _separate_with_retry(
                    arch, fname, processing_path, model_out, fmt, norm, None,
                    arch_kw, reporter=rep, desc=f"Slot {slot_index+1}")
        except BaseException as e:
            holder["error"] = str(e)
            rep.fail(str(e))
        finally:
            if processing_path and processing_path != audio:
                _cleanup_temp_wav(processing_path, audio)
            if base_out:
                _unmark_active(base_out)
            free_gpu(deep=True)

    th = threading.Thread(target=_work, daemon=True, name=f"sep-slot-{slot_index}")
    th.start()
    last = ""
    while th.is_alive():
        txt = rep.render()
        if txt != last:
            last = txt
            yield _noop_slot_outputs(txt)
        try:
            _st, _d, pct, _l, _e, _el = rep.snapshot()
            progress(min(0.99, pct), desc=f"{tag}: {_st}")
        except Exception:
            pass
        time.sleep(0.8)
    th.join(timeout=5)
    if holder["error"] or not holder["files"]:
        err = holder["error"] or "the model produced no output files"
        rep.fail(err)
        yield _empty_slot_outputs(rep.render(final_ok=False))
        return
    picked = _pick_stems(holder["files"], max_stems=6)
    if not picked:
        rep.fail("no usable stems were produced")
        yield _empty_slot_outputs(rep.render(final_ok=False))
        return
    stems_meta = []
    for p in picked:
        label, kind = _classify_stem(p)
        stems_meta.append({"path": p, "label": label, "kind": kind,
                           "sr": probe_sr(p) or 0})
    slot_state = {"model": selected_model, "weight": float(weight), "stems": stems_meta}
    choices = [f"Stem {i+1}: {sm['label']}" for i, sm in enumerate(stems_meta)]
    default = choices[0]
    for i, sm in enumerate(stems_meta):
        if sm["kind"] == "instrumental":
            default = choices[i]
            break

    audio_updates = []
    for i in range(6):
        if i < len(stems_meta):
            sm = stems_meta[i]
            audio_updates.append(_update(gr.Audio,
                visible=True, value=sm["path"],
                label=f"Stem {i+1}: {sm['label']} · {sm['sr']} Hz"))
        else:
            audio_updates.append(_update(gr.Audio, visible=False, value=None, label=f"Stem {i+1}"))

    radio = _update(gr.Radio, choices=choices, value=default, visible=True,
                      label=f"Include from '{selected_model}'")

    rep.log_add(f"produced {len(stems_meta)} stem(s) at ≥ {TARGET_SR} Hz")
    progress(1.0, desc="Done")
    yield (slot_state, radio, *audio_updates, _update(gr.Markdown, value=rep.render(final_ok=True)))


def clear_one_slot(i):
    return _empty_slot_outputs(f"🧩 Slot {i+1} 🧹 Cleared.")


def _resolve_stem(state, radio):
    if not state or not state.get("stems"):
        return None
    if not radio:
        return state["stems"][0]["path"]
    try:
        n = int(str(radio).split(":", 1)[0].replace("Stem", "").strip()) - 1
        if 0 <= n < len(state["stems"]):
            return state["stems"][n]["path"]
    except Exception:
        pass
    return state["stems"][0]["path"]


def ensemble_combine_from_slots(s1, s2, s3, s4, r1, r2, r3, r4,
                                progress=gr.Progress()):
    progress = _cli_progress(progress)
    selected = []
    for st, rv in zip([s1, s2, s3, s4], [r1, r2, r3, r4]):
        if not st:
            continue
        p = _resolve_stem(st, rv)
        if p and os.path.isfile(p):
            selected.append((st.get("model", "?"), p, float(st.get("weight", 1.0))))
    if not selected:
        return None, "❌ No stems to combine. Run at least one slot first."
    tw = sum(w for _, _, w in selected)
    if tw <= 0:
        return None, "❌ Total weight is zero."
    nw = [w / tw for _, _, w in selected]
    sr_target = TARGET_SR
    for _, p, _ in selected:
        sr_target = max(sr_target, probe_sr(p) or 0)
    lines = [f"Combining {len(selected)} stem(s) at {sr_target} Hz:"]
    for (m, p, w), n in zip(selected, nw):
        lines.append(f"  • {m} → {os.path.basename(p)}  (weight {w:.2f} → {n*100:.0f}%)")
    acc = None
    ml = None
    mc = 1
    loaded = []
    try:
        for i, ((_m, p, _w), w) in enumerate(zip(selected, nw)):
            progress(0.1 + 0.6 * (i / max(1, len(selected))),
                     desc=f"Loading {os.path.basename(p)}…")
            try:
                wav, _ = librosa.load(p, sr=sr_target, mono=False, res_type="soxr_hq")
            except Exception as e:
                return None, "\n".join(lines + [f"❌ Cannot read '{os.path.basename(p)}': {e}"])
            if wav.ndim == 1:
                wav = wav.reshape(1, -1)
            wav = wav.astype(np.float32, copy=False)
            loaded.append((wav, w))
            ml = wav.shape[-1] if ml is None else min(ml, wav.shape[-1])
            mc = max(mc, wav.shape[0])
        progress(0.75, desc="Mixing…")
        for wav, w in loaded:
            t = wav[..., :ml]
            if t.shape[0] < mc:
                reps = int(np.ceil(mc / t.shape[0]))
                t = np.tile(t, (reps, 1))[:mc]
            if acc is None:
                acc = (w * t).astype(np.float32)
            else:
                acc += (w * t).astype(np.float32)
        pk = float(np.max(np.abs(acc))) if acc is not None and acc.size else 0.0
        if pk > 1.0:
            acc /= pk
            lines.append(f"  peak {pk:.2f} → normalised to 1.0")
        out_dir = get_temp_output_dir()
        _mark_active(out_dir)
        try:
            out = os.path.join(out_dir, "ensemble_combined.wav")
            progress(0.9, desc="Writing output…")
            sf.write(out, acc.T if acc.ndim > 1 else acc, sr_target, subtype="PCM_24")
            ensure_min_samplerate(out)
            progress(1.0, desc="Combined")
            lines.append(f"✅ Written: {os.path.basename(out)} "
                         f"({sr_target} Hz · 24-bit · {mc} ch)")
            return out, "\n".join(lines)
        finally:
            _unmark_active(out_dir)
    except Exception as e:
        return None, "\n".join(lines + [f"❌ Combine failed: {e}"])
    finally:
        loaded.clear()
        del acc
        gc.collect()


# ──────────────────────────────────────────────────────────────────────
#  GPU 面板
# ──────────────────────────────────────────────────────────────────────
def gpu_status_text():
    """產生 GPU / Memory 分頁的狀態文字。

    [REV-INTEGRATED] FORCE_CPU 時只印硬體摘要（不呼叫任何 CUDA runtime API），
    避免在 CPU-only / 驅動異常環境下因為呼叫 CUDA 而拋錯；同時讓使用者
    仍可看到「GPU 存在但被忽略」的資訊。
    """
    lines = []
    exec_mode = ("isolated worker process (default)"
                 if ALWAYS_SUBPROCESS and SUBPROC_FALLBACK else "in-process")
    lines.append(f"**Execution mode:** {exec_mode}")
    lines.append(f"**Output policy:** every file ≥ {TARGET_SR} Hz")
    lines.append(f"**{_gpu_unlock_profile_state_text()}**")
    if FORCE_CPU:
        lines.append("**Device policy:** CPU forced (`UVR_FORCE_CPU=1` or `--cpu`).")
        if GPU_PRESENT:
            cap_txt = (f"SM {GPU_CAPABILITY[0]}.{GPU_CAPABILITY[1]}"
                       if GPU_CAPABILITY else "unknown")
            lines.append(f"**GPU (present but ignored):** {GPU_NAME} — "
                         f"{TOTAL_VRAM_GB:.1f} GB total · {cap_txt}")
        return "\n".join(lines)
    if not GPU_PRESENT:
        lines.append("**Device policy:** no CUDA device visible to PyTorch. "
                     "Check driver / torch build.")
        return "\n".join(lines)
    with GPU.lock:
        dead, fatal, ooms, lasterr = (GPU.context_dead, GPU.fatal_events,
                                      GPU.oom_events, GPU.last_error)
    free = vram_free_gb()
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:
        alloc = reserved = 0.0
    with MODEL_CACHE.lock:
        cached = MODEL_CACHE.key[1] if MODEL_CACHE.key else "none"
    smi = _nvidia_smi()
    lines.append(f"**GPU:** {GPU_NAME} — {TOTAL_VRAM_GB:.1f} GB total")
    state = ("in-process CUDA context ACTIVE" if not dead
             else "in-process context lost → isolated worker in use")
    lines.append(f"**State:** {state}")
    lines.append(f"**VRAM (torch):** {0.0 if free is None else free:.2f} GB free · "
                 f"{alloc:.2f} GB allocated · {reserved:.2f} GB reserved")
    if smi:
        lines.append(f"**VRAM (nvidia-smi):** {smi['used_mb']:.0f} / {smi['total_mb']:.0f} MB used · "
                     f"util {smi['util']}% · {smi['temp']} °C")
    lines.append(f"**Cached model:** `{cached}` (cache "
                 f"{'enabled' if MODEL_CACHE_ENABLED else 'disabled'})")
    lines.append(f"**Events:** {fatal} fatal CUDA fault(s), {ooms} OOM(s)")
    lines.append(f"**Last error:** `{(lasterr or 'none')[:300]}`")
    return "\n".join(lines)


def ui_free_memory():
    got = _acquire_sep_lock(timeout=5.0)
    try:
        MODEL_CACHE.clear(quiet=False)
        free_gpu(deep=True)
    finally:
        if got:
            _separator_lock.release()
    if not got:
        gr.Warning("A separation is still holding the lock; freed what was safe.")
    gr.Info("Released cached model and GPU memory.")
    return gpu_status_text()


def ui_retry_gpu():
    if not gpu_configured():
        return gpu_status_text()
    got = _acquire_sep_lock(timeout=5.0)
    try:
        MODEL_CACHE.clear()
        ok = gpu_healthy(allow_recovery=True)
    finally:
        if got:
            _separator_lock.release()
    if not got:
        gr.Warning("A separation is still holding the lock; re-test may be incomplete.")
        return gpu_status_text()
    gr.Info("GPU context is healthy again." if ok
            else "GPU context still unusable — the isolated worker will be used.")
    return gpu_status_text()


def ui_purge_temp():
    removed = 0
    with _active_folders_lock:
        snap = set(_active_folders)
    try:
        entries = os.listdir(TEMP_ROOT)
    except OSError:
        entries = []
    for f in entries:
        fp = os.path.abspath(os.path.join(TEMP_ROOT, f))
        if fp in snap or not os.path.isdir(fp):
            continue
        shutil.rmtree(fp, ignore_errors=True)
        removed += 1
    gr.Info(f"Removed {removed} temporary folder(s).")
    return gpu_status_text()


def read_mode_config():
    try:
        with open(config_file, "r", encoding="utf8") as f:
            return json.load(f).get("theme", {}).get("mode", "dark")
    except Exception:
        return "dark"


mode = read_mode_config()
SESSION_ID = uuid.uuid4().hex
js_func = f"""
function refresh() {{
    const url = new URL(window.location);
    const want_theme = '{mode}';
    const want_sid = '{SESSION_ID}';
    let changed = false;
    if (url.searchParams.get('__theme') !== want_theme) {{ url.searchParams.set('__theme', want_theme); changed = true; }}
    if (url.searchParams.get('__sid') !== want_sid) {{ url.searchParams.set('__sid', want_sid); changed = true; }}
    if (changed) window.location.href = url.href;
}}
"""

try:
    _THEME = loadThemes.load_json() or "NoCrypt/miku"
except Exception:
    _THEME = "NoCrypt/miku"

# ══════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════
with gr.Blocks(title="🎤 UVR5 UI 🎧", theme=_THEME, js=js_func) as app:
    gr.Markdown("<h1>🎤 UVR5 UI 🎧</h1>")
    gr.Markdown(i18n("If you like UVR5 UI you can star my repo on [GitHub](https://github.com/Eddycrack864/UVR5-UI)"))
    gr.Markdown(i18n("Try UVR5 UI on Hugging Face with A100 [here](https://huggingface.co/spaces/TheStinger/UVR5_UI)"))
    with gr.Tabs():
        with gr.TabItem("🎛️ Ensemble"):
            gr.Markdown("### 📥 Input — paste a **link** or drop an **audio / video** file (handled automatically)")
            with gr.Accordion("🔗 Fetch from a link (URL)", open=True):
                ens_link = gr.Textbox(label="URL", placeholder="YouTube / SoundCloud / etc.",
                                      lines=1, interactive=True)
                ens_dlbtn = gr.Button("Download into the input field", variant="primary")
            ens_input = gr.File(
                label="Audio / Video input (mp3, wav, flac, m4a, mp4, mkv, mov, webm, …)",
                file_types=ALL_MEDIA_EXTENSIONS, type="filepath", interactive=True)
            ens_source = gr.State(None)
            ens_preview = gr.Audio(label="Resolved audio (preview)", type="filepath",
                                   interactive=False)
            ens_input_status = gr.Markdown("No input loaded.")
            with gr.Accordion("⚙️ Performance / VRAM (applies to every slot)", open=False):
                with gr.Row():
                    ens_seg = gr.Dropdown(
                        label="Segment size override (0 = model default)",
                        choices=[0, 64, 128, 256, 384, 512, 768, 1024], value=0,
                        interactive=True)
                    ens_bs = gr.Dropdown(label="Batch size override (0 = model default)",
                                         choices=[0, 1, 2, 4], value=0,
                                         interactive=True)
                    ens_fmt = gr.Dropdown(label="Stem format", choices=OUTPUT_FORMATS,
                                          value="wav", interactive=True)
                gr.Markdown(
                    "Lower the segment size first if you hit CUDA/VRAM errors "
                    "(1024 → 512 → 256 → 128). The app also retries automatically with "
                    "halved and quartered segments before falling back to CPU. "
                    "Explicit overrides above the safe cap are honored with a warning "
                    "instead of being silently clamped. "
                    f"All stems are written at ≥ {TARGET_SR} Hz regardless of the model's "
                    "internal rate.")
            ens_input.change(fn=smart_media_ingest, inputs=[ens_input],
                             outputs=[ens_source, ens_preview, ens_input_status])
            ens_dlbtn.click(fn=download_audio_to_slot, inputs=[ens_link],
                            outputs=[ens_source, ens_preview, ens_input_status])
            s1_state = gr.State(None)
            s2_state = gr.State(None)
            s3_state = gr.State(None)
            s4_state = gr.State(None)

            def build_slot(n):
                gr.Markdown(f"### 🧩 Slot {n}")
                m = gr.Dropdown(label=f"Model {n}", choices=ensemble_dropdown_choices,
                                value=None, interactive=True, filterable=True)
                w = gr.Slider(label=f"Weight {n}", minimum=0.0, maximum=1.0, value=1.0,
                              step=0.05, interactive=True)
                with gr.Row():
                    sb = gr.Button("⚡ Separate", variant="primary", size="sm")
                    cb = gr.Button("🧹 Clear", variant="secondary", size="sm")
                st = gr.Markdown(value="Idle.", label=f"Slot {n} status")
                pk = gr.Radio(label=f"Include from slot {n}", choices=[], value=None,
                              visible=False, interactive=True)
                audios = [gr.Audio(label=f"Stem {i+1}", type="filepath", visible=False,
                                   interactive=False) for i in range(6)]
                return m, w, sb, cb, st, pk, audios

            with gr.Row():
                with gr.Column():
                    m1, w1, sb1, cb1, st1, pk1, a1s = build_slot(1)
                with gr.Column():
                    m2, w2, sb2, cb2, st2, pk2, a2s = build_slot(2)
            with gr.Row():
                with gr.Column():
                    m3, w3, sb3, cb3, st3, pk3, a3s = build_slot(3)
                with gr.Column():
                    m4, w4, sb4, cb4, st4, pk4, a4s = build_slot(4)
            gr.Markdown("---")
            comb_btn = gr.Button("🎛️ Combine Selected", variant="primary")
            ens_out = gr.Audio(label="Final Ensemble Output", type="filepath", interactive=False)
            comb_status = gr.Markdown(label="Combine report", value="")
            gr.Markdown("Slots run one at a time (app-wide lock) and share the same input. "
                        "Run any model, pick a stem, then combine with the weights above.")

            def _sep(i):
                def h(audio, sm, w, seg, bs, fmt, progress=gr.Progress()):
                    yield from run_one_ensemble_model_stream(
                        i, audio, sm, w, seg, bs, fmt, progress=progress)
                return h

            def _clr(i):
                def h():
                    return clear_one_slot(i)
                return h

            sb1.click(fn=_sep(0), inputs=[ens_source, m1, w1, ens_seg, ens_bs, ens_fmt],
                      outputs=[s1_state, pk1, *a1s, st1])
            sb2.click(fn=_sep(1), inputs=[ens_source, m2, w2, ens_seg, ens_bs, ens_fmt],
                      outputs=[s2_state, pk2, *a2s, st2])
            sb3.click(fn=_sep(2), inputs=[ens_source, m3, w3, ens_seg, ens_bs, ens_fmt],
                      outputs=[s3_state, pk3, *a3s, st3])
            sb4.click(fn=_sep(3), inputs=[ens_source, m4, w4, ens_seg, ens_bs, ens_fmt],
                      outputs=[s4_state, pk4, *a4s, st4])
            cb1.click(fn=_clr(0), inputs=[], outputs=[s1_state, pk1, *a1s, st1])
            cb2.click(fn=_clr(1), inputs=[], outputs=[s2_state, pk2, *a2s, st2])
            cb3.click(fn=_clr(2), inputs=[], outputs=[s3_state, pk3, *a3s, st3])
            cb4.click(fn=_clr(3), inputs=[], outputs=[s4_state, pk4, *a4s, st4])
            comb_btn.click(fn=ensemble_combine_from_slots,
                           inputs=[s1_state, s2_state, s3_state, s4_state,
                                   pk1, pk2, pk3, pk4],
                           outputs=[ens_out, comb_status])
        with gr.TabItem(i18n("Pitch Changer")):
            gr.Markdown("Pitch shift using the Rubber Band engine. Supports fractional "
                        f"semitones (e.g. +1.5, -2.25). Output is 24-bit ≥ {TARGET_SR} Hz.")
            pc_input = gr.File(
                label="Audio / Video input (auto-converted, never downsampled)",
                file_types=ALL_MEDIA_EXTENSIONS, type="filepath", interactive=True)
            pc_source = gr.State(None)
            pc_preview = gr.Audio(label="Resolved audio (preview)", type="filepath",
                                  interactive=False)
            pc_input_status = gr.Markdown("No input loaded.")
            with gr.Accordion("🔗 …or fetch from a link", open=False):
                pc_link = gr.Textbox(label="URL", placeholder="YouTube / SoundCloud / etc.",
                                     lines=1, interactive=True)
                pc_dlbtn = gr.Button("Download into the input field", variant="primary")
            pc_input.change(fn=smart_media_ingest, inputs=[pc_input],
                            outputs=[pc_source, pc_preview, pc_input_status])
            pc_dlbtn.click(fn=download_audio_to_slot, inputs=[pc_link],
                           outputs=[pc_source, pc_preview, pc_input_status])
            with gr.Row():
                with gr.Column(scale=2):
                    pc_steps = gr.Slider(label="Pitch (semitones)", minimum=-12, maximum=12,
                                         value=0, step=0.05, interactive=True)
                    pc_prec = gr.Number(label="Exact pitch", value=0.0, precision=4,
                                        interactive=True)
                with gr.Row():
                    pc_form = gr.Checkbox(label="Preserve Formant", value=True,
                                          interactive=True)
                    pc_qual = gr.Slider(label="Quality (1-5)", minimum=1, maximum=5,
                                        value=5, step=1, interactive=True)
            pc_btn = gr.Button("Pitch Shift", variant="primary")
            with gr.Row():
                with gr.Column(scale=1):
                    pc_out = gr.Audio(label="Output", type="filepath")

            def _s2n(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0

            def _n2s(v):
                if v is None or v == "":
                    return _skip()
                try:
                    fv = max(-12.0, min(12.0, float(v)))
                except (TypeError, ValueError):
                    return _skip()
                return _update(gr.Slider, value=round(fv / 0.05) * 0.05)

            pc_steps.release(fn=_s2n, inputs=[pc_steps], outputs=[pc_prec],
                             show_progress="hidden")
            pc_prec.submit(fn=_n2s, inputs=[pc_prec], outputs=[pc_steps],
                           show_progress="hidden")
            pc_btn.click(fn=pitch_shift, inputs=[pc_source, pc_prec, pc_form, pc_qual],
                         outputs=[pc_out])
        with gr.TabItem("🖥️ GPU / Memory"):
            gpu_md = gr.Markdown(gpu_status_text())
            with gr.Row():
                gpu_refresh = gr.Button("🔄 Refresh", variant="secondary")
                gpu_free = gr.Button("🧹 Free VRAM (unload cached model)", variant="primary")
                gpu_retry = gr.Button("♻️ Re-test / recover GPU context", variant="primary")
                gpu_purge = gr.Button("🗑️ Purge temp files", variant="secondary")
            gpu_refresh.click(fn=gpu_status_text, inputs=[], outputs=[gpu_md])
            gpu_free.click(fn=ui_free_memory, inputs=[], outputs=[gpu_md])
            gpu_retry.click(fn=ui_retry_gpu, inputs=[], outputs=[gpu_md])
            gpu_purge.click(fn=ui_purge_temp, inputs=[], outputs=[gpu_md])
            gr.Markdown(
                "**Isolated worker is the default.** Each separation runs in a throwaway "
                "process, so the OS reclaims 100% of the VRAM afterwards and a CUDA fault "
                "can never corrupt this app's context. Run with `--in-process` (or "
                "`UVR_ALWAYS_SUBPROCESS=0`) to use the old behaviour.\n"
                "**Fallback ladder:** full segment → ½ segment → ¼ segment → CPU. "
                "Duplicate attempts are skipped, and the planner clamps the model-default "
                "segment size to something sane for the detected VRAM before the first try. "
                "Explicit user overrides above the cap are honored with a warning.\n"
                "**If you hit `CUBLAS_STATUS_EXECUTION_FAILED` repeatedly:** lower the "
                "segment size, keep batch size at 1, and verify the installed torch matches "
                "the CUDA version reported by `nvidia-smi`.\n"
                "**GPU 解鎖設定檔：** 使用 `--gpu-unlock-profile` 或設定環境變數 "
                "`UVR_GPU_UNLOCK_PROFILE=1` 可啟用。此設定檔僅對 Linux SM 8.6 上的 "
                "RoFormer / MDX23C 請求音訊分離器原生 AMP；kernel module 必須另外安裝，"
                "本程式不會載入它。明確設定 `UVR_AUTOCAST=0/1` 會覆寫此設定檔。\n"
                "**Environment switches:** `UVR_ALWAYS_SUBPROCESS` (default 1), "
                "`UVR_ALLOW_CPU_FALLBACK`, `UVR_MODEL_CACHE`, `UVR_MODEL_IDLE_TIMEOUT`, "
                "`UVR_EXPANDABLE_SEGMENTS` (default 1), `UVR_AUTOCAST`, `UVR_CUDNN_BENCHMARK`, "
                "`UVR_SUBPROC_TIMEOUT` (5400), `UVR_INPROC_TIMEOUT` (1800), "
                "`UVR_MODEL_LOAD_TIMEOUT` (900), `UVR_TEMP_TTL` (10800), "
                f"`UVR_TARGET_SR` ({TARGET_SR}), `UVR_FORCE_CPU`, `UVR_CUDA_DEBUG`.")
        with gr.TabItem(i18n("Leaderboard")):
            with gr.Group():
                with gr.Row(equal_height=True):
                    lb_filter = gr.Dropdown(
                        label=i18n("List filter"),
                        choices=["vocals", "instrumental", "reverb", "echo", "noise",
                                 "crowd", "dry", "aspiration", "male", "woodwinds",
                                 "kick", "drums", "bass", "guitar", "piano", "other"],
                        value=None)
                    lb_btn = gr.Button(i18n("Show list!"), variant="primary")
            lb_out = gr.HTML(label=i18n("Leaderboard"))
            lb_btn.click(fn=leaderboard, inputs=[lb_filter], outputs=[lb_out])
        with gr.TabItem(i18n("Themes")):
            th_sel = gr.Dropdown(label=i18n("Theme"), choices=loadThemes.get_list(),
                                 value=loadThemes.read_json(), interactive=True)
            md_sel = gr.Dropdown(label=i18n("Mode"), choices=["light", "dark"],
                                 value=read_mode_config(), interactive=True)
            th_sel.change(fn=loadThemes.select_theme, inputs=th_sel, outputs=[])
            md_sel.change(fn=loadThemes.select_mode, inputs=md_sel, outputs=[])
        with gr.TabItem(i18n("Settings")):
            with gr.Accordion(i18n("Language selector"), open=False):
                lang_sel = gr.Dropdown(
                    label=i18n("Language"), value=get_language_settings(),
                    choices=["Language automatically detected by system"] + _available_languages(),
                    interactive=True)
                lang_sel.change(fn=save_lang_settings, inputs=[lang_sel], outputs=[])
            with gr.Accordion(i18n("Alternative model downloader"), open=False):
                with gr.Row():
                    dl_method = gr.Dropdown(label=i18n("Download method"), value=None,
                                            choices=["wget", "curl"], interactive=True)
                    dl_source = gr.Dropdown(label=i18n("Download source"), value=None,
                                            choices=["Github", "Hugging Face"], interactive=True)
                    dl_mkey = gr.Dropdown(label=i18n("Model to download"), value=None,
                                          choices=list(roformer_models.keys()) + mdx23c_models
                                          + mdxnet_models + vrarch_models, interactive=True)
                    dl_mbtn = gr.Button(i18n("Download!"), variant="primary")
                dl_mout = gr.Textbox(label=i18n("Output information"), lines=5,
                                     interactive=False)
                dl_mbtn.click(fn=alternative_model_downloader,
                              inputs=[dl_method, dl_mkey, dl_source], outputs=[dl_mout])
        with gr.TabItem(i18n("Credits")):
            gr.Markdown("""UVR5 UI by Eddycrack 864. Join AI HUB.
python-audio-separator by beveradb.
Thanks to Ilaria, Mikus, Nick088, NeoDev, ArisDev, Bebra228.
Thanks to yt-dlp devs and all translators. """)

# ── 反向代理 SSE 修正 ─────────────────────────────────────────────────
try:
    @app.app.middleware("http")
    async def _no_accel_buffer(request, call_next):
        response = await call_next(request)
        response.headers["X-Accel-Buffering"] = "no"
        return response
except Exception as _e:
    print(f"[warn] could not install no-buffering middleware: {_e}")

app.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    parser = ArgumentParser(description="Separate audio into multiple stems")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--listen-port", type=int, default=6138)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="強制使用 CPU 推論")
    parser.add_argument("--no-cpu-fallback", action="store_true",
                        help="永不回退至 CPU；失敗時直接拋出錯誤")
    parser.add_argument("--in-process", action="store_true",
                        help="停用預設的隔離 worker，改為在父行程內執行")
    parser.add_argument("--target-sr", type=int, default=None,
                        help=f"所有輸出的最低取樣率（預設 {TARGET_SR}）")
    parser.add_argument("--gpu-unlock-profile", action="store_true",
                        help="選用 Linux SM 8.6 的 RoFormer/MDX23C 原生 AMP；"
                             "實際硬體解鎖需另外安裝 gpu-unlock；"
                             "明確設定的 UVR_AUTOCAST 優先於此設定")
    args = parser.parse_args()
    if args.gpu_unlock_profile:
        GPU_UNLOCK_PROFILE = True
    if GPU_UNLOCK_PROFILE:
        print("[init] 已請求 gpu-unlock AMP 設定檔：符合條件的 CUDA RoFormer/MDX23C "
              "會向 audio-separator 請求原生 autocast；CPU 與其他架構維持原有策略。")
        if AUTOCAST_EXPLICIT:
            print(f"[init] 明確的 UVR_AUTOCAST 覆寫此設定檔：{USE_AUTOCAST}")
        print("[init] 本程式不驗證 kernel 解鎖。啟動 NVIDIA 工作負載前，"
              "請先於主機安裝並驗證 Zagitta/gpu-unlock。"
              "TF32 維持關閉；cuBLAS 穩定性設定不變。")
    if args.cpu:
        FORCE_CPU = True
        print("[init] --cpu 已指定：強制使用 CPU 推論。")
    if args.no_cpu_fallback:
        ALLOW_CPU_FALLBACK = False
        print("[init] 已停用 CPU 回退 — GPU 失敗時將直接拋出錯誤。")
    if args.in_process:
        ALWAYS_SUBPROCESS = False
        print("[init] --in-process：已停用隔離 worker（僅作為回退）。")
    if args.target_sr and args.target_sr >= 8000:
        TARGET_SR = int(args.target_sr)
        print(f"[init] 最低輸出取樣率設為 {TARGET_SR} Hz。")
    if ALWAYS_SUBPROCESS and not SUBPROC_FALLBACK:
        ALWAYS_SUBPROCESS = False
        print("[warn] worker 腳本無法使用 — 回退至父行程內執行。")

    print(f"[init] {_gpu_unlock_profile_state_text()}")
    if _should_suggest_gpu_unlock():
        print("[init] 💡 提示：偵測到 SM 8.6 GPU，若已安裝 gpu-unlock kernel module，"
              "可使用 --gpu-unlock-profile 或設定 UVR_GPU_UNLOCK_PROFILE=1，"
              "以對 RoFormer/MDX23C 取得 AMP 加速。")

    print(f"[init] policy: gpu_first={not FORCE_CPU and GPU_PRESENT} "
          f"| isolated_worker_default={ALWAYS_SUBPROCESS} "
          f"| worker_available={SUBPROC_FALLBACK} "
          f"| cpu_fallback={ALLOW_CPU_FALLBACK} "
          f"| model_cache={MODEL_CACHE_ENABLED} "
          f"| autocast={USE_AUTOCAST} "
          f"| gpu_unlock_profile={GPU_UNLOCK_PROFILE} "
          f"| target_sr={TARGET_SR}")
    try:
        app.launch(
            share=args.share,
            favicon_path="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
            server_name="0.0.0.0",
            server_port=args.listen_port,
            inbrowser=args.open,
            show_error=True,
            allowed_paths=[os.path.abspath(TEMP_ROOT)],
            max_file_size=os.environ.get("UVR_MAX_FILE_SIZE", "10gb"),
        )
    finally:
        try:
            MODEL_CACHE.clear()
        except Exception:
            pass
