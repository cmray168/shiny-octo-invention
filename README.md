# 🎵 UVR5 UI — GPU-First Hardened Build 🎵

> This README documents **this fork** of [Eddycrack864/UVR5-UI](https://github.com/Eddycrack864/UVR5-UI).
> The separation engine and the original UI concept are unchanged — what's different is
> the **execution model** (isolated worker by default), the **hardening around CUDA / cuBLAS**,
> the **VRAM-aware retry ladder**, and the **optional GPU-unlock AMP profile**.

<div align="center">

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Eddycrack864/UVR5-UI)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Eddycrack864/UVR5-UI/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Community-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/aihub)

</div>

---

## What's different in this build

The following points describe behaviour that is **not** in the upstream repository (or is
materially changed). Everything else (model list, separation math, UI layout) tracks
upstream — the hardened build only changes *how* work is executed and how failures are
recovered from.

### 1. Isolated worker by default (CUDA-safe)

Every separation runs in a **throwaway subprocess** (`_uvr_worker.py`) by default. This means:

- **A CUDA fault in the worker cannot corrupt the parent process's context.**
  Illegal memory access, `CUBLAS_STATUS_EXECUTION_FAILED`, `device-side assert` etc.
  kill the worker, not the app.
- **VRAM is fully reclaimed when the worker exits.** The OS reclaims 100% of the worker's
  VRAM on exit, so a bad run never leaves the parent stuck in an "OOM forever" state.
- Opt out with `--in-process` or `UVR_ALWAYS_SUBPROCESS=0` to restore the classic
  single-process behaviour (useful for debugging).

### 2. VRAM-aware retry ladder

For each separation, the planner picks a **sequence of attempts** based on the detected
VRAM. The pattern is:

```
full segment  →  ½ segment  →  ¼ segment  →  isolated CPU worker
```

- Duplicate attempts are skipped (no wasted VRAM churn).
- Before the first attempt, the planner **clamps the model-default segment size** to
  something sane for the detected VRAM (< 6 GB → 128, < 8 GB → 256, < 12 GB → 512, < 16 GB → 512).
- **Explicit user overrides above the cap are honoured with a warning** instead of being
  silently downgraded — you can push an RTX 3060 12 GB to `segment=768` if you want.
- The retry pause is **interruptible** so shutting the app down doesn't wait for the ladder.

### 3. cuBLAS stability hardening (set before `import torch`)

```
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DISABLE_ADDMM_CUDA_LT  = "1"
PYTORCH_CUDA_ALLOC_CONF= expandable_segments:True,garbage_collection_threshold:0.8
```

TF32 is forced **off** for `matmul` and `cudnn`, and `preferred_blas_library="cublas"` is
set when supported. This eliminates the most common `CUBLAS_STATUS_EXECUTION_FAILED` and
`illegal memory access` failures caused by cuBLASLt on 12 GB cards.

### 4. Optional GPU-unlock AMP profile (opt-in, Linux SM 8.6)

> ⚠️ **This does not unlock your GPU.** It only asks `audio-separator` to request native
> AMP autocast for RoFormer / MDX23C **on Linux, on SM 8.6 hardware** (RTX 30-series).
> The actual unlock is delivered by the separate
> [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, which you
> must install and verify yourself. This app **never** loads kernel modules, touches the
> driver, or changes your system configuration.

Enable with either:

```bash
./run-UVR5-UI.sh --gpu-unlock-profile
# or
UVR_GPU_UNLOCK_PROFILE=1 ./run-UVR5-UI.sh
```

An explicit `UVR_AUTOCAST=0` / `UVR_AUTOCAST=1` **overrides** this profile. The startup
banner and the **GPU / Memory** tab both report the current state. If the profile is off
but an SM 8.6 GPU is detected, the app prints a one-line hint.

### 5. Target sample rate enforcement

Every produced stem is guaranteed to be **≥ `UVR_TARGET_SR`** (default `44100`). Low-SR
inputs are upsampled with `soxr` (`aresample=resampler=soxr:precision=28`) via ffmpeg;
lossless fallback uses `librosa.resample(res_type="soxr_hq")`. This includes
outputs from the ensemble combiner and the pitch-shift tab.

### 6. 4-slot Ensemble tab

Four independent model slots, each with its own model picker, weight slider, and per-stem
preview. Per-slot controls (`segment`, `batch`, `format`) feed every slot; the combine
button mixes the selected stems into a single output with normalized weights.

Each slot's per-`Stem N` label shows the **real** sample rate, so you can see at a glance
which stems came in below 44.1 kHz and were upsampled.

### 7. GPU / Memory tab

Read-only status panel + the safe recovery actions:

- **Free VRAM** — unloads the cached model, runs `empty_cache`, `ipc_collect`, resets
  peak/accumulated stats, and gc's.
- **Re-test / recover GPU** — re-probes the parent-process context without
  touching a running worker.
- **Purge temp files** — removes `temp_outputs/*` except folders currently pinned by a
  live job.

All actions respect `_separator_lock` so they can't race with a separation.

### 8. Model cache + idle reaper

Off by default. Enable with `UVR_MODEL_CACHE=1`. When on, the loaded model is reused across
runs and released after `UVR_MODEL_IDLE_TIMEOUT` seconds (default `900`). The
`--in-process` mode is the main beneficiary; the isolated-worker path is intentionally
cache-free (worker lifetime = one separation).

### 9. Client-disconnect filter and SSE un-buffering

- `ClientDisconnect` is filtered out of `uvicorn.error` (benign browser aborts).
- `X-Accel-Buffering: no` middleware is installed so reverse proxies don't buffer the
  streaming progress events.
- `FORWARDED_ALLOW_IPS=*` and `GRADIO_HEARTBEAT_INTERVAL=8` are set by default for
  reverse-proxy deployments.

### 10. Gradio 4/5 compatibility fixes

- `_update()` returns a proper `{"__type__": "update", ...}` dict when `gr.update` is
  missing (Gradio 5), instead of constructing a new component instance.
- `_empty_slot_outputs()` no longer accidentally deletes temp directories.
- Streaming `Status` panels render with `\n` (not ` | `) so multiline state stays readable.

---

## Features

- User-friendly interface (Gradio)
- All VR Arch models
- All MDX-NET models
- Demucs v4 models
- MDX23C models
- Mel-Band Roformer models
- BS Roformer models
- Music source separation models
- VIP models
- **4-slot ensemble with weighted stem mixing**
- **Pitch changer** (Rubber Band, fractional semitones, formant preservation)
- Separation of audio/video from any site supported by [yt-dlp](https://github.com/yt-dlp/yt-dlp)
  (with **JS-runtime auto-discovery**: node / deno / bun / quickjs)
- **Batch separation** (in-process fallback path)
- **Multi-language UI** via `assets.i18n`
- **Theme selector**
- **Leaderboard** (read-only SDR listing)
- **GPU / Memory management tab**
- **Target sample rate enforcement** on every output
- Colab / Kaggle / Lightning.ai / Docker support
- Windows / Linux support

---

## Requirements

### Hardware

- **NVIDIA RTX 20-series or newer** (SM 7.5+). SM 8.6 (RTX 30) is the sweet spot for
  this build's hardening work.
- At least **10 GB free disk** for models + output. Add more if you plan to run large
  Roformer models.
- **≥ 12 GB VRAM** recommended for Roformer work. 8 GB works with reduced segment
  size; 6 GB and below will fall into the ½ / ¼ segment rungs.

> [!NOTE]
> Older GPUs (SM < 7.5) will run, but slowly. CPU is orders of magnitude slower — if you
> don't meet the hardware requirements, use the [Colab](https://colab.research.google.com/github/Eddycrack864/UVR5-UI/blob/main/UVR_UI.ipynb),
> [Kaggle](https://www.kaggle.com/code/eddycrack864/uvr5-ui),
> [Lightning.ai](https://lightning.ai/eddycrack864/studios/uvr5-ui) or
> [Hugging Face](https://huggingface.co/spaces/TheStinger/UVR5_UI) spaces.

### Software

- **Python 3.10+** (3.11 recommended)
- **PyTorch** matching your CUDA runtime (see `nvidia-smi` output)
- **Git**
- **FFmpeg** on `PATH` (required — audio extraction, upsampling, encoder fallback)
  - Windows: use the [automated installer](https://github.com/Eddycrack864/UVR5-UI/blob/main/info/ffmpeg-installer.bat)
    or download from [ffmpeg.org](https://www.ffmpeg.org/download.html)
  - Debian/Ubuntu: `sudo apt install ffmpeg git`
  - Arch: `sudo pacman -S ffmpeg git`
  - Fedora: `sudo dnf install ffmpeg git`
- **Rubber Band CLI** for the pitch changer: `sudo apt install rubberband-cli` (or
  `brew install rubberband`, or download the Windows binary from `breakfastquay.com`)

> [!IMPORTANT]
> FFmpeg **must** be on the PATH. On Windows, restart the terminal after adding it.

---

## Getting Started

```bash
git clone https://github.com/Eddycrack864/UVR5-UI.git
cd UVR5-UI
```

### 1. Installation

- **Windows:** double-click `UVR5-UI-installer.bat` (do **not** run as administrator).
- **Linux:** `chmod +x UVR5-UI-installer.sh && ./UVR5-UI-installer.sh`

### 2. Running

- **Windows:** double-click `run-UVR5-UI.bat`
- **Linux:** `chmod +x run-UVR5-UI.sh && ./run-UVR5-UI.sh`

Default address: **`http://0.0.0.0:6138`**

### 3. Updating

- **Windows:** `UVR5-UI-updater.bat`
- **Linux:** `chmod +x UVR5-UI-updater.sh && ./UVR5-UI-updater.sh`

If something fails, check [troubleshooting.md](https://github.com/Eddycrack864/UVR5-UI/blob/main/info/troubleshooting.md)
first, then open an issue.

---

## Command-line options (this build)

`app.py` accepts the following flags. They override the environment variables below.

| Flag | Effect |
|------|--------|
| `--share` | Expose a public Gradio share link. |
| `--listen-port N` | Port for the Gradio server (default `6138`). |
| `--open` | Open the browser automatically. |
| `--cpu` | Force CPU inference (`UVR_FORCE_CPU=1`). The GPU tab will still show hardware info but will not touch the CUDA runtime. |
| `--no-cpu-fallback` | If a GPU separation fails, raise instead of falling back to CPU. |
| `--in-process` | Disable the isolated worker; run separations in the parent process (classic behaviour). |
| `--target-sr N` | Minimum output sample rate (default `44100`). |
| `--gpu-unlock-profile` | Enable the optional GPU-unlock AMP profile (Linux / SM 8.6 / RoFormer + MDX23C only). Does **not** install or load the kernel module. |

> [!TIP]
> If you hit repeated `CUBLAS_STATUS_EXECUTION_FAILED` errors, **lower the segment size
> first** (1024 → 512 → 256 → 128), keep batch size at `1`, and verify that your installed
> `torch` matches the CUDA runtime reported by `nvidia-smi`. The retry ladder already does
> this automatically, but a manual first attempt avoids the trial-and-error.

---

## Environment variables

All defaults are chosen for **stability on 12 GB GPUs** — override only if you know why.

### Execution / GPU

| Variable | Default | Meaning |
|----------|---------|---------|
| `UVR_FORCE_CPU` | `0` | Force CPU. Same as `--cpu`. |
| `UVR_ALLOW_CPU_FALLBACK` | `1` | Allow the final rung of the retry ladder to be CPU. |
| `UVR_ALWAYS_SUBPROCESS` | `1` | Isolated worker is the default execution mode. `0` = in-process. |
| `UVR_SUBPROCESS_FALLBACK` | `1` | Fall back to subprocess when the worker script is available. |
| `UVR_MODEL_CACHE` | `0` | Reuse a loaded model across runs (only meaningful in-process). |
| `UVR_MODEL_IDLE_TIMEOUT` | `900` | Seconds of idle before the cache is released. |
| `UVR_CUDNN_BENCHMARK` | `0` | Enable `cudnn.benchmark` (variance vs. mean timing trade-off). |
| `UVR_AUTOCAST` | `0` | Explicit AMP override. If set, **overrides** the GPU-unlock profile. |
| `UVR_GPU_UNLOCK_PROFILE` | `0` | Request native AMP for RoFormer/MDX23C on Linux SM 8.6. Does **not** unlock the GPU. |

### Timeouts

| Variable | Default | Meaning |
|----------|---------|---------|
| `UVR_SUBPROC_TIMEOUT` | `5400` | Wall-clock timeout for one isolated worker. |
| `UVR_INPROC_TIMEOUT` | `1800` | Wall-clock timeout for one in-process separation. |
| `UVR_MODEL_LOAD_TIMEOUT` | `900` | Timeout for `load_model()`. |

### Output / Temp / Files

| Variable | Default | Meaning |
|----------|---------|---------|
| `UVR_TARGET_SR` | `44100` | Minimum output sample rate. |
| `UVR_TEMP_TTL` | `10800` | Temp folders older than this are purged by the janitor. |
| `UVR_MAX_FILE_SIZE` | `10gb` | Gradio upload limit. |

### Memory / CUDA

| Variable | Default | Meaning |
|----------|---------|---------|
| `UVR_EXPANDABLE_SEGMENTS` | `1` | Use `expandable_segments:True` in the allocator. |
| `UVR_ALLOC_CONF` | `garbage_collection_threshold:0.8` | Fallback allocator config when `UVR_EXPANDABLE_SEGMENTS=0`. |
| `UVR_CUDA_DEBUG` | `0` | `CUDA_LAUNCH_BLOCKING=1`. **Very slow** — debug only. |

### Logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `UVR_LOG_CLIENT_DISCONNECT` | `0` | Suppress the benign "ClientDisconnect" traceback from `uvicorn.error`. Set to `1` to log it. |
| `UVR_LOG_ONLY` | `0` | Disable `gr.Progress` output. |
| `UVR_CLI_MIN_DELTA` / `UVR_CLI_MIN_INTERVAL` | `0.04` / `0.9` | Throttling for the CLI progress wrapper. |

---

## GPU unlock profile — step by step

> ⚠️ **Read this whole section before enabling the profile.** Enabling the profile without
> actually installing the kernel module does **not** harm your system, but it also does
> **not** accelerate anything. The profile is a *request*; only the kernel module
> delivers the actual unlock.

1. Confirm your GPU is **SM 8.6** (RTX 3060 / 3060 Ti / 3070 / 3080 / 3090):
   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```
2. Install [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on the host
   (kernel module, `modprobe`, persistent configuration, etc.) and verify it works on
   its own using the project's own test suite.
3. Reboot / re-load the module.
4. Start this app with `--gpu-unlock-profile` (or `UVR_GPU_UNLOCK_PROFILE=1`).
5. Check the **GPU / Memory** tab — it should say
   `GPU 解鎖設定檔：已啟用｜目標裝置 8.6｜目前裝置 SM 8.6`.
6. Run a RoFormer or MDX23C separation; watch VRAM and throughput in `nvidia-smi -l 1`.

**What this build does *not* do:** it does not load kernel modules, modify drivers, or
patch any binary. If you have `UVR_AUTOCAST` explicitly set (`=0` or `=1`), the profile
is bypassed and your explicit value wins.

---

## Precompiled builds

Grab the zip for your platform:

- **[Windows](https://huggingface.co/Eddycrack864/UVR5-UI/tree/main/Windows)**
- **[Linux](https://huggingface.co/Eddycrack864/UVR5-UI/tree/main/Linux)**

Extract ("extract here" recommended) and use all normal features. You still need Git to
update.

A Chinese-oriented precompiled build by **jian27** (2.6 GB) is available
[here](https://www.jian27.com/html/2231.html).

---

## Docker / Jupyter

Advanced setup. Use the
[Jupyter Notebook](https://github.com/Eddycrack864/UVR5-UI/blob/main/UVR_UI_Jupyter.ipynb)
(by **iroaK**) to bootstrap UVR5 inside a GPU-enabled container.

Requirements/recommendations:

- Image `ubuntu/ubuntu:20.04` or newer
- ≥ 20 GB storage (more if you'll keep models)
- Jupyter ≥ 7.3.1
- Forward port `6138` for the UI (note: **not** 9999 — upstream docs are stale here;
  `--listen-port 9999` will change it if you want the old port)
- NVIDIA drivers + `nvidia-container-toolkit` installed

---

## Contributions

Issues and PRs welcome:

- [Open an issue](https://github.com/Eddycrack864/UVR5-UI/issues)
- [Open a pull request](https://github.com/Eddycrack864/UVR5-UI/pulls)

If you like the project, star the repo 💖

Donate to upstream UVR5:
[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/uvr5)

---

## TO-DO

- More models
- More output formats
- Structured logging for the worker (JSON, not markers)
- First-class support for `--in-process` model cache across ensemble slots

---

## Credits

- **python-audio-separator** by [beveradb](https://github.com/beveradb)
- **Ilaria** ([TheStingerX](https://github.com/TheStingerX)) — HF Space hosting and help 💖
- **Mikus** ([cappuch](https://github.com/cappuch)) — code help
- **Nick088** ([Nick088Official](https://github.com/Nick088Official)) — Roformer fixes
- **yt_dlp** devs — [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- **NeoDev** ([TheNeodev](https://github.com/TheNeodev)) — URL separation source and improvements
- **ArisDev** ([aris-py](https://github.com/aris-py)) — Kaggle port and improvements
- **Zagitta** — [gpu-unlock](https://github.com/Zagitta/gpu-unlock) (optional, external)

The hardening work in this build (isolated worker, retry ladder, cuBLAS fixes,
GPU-unlock profile plumbing, sample-rate enforcement, GPU/Memory tab) is layered on top of
upstream UVR5-UI without changing the underlying separation math.
