<h1 align="center"><b> 🎵 UVR5 UI — GPU-First Hardened Build 🎵 </b></h1>

<div align="center">

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Eddycrack864/UVR5-UI)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Eddycrack864/UVR5-UI/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Community-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/aihub)

</div>

> This README documents **this fork** of [Eddycrack864/UVR5-UI](https://github.com/Eddycrack864/UVR5-UI).
> The separation engine and the original UI concept are unchanged — what's different is the
> **execution model** (isolated worker by default), the **hardening around CUDA / cuBLAS**, the
> **VRAM-aware retry ladder**, the **optional GPU-unlock AMP profile**, a **properly documented
> Pitch Changer**, and a **modern Python 3.13 / CUDA 12.8 / Gradio 6 frozen dependency set**.

---

## What's different in this build

The following points describe behaviour that is **not** in the upstream repository (or is
materially changed). Everything else (model list, separation math, UI layout) tracks upstream —
the hardened build only changes *how* work is executed and how failures are recovered from.

### 1. Isolated worker by default (CUDA-safe)

Every separation runs in a **throwaway subprocess** (`_uvr_worker.py`) by default. This means:

- **A CUDA fault in the worker cannot corrupt the parent process's context.**
  Illegal memory access, `CUBLAS_STATUS_EXECUTION_FAILED`, `device-side assert`, etc. kill the
  worker, not the app.
- **VRAM is fully reclaimed when the worker exits.** The OS reclaims 100% of the worker's VRAM
  on exit, so a bad run never leaves the parent stuck in an "OOM forever" state.
- Opt out with `--in-process` or `UVR_ALWAYS_SUBPROCESS=0` to restore classic single-process
  behaviour (useful for debugging).

### 2. VRAM-aware retry ladder

For each separation the planner picks a sequence of attempts based on detected VRAM:

```
full segment  →  ½ segment  →  ¼ segment  →  isolated CPU worker
```

- Duplicate attempts are skipped (no wasted VRAM churn).
- Before the first attempt, the planner **clamps the model-default segment size** to something
  sane for the detected VRAM (< 6 GB → 128, < 8 GB → 256, < 12 GB → 512, < 16 GB → 512).
- **Explicit user overrides above the cap are honoured with a warning** instead of being silently
  downgraded — you can push an RTX 3060 12 GB to `segment=768` if you want.
- The retry pause is **interruptible** so shutting the app down doesn't wait for the ladder.

### 3. cuBLAS stability hardening (set before `import torch`)

```
CUBLAS_WORKSPACE_CONFIG  = ":4096:8"
DISABLE_ADDMM_CUDA_LT    = "1"
PYTORCH_CUDA_ALLOC_CONF  = expandable_segments:True,garbage_collection_threshold:0.8
```

TF32 is forced **off** for `matmul` and `cudnn`, and `preferred_blas_library="cublas"` is set
when supported. This eliminates the most common `CUBLAS_STATUS_EXECUTION_FAILED` and
`illegal memory access` failures caused by cuBLASLt on 12 GB cards.

### 4. Optional GPU-unlock AMP profile (opt-in, Linux SM 8.6)

> ⚠️ **This does not unlock your GPU.** It only asks `audio-separator` to request native AMP
> autocast for RoFormer / MDX23C **on Linux, on SM 8.6 hardware** (RTX 30-series). The actual
> unlock is delivered by the separate
> [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) kernel module, which you must
> install and verify yourself. This app **never** loads kernel modules, touches the driver, or
> changes your system configuration.

Enable with either:

```bash
python app.py --gpu-unlock-profile
# or
UVR_GPU_UNLOCK_PROFILE=1 python app.py
```

An explicit `UVR_AUTOCAST=0` / `UVR_AUTOCAST=1` **overrides** this profile. The startup banner
and the **GPU / Memory** tab both report the current state. If the profile is off but an SM 8.6
GPU is detected, the app prints a one-line hint.

> **Known UI quirk:** the GPU tab's *initial* render happens before CLI args are parsed, so it
> may show `未啟用` on first paint even when `--gpu-unlock-profile` is active. **Click 🔄 Refresh
> once** to see the correct state. Runtime behaviour is correct either way — the worker log
> (`[gpu-unlock] worker 請求 AMP 設定檔`) is the source of truth.

### 5. Target sample rate enforcement (applies to every output)

Every produced stem — including **pitch-shifted files** and **ensemble combiners** — is
guaranteed to be **≥ `UVR_TARGET_SR`** (default `44100`). Low-SR inputs are upsampled with `soxr`
(`aresample=resampler=soxr:precision=28`) via ffmpeg; the lossless fallback uses
`librosa.resample(res_type="soxr_hq")`.

### 6. Pitch Changer (Rubber Band, with target-SR guarantee)

A dedicated tab for **pitch shifting without changing speed** — useful for transposing a track
to a singer's key, fixing a recording that drifted a few cents, or preparing a karaoke backing
track in a specific key.

- **Fractional semitones.** The pitch is set either by the `-12 … +12` slider (0.05 step) or by
  typing an exact value into the **Exact pitch** number field (4 decimal places). Both stay in
  sync — moving the slider updates the number, and vice versa.
- **Formant preservation** (default **on**). Rubber Band's `--formant` keeps the vocal timbre
  natural when shifting male ↔ female ranges, instead of the chipmunk/demon effect.
- **Quality 1–5** (default **5**). Quality ≥ 4 adds `--pitch-hq --fine` to the Rubber Band
  command; quality 3 uses `--pitch-hq` alone; quality < 3 uses the fast path.
- **Input is auto-converted.** The same `smart_media_ingest` pipeline as the Ensemble tab handles
  video containers and low-SR files, so you can drop in an MP4 or a 22 kHz WAV and it just works.
- **URL download** is built into the tab — paste a YouTube / SoundCloud link and the audio is
  fetched to the input field.
- **Output is always 24-bit WAV at ≥ `UVR_TARGET_SR`.** Never a downsample, never a lossy
  intermediate. Result filename: `<stem>_pitch_<+1.5|-2>.wav`.
- **Requires the Rubber Band CLI** on `PATH` (see [System dependencies](#system-dependencies)).
  If it's missing, the tab raises a clear error instead of failing silently.

Internally it: loads via `librosa`, resamples to the target rate if needed, writes a temporary
FLOAT WAV, invokes `rubberband`, reads the result back, writes a 24-bit PCM output, and cleans up
the temp files. Long files show a heartbeat progress bar so the UI doesn't look frozen while
Rubber Band is churning.

### 7. 4-slot Ensemble tab

Four independent model slots, each with its own model picker, weight slider, and per-stem
preview. Per-slot controls (`segment`, `batch`, `format`) feed every slot; the combine button
mixes the selected stems into a single output with normalized weights.

Each slot's per-`Stem N` label shows the **real** sample rate, so you can see at a glance which
stems came in below 44.1 kHz and were upsampled.

### 8. GPU / Memory tab

Read-only status panel + the safe recovery actions:

- **Free VRAM** — unloads the cached model, runs `empty_cache`, `ipc_collect`, resets
  peak/accumulated stats, and gc's.
- **Re-test / recover GPU** — re-probes the parent-process context without touching a running
  worker.
- **Purge temp files** — removes `temp_outputs/*` except folders currently pinned by a live job.

All actions respect `_separator_lock` so they can't race with a separation.

### 9. Model cache + idle reaper

Off by default. Enable with `UVR_MODEL_CACHE=1`. When on, the loaded model is reused across runs
and released after `UVR_MODEL_IDLE_TIMEOUT` seconds (default `900`). The `--in-process` mode is
the main beneficiary; the isolated-worker path is intentionally cache-free (worker lifetime = one
separation).

### 10. Client-disconnect filter and SSE un-buffering

- `ClientDisconnect` is filtered out of `uvicorn.error` (benign browser aborts).
- `X-Accel-Buffering: no` middleware is installed so reverse proxies don't buffer the streaming
  progress events.
- `FORWARDED_ALLOW_IPS=*` and `GRADIO_HEARTBEAT_INTERVAL=8` are set by default for
  reverse-proxy deployments.

### 11. Gradio 4/5 compatibility shims (still in tree, no-op on Gradio 6)

- `_update()` returns a proper `{"__type__": "update", ...}` dict when `gr.update` is missing,
  instead of constructing a new component instance.
- `_empty_slot_outputs()` no longer accidentally deletes temp directories.
- Streaming `Status` panels render with `\n` (not ` | `) so multiline state stays readable.

On Gradio 6.22 these shims are inert — `gr.update` still exists. If Gradio ever removes it, the
fallback path fires and every Radio / Audio / Markdown update path in the UI needs rework.

---

## Features

- User-friendly Gradio interface
- All VR Arch models
- All MDX-NET models
- Demucs v4 models
- MDX23C models
- Mel-Band Roformer models
- BS Roformer models
- Music source separation models
- VIP models
- **4-slot ensemble with weighted stem mixing**
- **Pitch Changer** — Rubber Band engine, fractional semitones, formant preservation,
  quality 1–5, URL download built in, output always 24-bit ≥ 44.1 kHz
- Separation of audio/video from any site supported by
  [yt-dlp](https://github.com/yt-dlp/yt-dlp) — with **JS-runtime auto-discovery**
  (Node / Deno / Bun / QuickJS; **Deno is bundled** in the frozen stack)
- Batch separation (in-process fallback path)
- Multi-language UI via `assets.i18n`
- Theme selector
- Leaderboard (read-only SDR listing)
- **GPU / Memory management tab**
- **Target sample rate enforcement** on every output (stems, pitch, combine)
- Colab / Kaggle / Lightning.ai / Docker support
- Windows / Linux support

---

## Requirements

> The pins below are the **verified frozen stack** (`requirements-frozen.txt`), generated with
> `pip freeze` from a working env. Deviate and you own the consequences — the Gradio 6.22 /
> torch 2.7.1 / onnxruntime-gpu 1.28 combination is what `app.py` was tested against.

### Hardware

- **NVIDIA RTX 20-series or newer** (SM 7.5+). SM 8.6 (RTX 30) is the sweet spot for this
  build's hardening work.
- **CUDA 12.8-capable driver** (`≥ 570.xx`).
- At least **10 GB free disk** for models + output. Add more for large Roformer models.
- **≥ 12 GB VRAM** recommended for Roformer work. 8 GB works with reduced segment size; 6 GB
  and below will drop into the ½ / ¼ segment rungs automatically.

> [!NOTE]
> Older GPUs (SM < 7.5) will run, but slowly. CPU is orders of magnitude slower — if you don't
> meet the hardware requirements, use Colab / Kaggle / Lightning.ai / HF Spaces.

### Software — verified frozen stack

| Component | Version | Notes |
|---|---|---|
| Python | **3.13** | `onnx2torch-py313==1.6.0` in the freeze assumes CPython 3.13. 3.11 untested. |
| PyTorch | **`torch==2.7.1+cu128`** | CUDA 12.8 wheels. Requires `--extra-index-url https://download.pytorch.org/whl/cu128`. |
| torchaudio | **`torchaudio==2.7.1+cu128`** | ✅ Matches torch minor. Do **not** bump independently. |
| torchvision | **`torchvision==0.22.1+cu128`** | Aligns with torch 2.7.1. |
| Gradio | **`gradio==6.22.0`** | `_update()` / `_skip()` are Gradio 4/5 compat shims; no-ops on 6.22. |
| audio-separator | **`audio-separator==0.44.5`** | The `make_separator()` kwarg-stripping loop tolerates minor 0.4x drifts. |
| onnxruntime | **`onnxruntime-gpu==1.28.0`** | ✅ GPU build — MDX-NET models use `CUDAExecutionProvider`. `_probe_onnx_cuda()` reports availability at startup. |
| onnx | **`onnx==1.22.0`** + **`onnx-weekly==1.23.0.dev20260727`** | `onnx-weekly` is a **dev wheel**. If MDX-NET misbehaves, pin stable `onnx==1.17`. |
| numpy / scipy | **`2.4.6` / `1.17.1`** | |
| librosa | **`0.11.0`** | |
| yt-dlp | **`yt-dlp==2026.7.4`** + **`yt-dlp-ejs==0.8.0`** | EJS handles YouTube's remote JS components without a separate install. |
| Deno | **`deno==2.9.5`** | Bundled — `_find_js_runtimes()` picks it up automatically. No separate Node install needed. |

### System dependencies

- **FFmpeg** on `PATH` (required — audio extraction, upsampling, encoder fallback)
  - Debian/Ubuntu: `sudo apt install ffmpeg git`
  - Arch: `sudo pacman -S ffmpeg git`
  - Fedora: `sudo dnf install ffmpeg git`
  - Windows: [ffmpeg.org](https://www.ffmpeg.org/download.html) or the
    [automated installer](https://github.com/Eddycrack864/UVR5-UI/blob/main/info/ffmpeg-installer.bat)
- **Rubber Band CLI** — required for the **Pitch Changer** tab
  - Debian/Ubuntu: `sudo apt install rubberband-cli`
  - Arch: `sudo pacman -S rubberband`
  - Fedora: `sudo dnf install rubberband`
  - macOS: `brew install rubberband`
  - Windows: binary from [breakfastquay.com](https://breakfastquay.com/rubberband/)

> [!IMPORTANT]
> FFmpeg **must** be on the PATH. On Windows, restart the terminal after adding it.
> Without Rubber Band, the Pitch Changer tab raises a clear error when you click **Pitch Shift**
> — the rest of the app is unaffected.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Eddycrack864/UVR5-UI.git
cd UVR5-UI
```

### 1. Installation

Create a Python 3.13 venv and install the frozen stack:

```bash
python3.13 -m venv env
source env/bin/activate           # bash / zsh
# source env/bin/activate.fish    # fish
# .\env\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements-frozen.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

The `--extra-index-url` is **required** — the `+cu128` local version suffix does not exist on
default PyPI.

> The legacy `UVR5-UI-installer.bat` / `UVR5-UI-installer.sh` scripts still work but install the
> older upstream pins. **Use `requirements-frozen.txt` for this build.**

### 2. Running

**Recommended — from an activated venv** (works in every shell, and gives you a clean traceback):

```bash
# bash / zsh
source env/bin/activate
python app.py

# fish
source env/bin/activate.fish
python app.py
```

With the GPU-unlock profile enabled (Linux SM 8.6 only — see [§GPU unlock](#gpu-unlock-profile--step-by-step)):

```bash
python app.py --gpu-unlock-profile
```

Other common invocations:

```bash
python app.py --open                    # open the browser automatically
python app.py --share                   # public Gradio share link
python app.py --listen-port 6138        # default port is 6138
python app.py --cpu                     # force CPU inference
python app.py --in-process              # disable the isolated worker (debugging)
```

Default address: **`http://0.0.0.0:6138`**

**Optional convenience wrappers** — `run-UVR5-UI.bat` (Windows) and `run-UVR5-UI.sh` (Linux) are
thin wrappers around `cd` + `source env/bin/activate` + `python app.py "$@"`. If you already
activate the venv manually, they add nothing. Use whichever you prefer.

### 3. Updating

```bash
git pull
pip install -r requirements-frozen.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

`UVR5-UI-updater.bat` / `.sh` do the same thing but assume the upstream `requirements.txt`. If
you use them, re-run `pip install -r requirements-frozen.txt --extra-index-url ...` afterwards.

If something fails, check [troubleshooting.md](https://github.com/Eddycrack864/UVR5-UI/blob/main/info/troubleshooting.md)
first, then open an issue.

---

## Using the Pitch Changer

Open the **Pitch Changer** tab. The flow:

1. **Load audio** — drop a file into the input box, or paste a YouTube / SoundCloud link into the
   *…or fetch from a link* accordion and click **Download into the input field**. Video files are
   automatically stripped to audio; low-SR files are upsampled. The resolved audio appears in the
   preview player with its real sample rate and duration.
2. **Set the pitch** — drag the *Pitch (semitones)* slider (−12 … +12, 0.05 step) **or** type an
   exact value into *Exact pitch* (4 decimal places). They sync both ways. Positive = higher
   pitch, negative = lower.
   - Whole semitones for transposing keys: `+2` = up a whole tone, `-3` = down a minor third.
   - Fractional values for fine tuning: `+0.25` ≈ quarter semitone, `-1.5` = minor third down
     minus a quarter.
3. **Options** —
   - **Preserve Formant** (default ✅): keeps vocal timbre natural across large shifts. Turn it
     off if you *want* the chipmunk/demon effect.
   - **Quality** (1–5, default 5): 5 = best. Quality ≥ 4 adds `--pitch-hq --fine` to Rubber
     Band; quality 3 uses `--pitch-hq`; lower uses the fast path. On long files quality 5 can be
     slow — drop to 3 if you're iterating.
4. **Click Pitch Shift.** The result is a **24-bit WAV at ≥ 44.1 kHz** named
   `<stem>_pitch_<sign><value>.wav`, e.g. `vocals_pitch_+2.wav` or `song_pitch_-1.50.wav`.
5. **Preview / download** from the output player.

**Notes:**

- Pitch shifting is **speed-preserving**. If you want pitch *and* speed change, use a DAW — this
  tool is deliberately time-locked.
- Output format is fixed at 24-bit WAV; there is no format dropdown for this tab. Convert with
  ffmpeg if you need MP3 / FLAC.
- The result lands in a fresh `temp_outputs/<uuid>/` folder, so it does **not** overwrite your
  input.
- If you see `Rubber Band CLI not installed`, install `rubberband-cli` (Linux) / `rubberband`
  (macOS) / the Windows binary — see [System dependencies](#system-dependencies).

---

## Command-line options

`app.py` accepts the following flags. They override the environment variables below.

| Flag | Effect |
|------|--------|
| `--share` | Expose a public Gradio share link. |
| `--listen-port N` | Port for the Gradio server (default `6138`). |
| `--open` | Open the browser automatically. |
| `--cpu` | Force CPU inference (`UVR_FORCE_CPU=1`). The GPU tab still shows hardware info but does not touch the CUDA runtime. |
| `--no-cpu-fallback` | If a GPU separation fails, raise instead of falling back to CPU. |
| `--in-process` | Disable the isolated worker; run separations in the parent process (classic behaviour). |
| `--target-sr N` | Minimum output sample rate (default `44100`). Applies to stems, pitch, and combine output. |
| `--gpu-unlock-profile` | Enable the optional GPU-unlock AMP profile (Linux / SM 8.6 / RoFormer + MDX23C only). Does **not** install or load the kernel module. |

> [!TIP]
> If you hit repeated `CUBLAS_STATUS_EXECUTION_FAILED` errors, **lower the segment size first**
> (1024 → 512 → 256 → 128), keep batch size at `1`, and verify that your installed `torch`
> matches the CUDA runtime reported by `nvidia-smi`. The retry ladder already does this
> automatically, but a manual first attempt avoids the trial-and-error.

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
| `UVR_TARGET_SR` | `44100` | Minimum output sample rate (stems, pitch, combine). |
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
> actually installing the kernel module does **not** harm your system, but it also does **not**
> accelerate anything. The profile is a *request*; only the kernel module delivers the actual
> unlock.

1. Confirm your GPU is **SM 8.6** (RTX 3060 / 3060 Ti / 3070 / 3080 / 3090):

   ```bash
   nvidia-smi --query-gpu=name,compute_cap --format=csv
   ```

2. Install [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on the host (kernel
   module, `modprobe`, persistent configuration, etc.) and verify it works on its own using the
   project's own test suite.

3. Reboot / re-load the module.

4. Start this app with `--gpu-unlock-profile` (or `UVR_GPU_UNLOCK_PROFILE=1`):

   ```bash
   source env/bin/activate.fish
   python app.py --gpu-unlock-profile
   ```

5. Click **🔄 Refresh** on the **GPU / Memory** tab — it should now say
   `GPU 解鎖設定檔：已啟用｜目標裝置 8.6｜目前裝置 SM 8.6`.

6. Run a RoFormer or MDX23C separation and watch VRAM / throughput with `nvidia-smi -l 1`. The
   worker log should include `[gpu-unlock] worker 請求 AMP 設定檔：arch=roformer`.

**What this build does *not* do:** it does not load kernel modules, modify drivers, or patch any
binary. If you have `UVR_AUTOCAST` explicitly set (`=0` or `=1`), the profile is bypassed and
your explicit value wins.

---

## Verifying your install

```bash
python -c "import torch, torchaudio, gradio, onnxruntime as ort; \
  print('torch', torch.__version__); \
  print('torchaudio', torchaudio.__version__); \
  print('gradio', gradio.__version__); \
  print('ort providers', ort.get_available_providers())"
```

Expected output:

```
torch 2.7.1+cu128
torchaudio 2.7.1+cu128
gradio 6.22.0
ort providers ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

If `CUDAExecutionProvider` is **missing**, you installed `onnxruntime` (CPU build) instead of
`onnxruntime-gpu`. Uninstall and reinstall:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-gpu==1.28.0
```

Confirm Rubber Band is on the PATH (needed for the Pitch Changer):

```bash
rubberband --help    # should print usage, not "command not found"
```

And confirm your venv's `python` is the one you froze with:

```bash
which python       # should print .../UVR5-UI/env/bin/python
python --version   # should print Python 3.13.x
```

---

## Precompiled builds

Grab the zip for your platform (upstream pins — reinstall `requirements-frozen.txt` on top for
this build's stack):

- **[Windows](https://huggingface.co/Eddycrack864/UVR5-UI/tree/main/Windows)**
- **[Linux](https://huggingface.co/Eddycrack864/UVR5-UI/tree/main/Linux)**

Extract ("extract here" recommended) and use all normal features. You still need Git to update.

A Chinese-oriented precompiled build by **jian27** (2.6 GB) is available
[here](https://www.jian27.com/html/2231.html).

---

## Docker / Jupyter

Advanced setup. Use the
[Jupyter Notebook](https://github.com/Eddycrack864/UVR5-UI/blob/main/UVR_UI_Jupyter.ipynb) (by
**iroaK**) to bootstrap UVR5 inside a GPU-enabled container.

Requirements / recommendations:

- Image `ubuntu/ubuntu:20.04` or newer with CUDA 12.8 support
- ≥ 20 GB storage (more if you'll keep models)
- Jupyter ≥ 7.3.1
- Forward port **6138** for the UI (upstream docs are stale here — `--listen-port 9999` restores
  the old port if you want it)
- NVIDIA drivers + `nvidia-container-toolkit` installed
- Install `rubberband-cli` inside the container if you want the Pitch Changer tab

---

## Known caveats on this stack

- **GPU tab initial text is stale.** The Gradio UI is built before CLI args are parsed, so the
  GPU / Memory tab shows `未啟用` on first paint even with `--gpu-unlock-profile`. **Click 🔄
  Refresh** to correct it. Runtime behaviour is correct regardless.
- **`onnx-weekly==1.23.0.dev*`** is a dev wheel — not reproducible across CI. Acceptable for a
  personal build; replace with stable `onnx==1.22.0` if you fork for distribution.
- **Gradio 6.22** — `app.py`'s `_update()` fallback (`{"__type__": "update", ...}`) is still in
  the tree from the Gradio 4/5 era. It should never fire on 6.22, but if you see
  `AttributeError: update` in the logs, that means Gradio removed `gr.update` and every
  Radio / Audio / Markdown update path in the UI needs rework.
- **`torchaudio` must stay in lockstep with `torch`.** Freeze files that drift these apart fail
  at import with a CUDA-compile-mismatch error. Your frozen file has them aligned (`2.7.1` ↔
  `2.7.1`) — keep it that way.
- **Pitch Changer needs Rubber Band.** Without `rubberband-cli` on the PATH, the **Pitch Shift**
  button raises `Rubber Band CLI not installed` — the rest of the app is unaffected.

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
- More output formats (Pitch Changer is currently 24-bit WAV only)
- Structured logging for the worker (JSON, not markers)
- First-class support for `--in-process` model cache across ensemble slots
- Fix the GPU-tab initial-render-before-argparse quirk

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
- **Breakfast Quay** — [Rubber Band](https://breakfastquay.com/rubberband/) pitch-shifting engine

The hardening work in this build (isolated worker, retry ladder, cuBLAS fixes, GPU-unlock profile
plumbing, sample-rate enforcement, GPU/Memory tab, Python 3.13 / CUDA 12.8 / Gradio 6 migration)
is layered on top of upstream UVR5-UI without changing the underlying separation math.
