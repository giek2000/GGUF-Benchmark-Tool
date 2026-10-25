# GGUF Benchmark Tool — User Manual

Version 2.4.1 · 2026 Created by Ricardo de Koning · MIT License

---

## Table of contents

1. [What this tool does](#1-what-this-tool-does)
2. [Installing and launching](#2-installing-and-launching)
3. [The main window at a glance](#3-the-main-window-at-a-glance)
4. [Step-by-step: your first benchmark](#4-step-by-step-your-first-benchmark)
5. [Getting the benchmark engine (CPU / GPU)](#5-getting-the-benchmark-engine-cpu--gpu)
6. [The Benchmark tab](#6-the-benchmark-tab)
7. [Reading the results](#7-reading-the-results)
8. [The Results History tab](#8-the-results-history-tab)
9. [The Model Info tab](#9-the-model-info-tab)
10. [The System tab](#10-the-system-tab)
11. [Where your files are saved](#11-where-your-files-are-saved)
12. [Troubleshooting](#12-troubleshooting)
13. [Frequently asked questions](#13-frequently-asked-questions)
14. [Glossary](#14-glossary)

---

## 1. What this tool does

A **GGUF** file is a packaged AI language model you can run on your own computer
with [llama.cpp](https://github.com/ggml-org/llama.cpp). This tool measures **how
fast** a given GGUF model runs on your hardware and settings, reporting the speed
in **tokens per second**.

Under the hood it runs llama.cpp's official benchmark program, `llama-bench`, and
presents it with a simple window: pick a model, choose how hard to push it, click
**Run**, and read the numbers. It's useful for:

- Comparing models or quantizations (e.g. Q4_K_M vs Q8_0).
- Seeing the speed difference between CPU and GPU.
- Checking how threads, batch size, or GPU offload affect throughput.

The tool does **not** include the llama.cpp program itself. It downloads the
correct official build for you, or you can point it at one you already have.

---

## 2. Installing and launching

There is nothing to install. The app is a single file.

1. Download `GGUFBenchmark-vX.Y.Z-win-x64.exe` from the
   [Releases page](https://github.com/giek2000/GGUF-Benchmark-Tool/releases/latest).
2. Double-click it.
3. Windows SmartScreen may say "Windows protected your PC" because the file is
   from an unknown publisher. Click **More info**, then **Run anyway**. (This is
   normal for small open-source apps that are not code-signed.)

To uninstall, just delete the `.exe`. To also remove saved settings and history,
delete the two `.json` files next to it (see [section 11](#11-where-your-files-are-saved)).

**System requirements:** Windows 10 or 11, 64-bit. A GPU is optional — the app
works on CPU alone.

---

## 3. The main window at a glance

```
┌──────────────────────────────────────────────────────────────┐
│ Engine bar:  [engine path]  [Browse] [backend ▼] [Download]   │  ← top
│                                                 [List devices] │
├──────────────────────────────────────────────────────────────┤
│  Tabs:  Benchmark | Results History | Model Info | System      │
│                                                                │
│  (contents of the selected tab)                                │
│                                                                │
├──────────────────────────────────────────────────────────────┤
│ App name + version │      status      │ 2026 Ricardo de Koning │  ← footer
└──────────────────────────────────────────────────────────────┘
```

- **Engine bar** (always visible): where the `llama-bench` program is, with
  buttons to Browse for it, pick a backend and Download it, or List devices.
- **Tabs**: the four main screens.
- **Footer**: the app name and version on the left, live status in the middle,
  and the author credit on the right.

---

## 4. Step-by-step: your first benchmark

1. **Get the engine.** In the engine bar, leave the backend on **CPU (x64)** if
   you're unsure, and click **Download**. Wait for "Engine ready" in the output.
2. **Load a model.** On the **Benchmark** tab click **Load GGUF...** and pick a
   `.gguf` file. (No model yet? See the FAQ for where to get one.)
3. **Pick a preset.** Choose **Quick (pp512/tg128)** from the Preset dropdown.
4. **Run.** Click **Run Benchmark**. You'll see live output, then a results
   table with tokens/second.
5. **Review.** Open the **Results History** tab to see the run saved for later
   comparison.

That's the whole loop. Everything else is refinement.

---

## 5. Getting the benchmark engine (CPU / GPU)

The app needs llama.cpp's `llama-bench.exe`. It looks for it automatically in a
saved location, next to the app, and on your system PATH. If it isn't found, use
the engine bar:

- **Browse...** — select a `llama-bench.exe` you already have.
- **backend dropdown + Download** — fetch the correct official build from GitHub.

### Which backend should I pick?

| Your hardware                         | Pick                               |
|---------------------------------------|------------------------------------|
| No GPU / not sure                     | **CPU (x64)**                      |
| NVIDIA GPU, recent driver             | **CUDA newest (NVIDIA, x64)**      |
| NVIDIA GPU, older driver              | **CUDA 12.x (older drivers, x64)** |
| AMD or Intel GPU (easiest)            | **Vulkan (AMD/Intel/NVIDIA, x64)** |
| Intel GPU via oneAPI                  | **SYCL (Intel, x64)**              |
| AMD GPU via ROCm                      | **ROCm (AMD, x64)**                |

Notes:
- For CUDA, the matching CUDA runtime is downloaded automatically.
- The download shows live progress. Files are saved in a `llama.cpp` folder next
  to the app and selected for you.
- When you download (or load) a GPU build, the app sets **GPU layers (-ngl)** to
  `99` automatically so the GPU is actually used. The engine bar also shows the
  detected build type, e.g. `[CUDA GPU build]` or `[CPU build]`.
- **List devices** runs `llama-bench --list-devices` to show which GPUs/CPUs the
  current engine can see — handy to confirm your GPU is detected.

---

## 6. The Benchmark tab

### Model

- **Load GGUF...** — choose the model file to test. You stay on the Benchmark
  tab; the model's details are read in the background and are available on the
  Model Info tab whenever you want them.
- **Inspect** — switch to the Model Info tab and read the model's details.

### Preset

A dropdown of ready-made settings. Picking one fills in the parameters below.

| Preset                            | What it measures                               |
|-----------------------------------|------------------------------------------------|
| Quick (pp512/tg128)               | A fast, standard read + write test.            |
| Standard (pp512/tg256, r5)        | Same, more repetitions for a steadier number.  |
| Prompt sweep (128,256,512,1024)   | Read speed at several prompt sizes.            |
| Generation only (tg64/128/256)    | Write speed at several lengths.                |
| Context stress (pp2048/tg512)     | A heavier, longer test.                        |

### Parameters

| Field                | Flag   | Meaning                                              |
|----------------------|--------|------------------------------------------------------|
| Prompt tokens        | `-p`   | How many input tokens to process. `512`, or a list like `128,512`. |
| Gen tokens           | `-n`   | How many tokens to generate.                         |
| Threads              | `-t`   | CPU threads. Blank = automatic.                      |
| GPU layers           | `-ngl` | Layers to offload to the GPU. `99` = use the GPU fully (default on GPU builds), `0` = CPU only. |
| Batch size           | `-b`   | Logical batch size. Blank = default.                 |
| Repetitions          | `-r`   | How many times to repeat each test (averaged).       |
| Output format        | `-o`   | How results are printed: md / csv / json / jsonl / sql. |
| Extra args           | (raw)  | Any additional `llama-bench` flags, typed directly.  |

All numeric fields are checked before the run — if something isn't a valid
number, you'll get a clear message instead of a confusing engine error.

### Buttons

- **Run Benchmark** — start the test.
- **Stop** — cancel a running test (stops the engine and its sub-processes).
- **Clear** — empty the output area.
- **Save Output...** — save the on-screen output to a file.

While a test runs, a progress bar animates and the status shows "Benchmarking...".

---

## 7. Reading the results

A typical results table looks like this:

```
| model                   | size      | params  | backend | ngl | test  |          t/s |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | CUDA    |   0 | pp512 | 87.60 ± 10.83 |
| qwen3 32B Q4_K - Medium | 18.40 GiB | 32.76 B | CUDA    |   0 | tg128 |  1.54 ±  0.03 |
```

- **test** — `pp512` is prompt processing of 512 tokens (reading speed);
  `tg128` is text generation of 128 tokens (writing speed).
- **t/s** — tokens per second. Higher is faster. The `±` value is the variation
  across repetitions; smaller means a more consistent result.
- **backend** / **ngl** — confirms what actually ran (here, CUDA with 0 layers
  offloaded, so it read on GPU but generated on CPU).

A run is considered successful when the engine exits cleanly. Successful runs are
automatically recorded to history. Normal startup lines such as
`ggml_cuda_init: found 1 CUDA devices` are **not** errors — they appear on every
healthy GPU run.

---

## 8. The Results History tab

Every successful run is saved here with a timestamp so you can compare over time.

Columns: **When, Model, ngl, Test, Params, tokens/s, ± stddev**.

Buttons:
- **Refresh** — reload the table.
- **Delete selected** — remove the highlighted rows (select multiple with Ctrl/Shift).
- **Clear all** — wipe the history (asks for confirmation).
- **Export CSV...** — save the whole history as a spreadsheet-friendly file.

History survives closing the app. It's stored next to the app and capped at the
most recent 500 rows.

---

## 9. The Model Info tab

Click **Inspect loaded GGUF** (or **Inspect** on the Benchmark tab) to read a
model's header without loading the whole thing. It shows:

- File path and size, GGUF version.
- Architecture (e.g. `llama`, `qwen3`) and name.
- Quantization (e.g. `Q4_K`) and the mix of tensor types.
- Context length, number of layers, attention heads, embedding size, and more.

This is a fast, read-only peek — it never modifies the file.

---

## 10. The System tab

Click **Refresh system info** to detect your hardware:

- Operating system and Python/runtime details.
- CPU model, physical and logical core counts.
- Total RAM.
- GPU(s), including NVIDIA details via `nvidia-smi` when available.

Use this to record what a benchmark was run on, or to confirm your GPU is seen.

---

## 11. Where your files are saved

All files live **next to the `.exe`**:

| File                             | What it is                                  |
|----------------------------------|---------------------------------------------|
| `gguf_benchmark_config.json`     | Your settings (engine path, last model).    |
| `gguf_benchmark_history.json`    | Your saved benchmark results.               |
| `llama.cpp\`                     | Downloaded engine build(s).                 |

These are written safely (an interrupted save can't corrupt them). Delete them to
reset the app to a clean state.

---

## 12. Troubleshooting

**"llama-bench not found."**
Use **Browse...** to select a `llama-bench.exe`, or pick a backend and click
**Download**.

**CUDA error: "the provided PTX was compiled with an unsupported toolchain."**
Your NVIDIA driver is older than the CUDA build. Fixes, easiest first:
1. In the engine bar pick **CUDA 12.x (older drivers, x64)** and **Download**.
2. Update your NVIDIA driver, then retry the newest CUDA build.
3. Switch to the **Vulkan** backend.
4. Use a **CPU** build with GPU layers (`-ngl`) set to `0`.
The app detects this error and shows these steps automatically.

**The GPU isn't being used during a benchmark.**
Check **GPU layers (-ngl)**. If it's `0`, the whole model runs on the CPU even
with a GPU engine loaded — set it to `99` to offload all layers to the GPU. The
results table confirms this: the `backend` column should read `CUDA` / `Vulkan`
(not `CPU`) and `ngl` should be non-zero. When you use a GPU engine with
`-ngl 0`, the app now warns you and offers to switch to `99` before running.

**Out of memory.**
Lower **GPU layers (-ngl)**, use a smaller or more-quantized model, or reduce
**Batch size (-b)** / **Prompt tokens (-p)**.

**The benchmark was flagged as failed but looked fine.**
Success is now judged by the engine's exit code, not by log text, so normal GPU
startup lines are no longer mistaken for errors. Make sure you're on the latest
version.

**The download won't find a build for my backend.**
Not every backend is published for every release. Try **CPU (x64)** (always
available) or **Vulkan**, or download manually from the
[llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and use
**Browse...**.

**A run is taking a long time.**
Large models and high repetition counts take time. You can press **Stop** at any
point.

---

## 13. Frequently asked questions

**Is it free?** Yes, open source under the MIT license.

**Where do I get GGUF models?** From [Hugging Face](https://huggingface.co/models?library=gguf)
— search for a model name plus "GGUF".

**Does it send my data anywhere?** No. It only contacts GitHub when you choose to
download an engine. Your models and results stay on your machine.

**Does it run on Mac or Linux?** The prebuilt `.exe` is Windows x64. On other
systems you can run the Python source (`python gguf_benchmark.py`) with a
`llama-bench` for your platform.

**Why download the engine separately?** The llama.cpp binaries are large and
specific to your GPU type. Fetching the exact build keeps the app small and
always current.

---

## 14. Glossary

- **GGUF** — the file format for a local llama.cpp model.
- **llama.cpp** — the open-source engine that runs the model.
- **llama-bench** — llama.cpp's benchmarking program, which this app drives.
- **Quantization** — compressing model weights (e.g. Q4_K_M) to trade a little
  quality for much smaller size and faster speed.
- **Token** — a word-piece. Models read and write in tokens.
- **pp (prompt processing)** — reading the input prompt.
- **tg (text generation)** — producing new tokens.
- **ngl (number of GPU layers)** — how much of the model runs on the GPU.
- **tokens/second (t/s)** — the headline speed measurement.

---

*GGUF Benchmark Tool — 2026 Created by Ricardo de Koning. Not affiliated with
the llama.cpp project.*
