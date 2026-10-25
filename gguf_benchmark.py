"""
GGUF Model Benchmark Tool
=========================

A standalone GUI application to load a GGUF model file and benchmark it with
llama.cpp's `llama-bench`.

Features
--------
- Load any .gguf model and inspect its metadata (arch, quant, context, tensors).
- Configure benchmark parameters with quick-start presets.
- Run llama-bench with live streaming output and progress.
- GPU-backend aware engine download (CPU / CUDA / Vulkan) from official
  llama.cpp GitHub releases, plus manual browse.
- List compute devices the engine can see (--list-devices).
- Results history with persistence and a comparison table.
- System info panel (CPU / RAM / GPU detection).

The app is a thin wrapper around the official `llama-bench.exe`; it does not
bundle llama.cpp binaries (they are large and backend/hardware specific).

Author: Kiro
License: MIT
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import gguf_reader
import system_info


APP_NAME = "GGUF Benchmark Tool"
APP_VERSION = "2.3.0"
APP_AUTHOR = "Ricardo de Koning"
CONFIG_FILENAME = "gguf_benchmark_config.json"
HISTORY_FILENAME = "gguf_benchmark_history.json"
# The llama.cpp binary builds live under bXXXXX tags, NOT under /releases/latest
# (which points at a tooling release with no Windows binaries). So we list
# releases and pick the newest one that actually ships a Windows build.
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
LLAMA_RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases"

# Backend definitions. Each entry knows how to match its main llama build asset,
# and whether it additionally needs the CUDA runtime (cudart) zip to run.
# "match" is tried against asset names; the first asset matching any regex wins.
BACKENDS = {
    "CPU (x64)": {
        "match": [r"^llama-.*bin-win-cpu-x64\.zip$", r"^llama-.*bin-win-x64\.zip$"],
        "needs_cudart": False,
        "note": "Runs on any x64 CPU. No GPU required.",
    },
    "CUDA newest (NVIDIA, x64)": {
        # Newest CUDA toolkit variant (e.g. cuda-13.x). Needs a recent driver.
        # NOTE: must NOT match the separate 'cudart-...' runtime package, so we
        # require the leading 'llama-' prefix on the engine asset.
        "match": [r"^llama-.*bin-win-cuda-13.*-x64\.zip$",
                  r"^llama-.*bin-win-cuda-1[4-9].*-x64\.zip$",
                  r"^llama-.*bin-win-cuda.*-x64\.zip$"],
        "cudart_match": [r"^cudart-.*cuda-13.*-x64\.zip$", r"^cudart-.*cuda-1[4-9].*-x64\.zip$",
                         r"^cudart-.*cuda.*-x64\.zip$"],
        "needs_cudart": True,
        "note": "For NVIDIA GPUs with a RECENT driver. If you get an "
                "'unsupported toolchain' CUDA error, use 'CUDA 12.x' instead.",
    },
    "CUDA 12.x (older drivers, x64)": {
        # Older CUDA 12 toolkit - works with older NVIDIA drivers. This is the
        # fix for the 'provided PTX was compiled with an unsupported toolchain'
        # error that newer CUDA builds trigger on older drivers.
        "match": [r"^llama-.*bin-win-cuda-12.*-x64\.zip$"],
        "cudart_match": [r"^cudart-.*cuda-12.*-x64\.zip$"],
        "needs_cudart": True,
        "note": "For NVIDIA GPUs with OLDER drivers. Choose this if the newest "
                "CUDA build fails with an 'unsupported toolchain' error.",
    },
    "Vulkan (AMD/Intel/NVIDIA, x64)": {
        "match": [r"^llama-.*bin-win-vulkan-x64\.zip$"],
        "needs_cudart": False,
        "note": "Works on most modern GPUs (AMD, Intel, NVIDIA) via Vulkan drivers.",
    },
    "SYCL (Intel, x64)": {
        "match": [r"^llama-.*bin-win-sycl-x64\.zip$"],
        "needs_cudart": False,
        "note": "For Intel GPUs via oneAPI/SYCL.",
    },
    "ROCm (AMD, x64)": {
        "match": [r"^llama-.*bin-win-rocm.*-x64\.zip$"],
        "needs_cudart": False,
        "note": "For AMD GPUs via ROCm/HIP.",
    },
}

PRESETS = {
    "Quick (pp512/tg128)": {"p": "512", "n": "128", "r": "3"},
    "Standard (pp512/tg256, r5)": {"p": "512", "n": "256", "r": "5"},
    "Prompt sweep (128,256,512,1024)": {"p": "128,256,512,1024", "n": "0", "r": "3"},
    "Generation only (tg64/128/256)": {"p": "0", "n": "64,128,256", "r": "3"},
    "Context stress (pp2048/tg512)": {"p": "2048", "n": "512", "r": "3"},
}


# --------------------------------------------------------------------------- #
# Paths / config / history persistence
# --------------------------------------------------------------------------- #
def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(rel: str) -> Path:
    """Path to a bundled resource (works under PyInstaller onefile)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / rel
    return app_dir() / rel


def _json_load(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _json_save(path: Path, data) -> None:
    """Write JSON atomically: dump to a temp file then replace, so an
    interrupted write can never corrupt the existing file."""
    tmp = None
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        # Best-effort: clean up a stray temp file if the replace failed.
        try:
            if tmp is not None and tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def load_config() -> dict:
    return _json_load(app_dir() / CONFIG_FILENAME, {})


def save_config(cfg: dict) -> None:
    _json_save(app_dir() / CONFIG_FILENAME, cfg)


def load_history() -> list:
    data = _json_load(app_dir() / HISTORY_FILENAME, [])
    if not isinstance(data, list):
        return []
    # Keep only well-formed dict entries so a hand-edited or corrupt file
    # can't crash the history table.
    return [e for e in data if isinstance(e, dict)]


def save_history(history: list) -> None:
    _json_save(app_dir() / HISTORY_FILENAME, history)


def find_llama_bench(configured: str | None = None) -> str | None:
    exe_name = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    if configured and Path(configured).is_file():
        return str(Path(configured))
    base = app_dir()
    for c in (base / exe_name, base / "llama.cpp" / exe_name, base / "bin" / exe_name):
        if c.is_file():
            return str(c)
    # also search any extracted llama.cpp subfolder
    sub = base / "llama.cpp"
    if sub.is_dir():
        for p in sub.rglob(exe_name):
            return str(p)
    return shutil.which(exe_name)


# --------------------------------------------------------------------------- #
# Benchmark parameters & result parsing
# --------------------------------------------------------------------------- #
@dataclass
class BenchParams:
    model_path: str = ""
    n_prompt: str = "512"
    n_gen: str = "128"
    n_threads: str = ""
    n_gpu_layers: str = "0"
    batch_size: str = ""
    repetitions: str = "3"
    output_format: str = "md"
    extra_args: str = ""

    def validate(self) -> str | None:
        """Return an error message if any field is invalid, else None.
        Numeric fields accept comma-separated non-negative integers (llama-bench
        allows e.g. -p 128,512). Threads/ngl/batch accept a single integer."""
        def check_int_list(value: str, label: str, allow_empty: bool) -> str | None:
            value = value.strip()
            if not value:
                return None if allow_empty else f"{label} cannot be empty."
            for part in value.split(","):
                part = part.strip()
                if not part:
                    return f"{label} has an empty value in the list."
                if not part.isdigit():
                    return f"{label} must be whole number(s), got '{part}'."
            return None

        def check_int(value: str, label: str, allow_empty: bool, min_v: int = 0) -> str | None:
            value = value.strip()
            if not value:
                return None if allow_empty else f"{label} cannot be empty."
            if not (value.isdigit() and int(value) >= min_v):
                return f"{label} must be a whole number >= {min_v}, got '{value}'."
            return None

        checks = [
            check_int_list(self.n_prompt, "Prompt tokens (-p)", allow_empty=True),
            check_int_list(self.n_gen, "Gen tokens (-n)", allow_empty=True),
            check_int(self.n_threads, "Threads (-t)", allow_empty=True, min_v=1),
            check_int(self.n_gpu_layers, "GPU layers (-ngl)", allow_empty=True, min_v=0),
            check_int(self.batch_size, "Batch size (-b)", allow_empty=True, min_v=1),
            check_int(self.repetitions, "Repetitions (-r)", allow_empty=True, min_v=1),
        ]
        if not self.n_prompt.strip() and not self.n_gen.strip():
            return "Set at least one of Prompt tokens (-p) or Gen tokens (-n)."
        if self.output_format not in ("md", "csv", "json", "jsonl", "sql"):
            return f"Unknown output format: {self.output_format}"
        for err in checks:
            if err:
                return err
        # Extra args must parse cleanly.
        try:
            shlex.split(self.extra_args)
        except ValueError as exc:
            return f"Extra args could not be parsed: {exc}"
        return None

    def build_command(self, llama_bench: str, force_format: str | None = None) -> list[str]:
        cmd: list[str] = [llama_bench, "-m", self.model_path]

        def add(flag: str, value: str):
            value = value.strip()
            if value:
                cmd.extend([flag, value])

        add("-p", self.n_prompt)
        add("-n", self.n_gen)
        add("-t", self.n_threads)
        add("-ngl", self.n_gpu_layers)
        add("-b", self.batch_size)
        add("-r", self.repetitions)
        add("-o", force_format or self.output_format)
        extra = self.extra_args.strip()
        if extra:
            try:
                cmd.extend(shlex.split(extra))
            except ValueError:
                # Fall back to naive split if shlex can't parse (already warned
                # at validation time).
                cmd.extend(extra.split())
        return cmd


def parse_json_results(text: str) -> list[dict]:
    """Parse llama-bench JSON output into a list of result rows."""
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        # try to find the JSON array within noisy output
        m = re.search(r"\[\s*{.*}\s*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []


# --------------------------------------------------------------------------- #
# GUI Application
# --------------------------------------------------------------------------- #
class BenchmarkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("960x760")
        self.minsize(820, 660)

        self._apply_icon()

        self.cfg = load_config()
        self.history = load_history()
        self.llama_bench_path: str | None = find_llama_bench(self.cfg.get("llama_bench_path"))

        self._proc: subprocess.Popen | None = None
        self._out_queue: "queue.Queue" = queue.Queue()
        self._running = False
        self._last_returncode: int | None = None
        self._captured_text: str = ""

        self._build_ui()
        self._refresh_engine_label()
        self._refresh_history_table()
        self.after(100, self._drain_output_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_icon(self):
        try:
            ico = resource_path("icon.ico")
            if ico.is_file() and os.name == "nt":
                self.iconbitmap(str(ico))
        except Exception:
            pass

    # ---- UI ---------------------------------------------------------------#
    def _build_ui(self):
        # Engine bar (shared across tabs)
        engine_frame = ttk.LabelFrame(self, text="llama.cpp benchmark engine")
        engine_frame.pack(fill="x", padx=8, pady=(8, 4))
        self.engine_var = tk.StringVar()
        ttk.Label(engine_frame, textvariable=self.engine_var, foreground="#444").pack(
            side="left", fill="x", expand=True, padx=8, pady=6)
        ttk.Button(engine_frame, text="Browse...", command=self._browse_engine).pack(side="left", padx=2, pady=6)
        self.backend_var = tk.StringVar(value="CPU (x64)")
        ttk.Combobox(engine_frame, textvariable=self.backend_var, width=26, state="readonly",
                     values=list(BACKENDS.keys())).pack(side="left", padx=2, pady=6)
        ttk.Button(engine_frame, text="Download", command=self._download_engine).pack(side="left", padx=2, pady=6)
        ttk.Button(engine_frame, text="List devices", command=self._list_devices).pack(side="left", padx=2, pady=6)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=4)
        self._build_benchmark_tab()
        self._build_history_tab()
        self._build_modelinfo_tab()
        self._build_system_tab()

        # Status bar: app name + version (left), live status (center),
        # attribution (right).
        self.status_var = tk.StringVar(value="Ready")
        statusbar = ttk.Frame(self, relief="sunken")
        statusbar.pack(fill="x", side="bottom")
        ttk.Label(statusbar, text=f"{APP_NAME} v{APP_VERSION}", anchor="w").pack(
            side="left", padx=8, pady=2)
        ttk.Label(statusbar, text=f"\u00a9 2026 Created by {APP_AUTHOR}", anchor="e",
                  foreground="#888").pack(side="right", padx=8, pady=2)
        ttk.Label(statusbar, textvariable=self.status_var, anchor="center",
                  foreground="#666").pack(side="right", fill="x", expand=True, padx=8, pady=2)

    def _build_benchmark_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Benchmark")

        model_frame = ttk.LabelFrame(tab, text="GGUF model")
        model_frame.pack(fill="x", padx=8, pady=6)
        self.model_var = tk.StringVar(value=self.cfg.get("last_model", ""))
        ttk.Entry(model_frame, textvariable=self.model_var).pack(side="left", fill="x", expand=True, padx=8, pady=6)
        ttk.Button(model_frame, text="Load GGUF...", command=self._browse_model).pack(side="left", padx=4, pady=6)
        ttk.Button(model_frame, text="Inspect", command=self._inspect_current_model).pack(side="left", padx=4, pady=6)

        # Presets
        preset_frame = ttk.Frame(tab)
        preset_frame.pack(fill="x", padx=8, pady=2)
        ttk.Label(preset_frame, text="Preset:").pack(side="left")
        self.preset_var = tk.StringVar(value="Quick (pp512/tg128)")
        pc = ttk.Combobox(preset_frame, textvariable=self.preset_var, width=34, state="readonly",
                          values=list(PRESETS.keys()))
        pc.pack(side="left", padx=6)
        pc.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

        # Parameters
        param_frame = ttk.LabelFrame(tab, text="Benchmark parameters")
        param_frame.pack(fill="x", padx=8, pady=6)
        self.p_prompt = tk.StringVar(value="512")
        self.p_gen = tk.StringVar(value="128")
        self.p_threads = tk.StringVar(value="")
        self.p_ngl = tk.StringVar(value="0")
        self.p_batch = tk.StringVar(value="")
        self.p_reps = tk.StringVar(value="3")
        self.p_format = tk.StringVar(value="md")
        self.p_extra = tk.StringVar(value="")

        rows = [
            ("Prompt tokens (-p)", self.p_prompt, "e.g. 512 or 128,512"),
            ("Gen tokens (-n)", self.p_gen, "e.g. 128"),
            ("Threads (-t)", self.p_threads, "blank = auto"),
            ("GPU layers (-ngl)", self.p_ngl, "0 = CPU only, 99 = all"),
            ("Batch size (-b)", self.p_batch, "blank = default"),
            ("Repetitions (-r)", self.p_reps, "repeats per test"),
        ]
        for i, (label, var, hint) in enumerate(rows):
            r, c = divmod(i, 2)
            cell = ttk.Frame(param_frame)
            cell.grid(row=r, column=c, sticky="ew", padx=8, pady=4)
            param_frame.columnconfigure(c, weight=1)
            ttk.Label(cell, text=label, width=18).pack(side="left")
            ttk.Entry(cell, textvariable=var, width=14).pack(side="left")
            ttk.Label(cell, text=hint, foreground="#888").pack(side="left", padx=6)

        fmt_cell = ttk.Frame(param_frame)
        fmt_cell.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(fmt_cell, text="Output format (-o)", width=18).pack(side="left")
        ttk.Combobox(fmt_cell, textvariable=self.p_format, width=12, state="readonly",
                     values=["md", "csv", "json", "jsonl", "sql"]).pack(side="left")
        extra_cell = ttk.Frame(param_frame)
        extra_cell.grid(row=3, column=1, sticky="ew", padx=8, pady=4)
        ttk.Label(extra_cell, text="Extra args", width=18).pack(side="left")
        ttk.Entry(extra_cell, textvariable=self.p_extra).pack(side="left", fill="x", expand=True)

        action = ttk.Frame(tab)
        action.pack(fill="x", padx=8, pady=6)
        self.run_btn = ttk.Button(action, text="Run Benchmark", command=self._run_benchmark)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(action, text="Stop", command=self._stop_benchmark, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(action, text="Clear", command=self._clear_output).pack(side="left")
        ttk.Button(action, text="Save Output...", command=self._save_output).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(action, mode="indeterminate", length=160)
        self.progress.pack(side="right")

        out_frame = ttk.LabelFrame(tab, text="Output")
        out_frame.pack(fill="both", expand=True, padx=8, pady=6)
        self.output = tk.Text(out_frame, wrap="none", font=("Consolas", 10), background="#1e1e1e",
                              foreground="#e0e0e0", insertbackground="#e0e0e0")
        ys = ttk.Scrollbar(out_frame, orient="vertical", command=self.output.yview)
        xs = ttk.Scrollbar(out_frame, orient="horizontal", command=self.output.xview)
        self.output.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        out_frame.rowconfigure(0, weight=1)
        out_frame.columnconfigure(0, weight=1)

    def _build_history_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Results History")

        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Refresh", command=self._refresh_history_table).pack(side="left")
        ttk.Button(bar, text="Delete selected", command=self._delete_history_selected).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear all", command=self._clear_history).pack(side="left")
        ttk.Button(bar, text="Export CSV...", command=self._export_history_csv).pack(side="left", padx=6)

        cols = ("when", "model", "backend", "test", "params", "tps", "stddev")
        headings = {
            "when": "When", "model": "Model", "backend": "ngl", "test": "Test",
            "params": "Params", "tps": "tokens/s", "stddev": "± stddev",
        }
        widths = {"when": 135, "model": 200, "backend": 45, "test": 90,
                  "params": 120, "tps": 90, "stddev": 80}
        self.hist_tree = ttk.Treeview(tab, columns=cols, show="headings", selectmode="extended")
        for c in cols:
            self.hist_tree.heading(c, text=headings[c])
            self.hist_tree.column(c, width=widths[c], anchor="w")
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=ys.set)
        self.hist_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=6)
        ys.pack(side="left", fill="y", pady=6)

    def _build_modelinfo_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Model Info")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Inspect loaded GGUF", command=self._inspect_current_model).pack(side="left")
        self.modelinfo = tk.Text(tab, wrap="word", font=("Consolas", 10))
        self.modelinfo.pack(fill="both", expand=True, padx=8, pady=6)
        self.modelinfo.insert("end", "Load a GGUF model on the Benchmark tab, then click 'Inspect'.")

    def _build_system_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="System")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Refresh system info", command=self._refresh_system).pack(side="left")
        self.systext = tk.Text(tab, wrap="word", font=("Consolas", 10))
        self.systext.pack(fill="both", expand=True, padx=8, pady=6)
        self.systext.insert("end", "Click 'Refresh system info' to detect CPU / RAM / GPU.")

    # ---- Presets ----------------------------------------------------------#
    def _apply_preset(self):
        p = PRESETS.get(self.preset_var.get())
        if not p:
            return
        self.p_prompt.set(p.get("p", self.p_prompt.get()))
        self.p_gen.set(p.get("n", self.p_gen.get()))
        self.p_reps.set(p.get("r", self.p_reps.get()))

    # ---- Engine -----------------------------------------------------------#
    def _refresh_engine_label(self):
        if self.llama_bench_path:
            self.engine_var.set(f"Engine: {self.llama_bench_path}")
        else:
            self.engine_var.set("Engine: NOT FOUND - Browse or Download a build.")

    def _browse_engine(self):
        pattern = "llama-bench.exe" if os.name == "nt" else "llama-bench"
        path = filedialog.askopenfilename(
            title="Select llama-bench executable",
            filetypes=[("llama-bench", pattern), ("Executables", "*.exe"), ("All files", "*.*")])
        if path:
            self.llama_bench_path = path
            self.cfg["llama_bench_path"] = path
            save_config(self.cfg)
            self._refresh_engine_label()

    def _download_engine(self):
        if self._running:
            messagebox.showinfo(APP_NAME, "A task is already running.")
            return
        backend = self.backend_var.get()
        note = BACKENDS.get(backend, {}).get("note", "")
        msg = (f"Download the latest official llama.cpp '{backend}' build from GitHub?\n\n"
               f"{note}\n\n"
               "The files are saved next to this app and selected automatically.")
        if not messagebox.askyesno(APP_NAME, msg):
            return
        self.nb.select(0)
        self._clear_output()
        self._set_running(True, status=f"Downloading {backend} build...")
        self._start_worker(self._download_engine_worker, backend)

    def _fetch_json(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": APP_NAME,
                                                   "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _find_latest_windows_release(self, backend_match: list[str]):
        """Return (release_dict, tag) of the newest release that ships the wanted
        Windows asset. The llama.cpp binaries live under bXXXXX tags, so we scan
        the releases list rather than trusting /releases/latest."""
        releases = self._fetch_json(LLAMA_RELEASES_API)
        if not isinstance(releases, list):
            releases = [releases]
        for rel in releases:
            for asset in rel.get("assets", []):
                name = asset.get("name", "")
                if any(re.search(pat, name) for pat in backend_match):
                    return rel, rel.get("tag_name", "?")
        return None, None

    def _pick_asset(self, release: dict, patterns: list[str]):
        """Return (url, name, size) for the first asset matching patterns in order."""
        for pat in patterns:
            for asset in release.get("assets", []):
                name = asset.get("name", "")
                if re.search(pat, name):
                    return asset.get("browser_download_url"), name, asset.get("size", 0)
        return None, None, 0

    @staticmethod
    def _is_trusted_download_url(url: str) -> bool:
        """Only allow downloads over HTTPS from GitHub-owned hosts. The result
        is extracted and an .exe inside is later executed, so the source must
        be constrained even though the API itself is trusted."""
        try:
            parts = urllib.parse.urlparse(url)
        except Exception:
            return False
        if parts.scheme != "https":
            return False
        host = (parts.hostname or "").lower()
        allowed = ("github.com", "githubusercontent.com")
        return any(host == d or host.endswith("." + d) for d in allowed)

    def _download_to(self, url: str, name: str, dest_dir: Path):
        """Download a zip with progress reporting and extract into dest_dir."""
        if not self._is_trusted_download_url(url):
            raise ValueError(f"refusing download from untrusted URL: {url}")
        self._out_queue.put(f"Downloading {name} ...\n")
        req = urllib.request.Request(url, headers={"User-Agent": APP_NAME})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            chunks = []
            read = 0
            last_pct = -5
            while True:
                chunk = resp.read(262144)
                if not chunk:
                    break
                chunks.append(chunk)
                read += len(chunk)
                if total:
                    pct = int(read * 100 / total)
                    if pct >= last_pct + 5:
                        last_pct = pct
                        self._out_queue.put(
                            f"\r  {pct:3d}%  ({read//(1024*1024)}/{total//(1024*1024)} MB)")
            blob = b"".join(chunks)
        self._out_queue.put(f"\n  Extracting {name} ...\n")
        dest_dir = dest_dir.resolve()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            # Guard against path traversal ("zip slip"): refuse any entry that
            # would land outside dest_dir.
            for member in zf.namelist():
                target = (dest_dir / member).resolve()
                # Component-based containment check (a plain string prefix would
                # wrongly accept a sibling like 'llama.cpp-evil').
                if dest_dir != target and dest_dir not in target.parents:
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(dest_dir)

    def _download_engine_worker(self, backend: str):
        try:
            spec = BACKENDS.get(backend, {})
            match = spec.get("match", [])
            self._out_queue.put(f"Finding the newest llama.cpp release with a '{backend}' build...\n")

            release, tag = self._find_latest_windows_release(match)
            if not release:
                self._out_queue.put(
                    f"\nCouldn't find a '{backend}' Windows build in recent releases.\n"
                    "This backend may not be published for your platform right now.\n\n"
                    "What you can do:\n"
                    "  1. Try the CPU (x64) build, which is always available, or\n"
                    "  2. Pick a different GPU backend (Vulkan works on most GPUs), or\n"
                    f"  3. Download manually and use 'Browse...':\n     {LLAMA_RELEASES_PAGE}\n")
                return

            self._out_queue.put(f"Using release {tag}.\n")
            dest = app_dir() / "llama.cpp"
            dest.mkdir(exist_ok=True)

            # Main engine zip
            url, name, size = self._pick_asset(release, match)
            if not url:
                self._out_queue.put("Could not resolve the download URL. Try 'Browse...' instead.\n")
                return
            if size:
                self._out_queue.put(f"Engine package: {name} ({size//(1024*1024)} MB)\n")
            self._download_to(url, name, dest)

            # CUDA runtime, if this backend needs it
            if spec.get("needs_cudart"):
                c_url, c_name, c_size = self._pick_asset(release, spec.get("cudart_match", []))
                if c_url:
                    self._out_queue.put(
                        f"\nCUDA build needs the CUDA runtime. Fetching {c_name}"
                        f"{f' ({c_size//(1024*1024)} MB)' if c_size else ''}...\n")
                    self._download_to(c_url, c_name, dest)
                else:
                    self._out_queue.put(
                        "\nNote: couldn't find a matching CUDA runtime (cudart) package.\n"
                        "If benchmarking fails with a missing cudart/cublas DLL, install the\n"
                        "NVIDIA CUDA Toolkit, or download the cudart zip manually from:\n"
                        f"  {LLAMA_RELEASES_PAGE}\n")

            found = next((str(p) for p in dest.rglob("llama-bench.exe")), None)
            if not found:
                self._out_queue.put(
                    "\nDownloaded and extracted, but llama-bench.exe wasn't found inside.\n"
                    "The release layout may have changed. Use 'Browse...' to locate it manually.\n")
                return

            self.llama_bench_path = found
            self.cfg["llama_bench_path"] = found
            save_config(self.cfg)
            self._out_queue.put(f"\nDone. Engine ready:\n  {found}\n")
            if spec.get("needs_cudart"):
                self._out_queue.put("Tip: set 'GPU layers (-ngl)' to 99 to offload the whole model.\n")
            # _start_worker emits __DONE__
        except urllib.error.HTTPError as exc:
            self._out_queue.put(
                f"\nGitHub returned HTTP {exc.code} while downloading.\n"
                "This is often a temporary rate limit. Wait a minute and retry,\n"
                f"or download manually: {LLAMA_RELEASES_PAGE}\n")
        except urllib.error.URLError as exc:
            self._out_queue.put(
                f"\nNetwork error: {exc.reason}\n"
                "Check your internet connection or firewall, then retry.\n"
                f"You can also download manually: {LLAMA_RELEASES_PAGE}\n")
        except Exception as exc:  # noqa: BLE001
            self._out_queue.put(
                f"\nDownload failed: {exc}\n"
                f"You can download manually and use 'Browse...': {LLAMA_RELEASES_PAGE}\n")

    def _list_devices(self):
        if self._running:
            return
        if not self._engine_ok():
            return
        self.nb.select(0)
        self._clear_output()
        self._append_output("Listing compute devices...\n\n")
        self._set_running(True, status="Listing devices...")
        cmd = [self.llama_bench_path, "--list-devices"]
        self._start_worker(self._run_worker, cmd, False)

    # ---- Model ------------------------------------------------------------#
    def _browse_model(self):
        initial = self.cfg.get("last_model_dir", "")
        path = filedialog.askopenfilename(
            title="Select a GGUF model file",
            initialdir=initial if initial and Path(initial).is_dir() else None,
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.cfg["last_model"] = path
            self.cfg["last_model_dir"] = str(Path(path).parent)
            save_config(self.cfg)
            self._inspect_current_model()

    def _inspect_current_model(self):
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(APP_NAME, "Load a GGUF model first.")
            return
        info = gguf_reader.read_gguf(model)
        self.modelinfo.delete("1.0", "end")
        if info.error:
            self.modelinfo.insert("end", f"Could not read model:\n{info.error}")
        else:
            self.modelinfo.insert("end", "\n".join(info.summary_lines()))
            self.status_var.set(f"Inspected: {info.architecture} / {info.quantization}")
        self.nb.select(2)

    # ---- Run --------------------------------------------------------------#
    def _engine_ok(self) -> bool:
        if not self.llama_bench_path or not Path(self.llama_bench_path).is_file():
            messagebox.showwarning(APP_NAME, "llama-bench not found. Use Browse or Download.")
            return False
        return True

    def _collect_params(self) -> BenchParams | None:
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(APP_NAME, "Please load a GGUF model file first.")
            return None
        if not Path(model).is_file():
            messagebox.showwarning(APP_NAME, f"Model file not found:\n{model}")
            return None
        params = BenchParams(
            model_path=model, n_prompt=self.p_prompt.get(), n_gen=self.p_gen.get(),
            n_threads=self.p_threads.get(), n_gpu_layers=self.p_ngl.get(),
            batch_size=self.p_batch.get(), repetitions=self.p_reps.get(),
            output_format=self.p_format.get(), extra_args=self.p_extra.get())
        err = params.validate()
        if err:
            messagebox.showwarning(APP_NAME, err)
            return None
        return params

    def _run_benchmark(self):
        if self._running or not self._engine_ok():
            return
        params = self._collect_params()
        if params is None:
            return
        # Run with the user's chosen display format, but also capture JSON for history.
        display_cmd = params.build_command(self.llama_bench_path)
        self._clear_output()
        self._append_output("Command:\n  " + " ".join(_quote(a) for a in display_cmd) + "\n\n")
        self._set_running(True, status="Benchmarking...")
        self._start_worker(self._benchmark_sequence, params, display_cmd)

    def _benchmark_sequence(self, params: BenchParams, display_cmd: list[str]):
        # 1) Stream the human-readable run to the console.
        first_text = self._run_blocking(display_cmd, stream=True)
        rc = self._last_returncode

        # 2) Detect failure. llama-bench returns non-zero or crashes hard on
        #    backend/driver errors. Don't re-run a crashing command for history.
        failed = (rc is None) or (rc != 0) or _looks_like_failure(first_text)
        if failed:
            self._out_queue.put("\n" + _diagnose_failure(rc, first_text, params))
            return  # _start_worker emits __DONE__

        # 3) Capture structured results for history.
        if params.output_format not in ("json", "jsonl"):
            self._out_queue.put("\nCapturing structured results for history...\n")
            json_cmd = params.build_command(self.llama_bench_path, force_format="json")
            json_text = self._run_blocking(json_cmd, stream=False)
            if self._last_returncode not in (0, None) or _looks_like_failure(json_text):
                self._out_queue.put("(History capture run did not complete cleanly; skipping.)\n")
                json_text = ""
        else:
            json_text = first_text

        rows = parse_json_results(json_text)
        if rows:
            self._store_results(params, rows)
            self._out_queue.put(f"Recorded {len(rows)} result row(s) to history.\n")
        else:
            self._out_queue.put("No structured results were produced, so nothing was added to history.\n")
        # _start_worker emits __DONE__

    def _run_blocking(self, cmd: list[str], stream: bool) -> str:
        """Run a command to completion. If stream, push lines live; return full text.
        Also records self._last_returncode for the caller to inspect."""
        buf: list[str] = []
        self._last_returncode = None
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, creationflags=creationflags,
                cwd=str(Path(self.llama_bench_path).parent))
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                buf.append(line)
                if stream:
                    self._out_queue.put(line)
            self._proc.wait()
            self._last_returncode = self._proc.returncode
            if stream:
                self._out_queue.put(f"\n[Process exited with code {self._proc.returncode}]\n")
        except FileNotFoundError:
            self._out_queue.put("Error: could not launch llama-bench. Check the engine path.\n")
            self._last_returncode = -1
        except Exception as exc:  # noqa: BLE001
            self._out_queue.put(f"Error while running: {exc}\n")
            self._last_returncode = -1
        finally:
            self._proc = None
        self._captured_text = "".join(buf)
        return self._captured_text

    def _run_worker(self, cmd: list[str], record: bool):
        self._run_blocking(cmd, stream=True)
        # _start_worker emits __DONE__

    def _store_results(self, params: BenchParams, rows: list[dict]):
        """Build history entries (safe on the worker thread) then commit the
        mutation, save, and refresh on the UI thread to avoid a data race with
        the delete/export/refresh handlers."""
        when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_name = Path(params.model_path).name
        entries = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # llama-bench json keys vary slightly across versions; be defensive.
            test = row.get("test") or _derive_test_name(row)
            avg = row.get("avg_ts") or row.get("t/s") or row.get("tokens_per_second")
            std = row.get("stddev_ts") or row.get("stddev")
            entries.append({
                "when": when,
                "model": model_name,
                "model_path": params.model_path,
                "ngl": str(row.get("n_gpu_layers", params.n_gpu_layers)),
                "test": str(test),
                "params": f"p={params.n_prompt} n={params.n_gen} t={params.n_threads or 'auto'} b={params.batch_size or 'def'}",
                "tps": _fmt_num(avg),
                "stddev": _fmt_num(std),
                "raw": row,
            })
        if entries:
            self.after(0, lambda: self._commit_results(entries))

    def _commit_results(self, entries: list[dict]):
        """Runs on the UI thread: mutate history, persist, refresh table."""
        for entry in entries:
            self.history.insert(0, entry)
        self.history = self.history[:500]
        save_history(self.history)
        self._refresh_history_table()

    # ---- History ----------------------------------------------------------#
    def _refresh_history_table(self):
        if not hasattr(self, "hist_tree"):
            return
        self.hist_tree.delete(*self.hist_tree.get_children())
        for i, e in enumerate(self.history):
            self.hist_tree.insert("", "end", iid=str(i), values=(
                e.get("when", ""), e.get("model", ""), e.get("ngl", ""),
                e.get("test", ""), e.get("params", ""), e.get("tps", ""), e.get("stddev", "")))

    def _delete_history_selected(self):
        sel = self.hist_tree.selection()
        if not sel:
            return
        idxs = sorted((int(i) for i in sel), reverse=True)
        for i in idxs:
            if 0 <= i < len(self.history):
                del self.history[i]
        save_history(self.history)
        self._refresh_history_table()

    def _clear_history(self):
        if self.history and messagebox.askyesno(APP_NAME, "Clear all benchmark history?"):
            self.history = []
            save_history(self.history)
            self._refresh_history_table()

    def _export_history_csv(self):
        if not self.history:
            messagebox.showinfo(APP_NAME, "History is empty.")
            return
        path = filedialog.asksaveasfilename(
            title="Export history to CSV", defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["when", "model", "ngl", "test", "params", "tokens_per_second", "stddev"])
                for e in self.history:
                    w.writerow([e.get("when"), e.get("model"), e.get("ngl"), e.get("test"),
                                e.get("params"), e.get("tps"), e.get("stddev")])
            self.status_var.set(f"Exported history to {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_NAME, f"Export failed:\n{exc}")

    # ---- System -----------------------------------------------------------#
    def _refresh_system(self):
        self.systext.delete("1.0", "end")
        self.systext.insert("end", "Collecting system information...\n")
        threading.Thread(target=self._refresh_system_worker, daemon=True).start()

    def _refresh_system_worker(self):
        # Launched as a raw thread (not via _start_worker) because it never
        # toggles the running state; it only marshals its text back via after().
        try:
            lines = system_info.collect()
        except Exception as exc:  # noqa: BLE001
            lines = [f"Could not collect system info: {exc}"]
        text = "\n".join(lines)
        self.after(0, lambda: (self.systext.delete("1.0", "end"), self.systext.insert("end", text)))

    # ---- Output plumbing --------------------------------------------------#
    def _drain_output_queue(self):
        try:
            while True:
                item = self._out_queue.get_nowait()
                if item == "__DONE__":
                    self._set_running(False, status="Done")
                else:
                    self._append_output(item)
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    def _append_output(self, text: str):
        # Support carriage-return progress updates: a leading '\r' means
        # "rewrite the current (last) line" rather than append a new one.
        if text.startswith("\r"):
            # delete from the start of the last line to the end, then insert
            self.output.delete("end-1l linestart", "end")
            self.output.insert("end", "\n" + text[1:])
        else:
            self.output.insert("end", text)
        self.output.see("end")

    def _clear_output(self):
        self.output.delete("1.0", "end")

    def _save_output(self):
        content = self.output.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo(APP_NAME, "There is no output to save.")
            return
        ext = self.p_format.get()
        default_ext = f".{ext}" if ext in {"csv", "json", "jsonl", "md"} else ".txt"
        path = filedialog.asksaveasfilename(
            title="Save benchmark output", defaultextension=default_ext,
            filetypes=[("Markdown", "*.md"), ("CSV", "*.csv"), ("JSON", "*.json"),
                       ("Text", "*.txt"), ("All files", "*.*")])
        if path:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content + "\n")
                self.status_var.set(f"Saved to {path}")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(APP_NAME, f"Could not save file:\n{exc}")

    def _terminate_proc(self):
        """Terminate the running engine process and its children, if any."""
        proc = self._proc
        if not proc or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                # taskkill /T kills the whole process tree, which plain
                # terminate() may miss for engines that spawn children.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    timeout=10)
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            # Last resort
            try:
                proc.kill()
            except Exception:
                pass

    def _stop_benchmark(self):
        if self._proc and self._proc.poll() is None:
            self._terminate_proc()
            self._append_output("\n[Stopped by user]\n")

    # ---- Worker management ------------------------------------------------#
    def _start_worker(self, target, *args):
        """Launch a worker thread that is GUARANTEED to end the running state.
        If the worker raises, the error is reported and '__DONE__' is still
        emitted so the UI never gets stuck disabled."""
        def runner():
            try:
                target(*args)
            except Exception as exc:  # noqa: BLE001
                self._out_queue.put(f"\nUnexpected error: {exc}\n")
            finally:
                # Ensure the UI is released exactly once.
                self._out_queue.put("__DONE__")
        threading.Thread(target=runner, daemon=True).start()

    # ---- State ------------------------------------------------------------#
    def _set_running(self, running: bool, status: str = ""):
        self._running = running
        self.run_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self.status_var.set(status)
        self._refresh_engine_label()

    def _on_close(self):
        if self._running and self._proc and self._proc.poll() is None:
            if not messagebox.askyesno(APP_NAME, "A task is running. Quit anyway?"):
                return
            self._terminate_proc()
        self.destroy()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _quote(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def _fmt_num(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "" if v is None else str(v)


def _looks_like_failure(text: str) -> bool:
    """Heuristic detection of a failed/crashed llama-bench run from its output."""
    if not text:
        return False
    low = text.lower()
    markers = (
        "cuda error",
        "ggml_cuda",
        "unsupported toolchain",
        "out of memory",
        "failed to load model",
        "error loading model",
        "unknown argument",
        "invalid argument",
        "no such file",
        "assert",
        "terminate called",
        "vk::",            # vulkan errors
        "hip error",       # rocm errors
        "sycl",            # sycl errors often print SYCL exception
    )
    # "sycl" alone is too broad (appears in normal SYCL backend load lines),
    # so only count it when paired with an error word.
    hit = any(m in low for m in markers if m != "sycl")
    if not hit and "sycl" in low and ("error" in low or "exception" in low):
        hit = True
    return hit


def _diagnose_failure(rc, text: str, params: "BenchParams") -> str:
    """Produce a friendly, actionable explanation for a failed run."""
    low = (text or "").lower()
    lines = ["=" * 60, "Benchmark did NOT complete successfully.", ""]

    # Windows hard-crash exit code (0xC0000409 etc.)
    crashed_hard = isinstance(rc, int) and (rc < 0 or rc > (1 << 30))

    if "unsupported toolchain" in low or ("cuda error" in low and "ptx" in low):
        lines += [
            "Cause: the CUDA build is newer than your NVIDIA driver supports.",
            "The GPU was detected, but the kernels were compiled with a CUDA",
            "toolkit your installed driver can't run ('unsupported toolchain').",
            "",
            "How to fix (easiest first):",
            "  1. In the engine bar, pick 'CUDA 12.x (older drivers, x64)' and",
            "     click Download. This matches older NVIDIA drivers and usually",
            "     fixes this exact error. Then run again.",
            "  2. Or update your NVIDIA driver to the latest version and retry",
            "     with the newest CUDA build.",
            "  3. Or switch to the 'Vulkan' backend - it runs on this GPU without",
            "     the CUDA toolchain requirement. Select it and click Download.",
            "  4. Or run on CPU: pick a CPU build and set 'GPU layers (-ngl)' to 0.",
        ]
    elif "cuda error" in low or "ggml_cuda" in low:
        lines += [
            "Cause: a CUDA error occurred on the GPU backend.",
            "",
            "How to fix (pick one):",
            "  - Update your NVIDIA driver and retry.",
            "  - Try the Vulkan backend (more tolerant of driver versions).",
            "  - Reduce GPU layers (-ngl) or try a smaller / more-quantized model.",
            "  - Fall back to a CPU build (-ngl 0).",
        ]
    elif "out of memory" in low:
        lines += [
            "Cause: the GPU (or system) ran out of memory loading the model.",
            "",
            "How to fix (pick one):",
            "  - Lower 'GPU layers (-ngl)' so fewer layers are offloaded.",
            "  - Use a smaller or more aggressively quantized GGUF (e.g. Q4_K_M).",
            "  - Reduce batch size (-b) or prompt size (-p).",
        ]
    elif "failed to load model" in low or "error loading model" in low:
        lines += [
            "Cause: the model file could not be loaded.",
            "",
            "How to fix:",
            "  - Confirm the .gguf file isn't corrupt or still downloading.",
            "  - Check the Model Info tab to verify it's a valid GGUF.",
            f"  - Path tried: {params.model_path}",
        ]
    elif "unknown argument" in low or "invalid argument" in low:
        lines += [
            "Cause: llama-bench rejected an argument.",
            "",
            "How to fix:",
            "  - Check the 'Extra args' field for typos or unsupported flags.",
            "  - Different llama.cpp versions accept different options.",
        ]
    elif crashed_hard:
        lines += [
            f"The engine crashed (exit code {rc}).",
            "This usually means a GPU driver / backend mismatch.",
            "",
            "How to fix (pick one):",
            "  - Update your GPU driver and retry.",
            "  - Try the Vulkan backend, or a CPU build (-ngl 0).",
            "  - Download a build matching your driver's CUDA version.",
        ]
    else:
        lines += [
            f"The engine exited with code {rc}.",
            "Review the output above for the specific error.",
            "If it's GPU related, try the Vulkan backend or a CPU build (-ngl 0).",
        ]

    lines += ["", "Nothing was added to results history for this run.", "=" * 60, ""]
    return "\n".join(lines)


def _derive_test_name(row: dict) -> str:
    n_prompt = row.get("n_prompt")
    n_gen = row.get("n_gen")
    if n_prompt and not n_gen:
        return f"pp{n_prompt}"
    if n_gen and not n_prompt:
        return f"tg{n_gen}"
    if n_prompt and n_gen:
        return f"pp{n_prompt}+tg{n_gen}"
    return "test"


def main():
    app = BenchmarkApp()
    app.mainloop()


if __name__ == "__main__":
    main()
