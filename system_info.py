"""
System information helpers (standard library + optional Windows tooling).

Collects CPU, RAM and GPU details to display in the app. Uses `platform`,
`os`, and best-effort shell queries (wmic / PowerShell / nvidia-smi) on Windows.
All queries are defensive: if a tool is missing, that field is simply omitted.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _run(cmd: list[str], timeout: int = 8) -> str:
    try:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            creationflags=creationflags,
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def cpu_name() -> str:
    name = platform.processor() or ""
    if os.name == "nt":
        # processor() often returns a short id on Windows; try registry via reg.
        reg = _run([
            "reg", "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v", "ProcessorNameString",
        ])
        for line in reg.splitlines():
            if "ProcessorNameString" in line:
                parts = line.split("    ")
                name = parts[-1].strip() or name
    return name or platform.machine()


def cpu_cores() -> tuple[int, int]:
    """Returns (logical, physical-best-effort)."""
    logical = os.cpu_count() or 0
    physical = logical
    if os.name == "nt":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
        ])
        try:
            physical = int(out.strip()) or logical
        except ValueError:
            physical = logical
    return logical, physical


def total_ram_gib() -> float:
    if os.name == "nt":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        try:
            return int(out.strip()) / (1024 ** 3)
        except ValueError:
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, AttributeError, OSError):
        return 0.0


def gpus() -> list[str]:
    found: list[str] = []
    # NVIDIA via nvidia-smi
    if shutil.which("nvidia-smi"):
        out = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        for line in out.splitlines():
            line = line.strip()
            if line:
                found.append(f"NVIDIA: {line}")
    # Generic listing on Windows (covers AMD/Intel too)
    if os.name == "nt":
        out = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name",
        ])
        for line in out.splitlines():
            line = line.strip()
            if line and not any(line in f for f in found):
                found.append(line)
    return found


def collect() -> list[str]:
    logical, physical = cpu_cores()
    lines = [
        f"OS:            {platform.system()} {platform.release()} ({platform.version()})",
        f"Machine:       {platform.machine()}",
        f"Python:        {platform.python_version()}",
        f"CPU:           {cpu_name()}",
        f"CPU cores:     {physical} physical / {logical} logical",
        f"Total RAM:     {total_ram_gib():.1f} GiB",
    ]
    gpu_list = gpus()
    if gpu_list:
        lines.append("GPU(s):")
        lines.extend(f"   - {g}" for g in gpu_list)
    else:
        lines.append("GPU(s):        none detected")
    return lines


if __name__ == "__main__":
    print("\n".join(collect()))
