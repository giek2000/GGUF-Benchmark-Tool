"""
Minimal GGUF header reader (pure standard library).

Parses the GGUF file header and key/value metadata so the app can show model
details (architecture, context length, quantization mix, tensor count, etc.)
without loading the whole model or depending on llama.cpp.

GGUF format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types
(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32,
    T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

# ggml tensor type id -> human name (subset covering common quants)
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
    23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16",
}


@dataclass
class GGUFInfo:
    path: str = ""
    version: int = 0
    tensor_count: int = 0
    kv_count: int = 0
    metadata: dict = field(default_factory=dict)
    tensor_types: dict = field(default_factory=dict)  # type name -> count
    file_size: int = 0
    error: str = ""

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", Path(self.path).stem if self.path else ""))

    @property
    def quantization(self) -> str:
        # file_type is a general enum; the tensor_types mix is more informative
        if self.tensor_types:
            # dominant quant among non-F32/F16 types, else most common
            ranked = sorted(self.tensor_types.items(), key=lambda kv: kv[1], reverse=True)
            for name, _ in ranked:
                if name not in ("F32", "F16", "BF16"):
                    return name
            return ranked[0][0]
        ft = self.metadata.get("general.file_type")
        return f"file_type={ft}" if ft is not None else "unknown"

    def context_length(self) -> int | None:
        arch = self.architecture
        for key in (f"{arch}.context_length", "llama.context_length"):
            if key in self.metadata:
                try:
                    return int(self.metadata[key])
                except (TypeError, ValueError):
                    return None
        return None

    def summary_lines(self) -> list[str]:
        lines = [
            f"File:            {self.path}",
            f"File size:       {self.file_size / (1024**3):.2f} GiB",
            f"GGUF version:    {self.version}",
            f"Architecture:    {self.architecture}",
            f"Name:            {self.name}",
            f"Quantization:    {self.quantization}",
            f"Tensors:         {self.tensor_count}",
            f"Metadata keys:   {self.kv_count}",
        ]
        ctx = self.context_length()
        if ctx:
            lines.append(f"Context length:  {ctx}")

        arch = self.architecture
        interesting = [
            (f"{arch}.embedding_length", "Embedding dim"),
            (f"{arch}.block_count", "Layers (blocks)"),
            (f"{arch}.attention.head_count", "Attention heads"),
            (f"{arch}.attention.head_count_kv", "KV heads"),
            (f"{arch}.feed_forward_length", "Feed-forward dim"),
            (f"{arch}.rope.freq_base", "RoPE freq base"),
            ("general.quantization_version", "Quant version"),
        ]
        for key, label in interesting:
            if key in self.metadata:
                lines.append(f"{label + ':':16} {self.metadata[key]}")

        if self.tensor_types:
            mix = ", ".join(f"{k}:{v}" for k, v in sorted(self.tensor_types.items()))
            lines.append(f"Tensor type mix: {mix}")
        return lines


class GGUFParseError(Exception):
    """Raised when a GGUF file's self-described sizes are implausible."""


def _remaining(f, file_size: int) -> int:
    return file_size - f.tell()


def _read_exact(f, n: int):
    data = f.read(n)
    if len(data) != n:
        raise GGUFParseError("unexpected end of file")
    return data


def _read_str(f, file_size: int) -> str:
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    # A string can't be longer than the bytes left in the file. This rejects a
    # hostile/corrupt length (e.g. 2**64-1) before allocating.
    if length > _remaining(f, file_size):
        raise GGUFParseError("string length exceeds file size")
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_scalar(f, vtype: int):
    try:
        fmt = {
            T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
            T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
            T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
        }[vtype]
    except KeyError:
        raise GGUFParseError(f"unknown value type {vtype}")
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(f, size))[0]


def _read_value(f, vtype: int, file_size: int):
    if vtype == T_STRING:
        return _read_str(f, file_size)
    if vtype == T_ARRAY:
        (elem_type,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        # The smallest element is 1 byte, so count can't exceed the bytes left.
        if count > _remaining(f, file_size):
            raise GGUFParseError("array count exceeds file size")
        # Avoid loading enormous arrays (e.g. tokenizer vocab) fully into memory.
        max_items = 64
        values = []
        for i in range(count):
            v = _read_value(f, elem_type, file_size)
            if i < max_items:
                values.append(v)
        if count > max_items:
            return f"[array len={count}, first {max_items} shown] " + repr(values)
        return values
    return _read_scalar(f, vtype)


def read_gguf(path: str, read_tensor_types: bool = True) -> GGUFInfo:
    """Read GGUF header + metadata. Returns GGUFInfo (with .error set on failure)."""
    info = GGUFInfo(path=path)
    try:
        info.file_size = Path(path).stat().st_size
        fsize = info.file_size
        with open(path, "rb") as f:
            (magic,) = struct.unpack("<I", _read_exact(f, 4))
            if magic != GGUF_MAGIC:
                info.error = "Not a GGUF file (bad magic number)."
                return info
            (info.version,) = struct.unpack("<I", _read_exact(f, 4))
            (info.tensor_count,) = struct.unpack("<Q", _read_exact(f, 8))
            (info.kv_count,) = struct.unpack("<Q", _read_exact(f, 8))

            # Each entry occupies several bytes, so neither count can exceed the
            # file size. This rejects corrupt/hostile counts before looping.
            if info.kv_count > fsize or info.tensor_count > fsize:
                info.error = "GGUF header counts exceed file size (corrupt file)."
                return info

            for _ in range(info.kv_count):
                key = _read_str(f, fsize)
                (vtype,) = struct.unpack("<I", _read_exact(f, 4))
                value = _read_value(f, vtype, fsize)
                info.metadata[key] = value

            if read_tensor_types:
                try:
                    for _ in range(info.tensor_count):
                        _name = _read_str(f, fsize)
                        (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
                        if n_dims > 8:  # GGML_MAX_DIMS is 4; 8 is a generous cap
                            raise GGUFParseError("implausible tensor dimension count")
                        _read_exact(f, 8 * n_dims)  # dimensions (uint64 each)
                        (ttype,) = struct.unpack("<I", _read_exact(f, 4))
                        _read_exact(f, 8)  # offset (uint64)
                        tname = GGML_TYPE_NAMES.get(ttype, f"type_{ttype}")
                        info.tensor_types[tname] = info.tensor_types.get(tname, 0) + 1
                except Exception:
                    # Tensor info is best-effort; metadata is the important part.
                    pass
    except FileNotFoundError:
        info.error = "File not found."
    except Exception as exc:  # noqa: BLE001
        info.error = f"Failed to parse GGUF: {exc}"
    return info


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python gguf_reader.py <model.gguf>")
        raise SystemExit(1)
    nfo = read_gguf(sys.argv[1])
    if nfo.error:
        print("ERROR:", nfo.error)
    else:
        print("\n".join(nfo.summary_lines()))
