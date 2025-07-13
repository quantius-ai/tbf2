# ─────────────────────────────────────────────────────────────
#  Token Block Format 2
#  > Darin Tanner, Elijah Tribhuwan, Sharad Sreekanth
#  Copyright (c) 2025 Quantius AI LLC.
#  License: MIT
#
#  Permission is hereby granted, free of charge, to any person obtaining
#  a copy of this software and associated documentation files (the
#  “Software”), to deal in the Software without restriction, subject to
#  the MIT License.
#
#  SPDX-License-Identifier: MIT
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import bz2
import hashlib
import io
import lzma
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    "write_tbf2",
    "read_tbf2",
    "get_tbf2_info",
    "sha256_contents",
    "is_valid_payload",
    "register_compression",
    "get_compression",
    "remove_compression",
    "TBF2Reader",
    "TBF2Writer",
]

# ---------------------------------------------------------------------
# Format constants & helpers
# ---------------------------------------------------------------------

_MODE_TO_FLAG: dict[str, int] = {
    "1bit": 0x01,
    "2bit": 0x02,
    "4bit": 0x04,
    "8bit": 0x08,
    "12bit": 0x0C,
    "16bit": 0x10,
    "24bit": 0x18,
    "32bit": 0x20,
    "48bit": 0x30,
    "64bit": 0x40,
    "anybit": 0xFF,
}
_FLAG_TO_MODE = {v: k for k, v in _MODE_TO_FLAG.items()}

# Built-in compression IDs
_COMP_NAME_TO_ID: dict[str, int] = {
    "none": 0,
    "default": 1,  # alias for zlib (legacy compatibility)
    "zlib": 1,
    "bz2": 2,
    "lzma": 3,
}
_COMP_ID_TO_NAME: dict[int, str] = {
    v: k for k, v in _COMP_NAME_TO_ID.items() if k != "default"
}


@dataclass
class _CompressionCodec:
    compress: Callable[[bytes, int], bytes]
    decompress: Callable[[bytes], bytes]
    default_level: int = 6


# Runtime-registered codecs
_CUSTOM_CODECS: dict[int, _CompressionCodec] = {}
_CUSTOM_NAME_TO_ID: dict[str, int] = {}

# ------------------------------------------------------------------
# Public codec-management helpers
# ------------------------------------------------------------------


def _name_in_use(name: str) -> bool:
    """Return True if *name* already resolves to any codec."""
    if name in _CUSTOM_NAME_TO_ID:
        return True
    cid = _COMP_NAME_TO_ID.get(name)
    return cid in _CUSTOM_CODECS or (cid is not None and cid < 4)


def register_compression(
    name: str,
    compress_fn: Callable[[bytes, int], bytes],
    decompress_fn: Callable[[bytes], bytes],
    *,
    default_level: int = 6,
    comp_id: int | None = None,
) -> int:
    """Register a custom compression codec at runtime."""
    if _name_in_use(name):
        raise ValueError(f"Compression name {name!r} is already registered")
    if not (0 <= default_level <= 255):
        raise ValueError("default_level must be 0–255")

    if comp_id is None:
        taken = set(_COMP_ID_TO_NAME).union(_CUSTOM_CODECS)
        for cid in range(4, 256):
            if cid not in taken:
                comp_id = cid
                break
        else:
            raise RuntimeError("No available compression ID (0–255 exhausted)")
    else:
        if not (4 <= comp_id <= 255):
            raise ValueError("comp_id must be 4–255 (0–3 reserved)")
        if comp_id in _CUSTOM_CODECS or comp_id in _COMP_ID_TO_NAME:
            raise ValueError(f"Compression ID {comp_id} already in use")

    codec = _CompressionCodec(compress_fn, decompress_fn, default_level)
    _CUSTOM_CODECS[comp_id] = codec
    _CUSTOM_NAME_TO_ID[name] = comp_id
    _COMP_NAME_TO_ID[name] = comp_id
    _COMP_ID_TO_NAME[comp_id] = name
    return comp_id


def get_compression(identifier: Union[int, str]) -> _CompressionCodec:
    """Return a custom codec by *name* or *ID*."""
    if isinstance(identifier, str):
        cid = _CUSTOM_NAME_TO_ID.get(identifier)
        if cid is None:
            raise KeyError(f"No custom codec named {identifier!r}")
        return _CUSTOM_CODECS[cid]

    if isinstance(identifier, int):
        try:
            return _CUSTOM_CODECS[identifier]
        except KeyError as exc:
            raise KeyError(f"No custom codec with ID {identifier}") from exc

    raise TypeError("identifier must be str or int")


def remove_compression(identifier: Union[int, str]) -> None:
    """Unregister a previously-added custom codec.

    Built-in algorithms (IDs 0-3) cannot be removed.
    """
    # Resolve *identifier* → comp_id
    if isinstance(identifier, str):
        if identifier in _CUSTOM_NAME_TO_ID:
            cid = _CUSTOM_NAME_TO_ID[identifier]
        elif identifier in _COMP_NAME_TO_ID and _COMP_NAME_TO_ID[identifier] < 4:
            raise ValueError("Built-in codecs cannot be removed")
        else:
            raise KeyError(f"No custom codec named {identifier!r}")
    else:
        cid = identifier
        if cid in (0, 1, 2, 3):
            raise ValueError("Built-in codecs cannot be removed")

    # Remove codec
    try:
        codec_name = _COMP_ID_TO_NAME.pop(cid)
        _CUSTOM_CODECS.pop(cid)
    except KeyError as exc:
        raise KeyError(f"No custom codec with ID {cid}") from exc

    _COMP_NAME_TO_ID.pop(codec_name, None)
    _CUSTOM_NAME_TO_ID.pop(codec_name, None)

# ------------------------------------------------------------------
# Header constants
# ------------------------------------------------------------------

_HEADER_FMT = ">BQQL"  # mode(1) | reserved(8) | chunks(8) | max_tokens(4)
_HEADER_SIZE = 32
_HEADER_PACKER = struct.Struct(_HEADER_FMT)
_HEADER_RESERVED_TAIL = b"\x00" * (_HEADER_SIZE - _HEADER_PACKER.size)

# “reserved” layout (little-endian for convenience):
#   bits   0-7   → compression ID
#   bits   8-15  → compression level (0 → default)
#   bit    16    → signed integers flag
_SIGN_FLAG = 1 << 16

_UINT32 = struct.Struct(">I")
_USE_OS_PWRITE = hasattr(os, "pwrite")

# Default “good balance” levels for built-ins
_DEFAULT_LEVEL_FOR_ID = {0: 0, 1: 6, 2: 9, 3: 6}

# ------------------------------------------------------------------
# Optional NumPy acceleration
# ------------------------------------------------------------------

try:
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _HAVE_NUMPY = False

# ------------------------------------------------------------------
# Pure-Python bit-unpack helper
# ------------------------------------------------------------------


def _unpack_bits_py(data: bytes, n: int, bits: int) -> List[int]:
    out: List[int] = [0] * n
    bitpos = 0
    for i in range(n):
        v = 0
        for j in range(bits):
            byte_index = bitpos // 8
            bit_index = 7 - (bitpos % 8)
            if data[byte_index] >> bit_index & 1:
                v |= 1 << (bits - 1 - j)
            bitpos += 1
        out[i] = v
    return out


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------


def _signed_range(bits: int) -> Tuple[int, int]:
    """Return (min, max) for two’s-complement *bits*-wide signed integers."""
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _zigzag_encode(n: int) -> int:
    """ZigZag-encode *n* to an unsigned integer (arbitrary precision)."""
    return (n << 1) ^ (n >> (n.bit_length() or 1))


def _zigzag_decode(z: int) -> int:
    """Inverse of :func:`_zigzag_encode`."""
    return (z >> 1) ^ (-(z & 1))


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------


def write_tbf2(
    filename: str | os.PathLike[str],
    data: Sequence[Sequence[int]],
    *,
    token_mode_16bit: bool = False,
    signed: bool = False,
    compression: Literal["none", "default", "zlib", "bz2", "lzma"] | str = "none",
    compression_level: Optional[int] = None,
    use_numpy: bool | None = None,
) -> None:
    """One-shot helper: write *data* directly to a TBF2 file."""
    # Determine token mode
    if token_mode_16bit:
        chosen_mode: Literal["16bit", "24bit", "anybit"] = "16bit"
    else:
        max_mag = 0
        for blk in data:
            if not blk:
                continue
            m = max(abs(int(x)) for x in blk)
            max_mag = max(max_mag, m)
            if max_mag >= (1 << 23):
                break
        if max_mag >= (1 << (23 if signed else 24)):
            chosen_mode = "anybit"
        else:
            chosen_mode = "24bit"

        # Buffer an exhausted iterator
        if not isinstance(data, list):
            data = list(data)

    # Write file
    with TBF2Writer(
        filename,
        token_mode=chosen_mode,
        signed=signed,
        compression=compression,
        compression_level=compression_level,
        use_numpy=use_numpy,
    ) as w:
        for block in data:
            w.write(block)


def read_tbf2(
    filename: str | os.PathLike[str],
    *,
    use_numpy: bool | None = None,
) -> List[List[int]]:
    """Read *filename* and return all chunks."""
    with TBF2Reader(filename, use_numpy=use_numpy) as r:
        return list(r)


def get_tbf2_info(filename: str | os.PathLike[str]) -> dict:
    """Return header metadata without parsing payloads."""
    with open(filename, "rb") as f:
        raw = f.read(_HEADER_SIZE)
    if len(raw) != _HEADER_SIZE:
        raise ValueError("Invalid or truncated header.")
    flag, reserved, chunk_count, max_tokens = _HEADER_PACKER.unpack(
        raw[: _HEADER_PACKER.size]
    )
    comp_id = reserved & 0xFF
    comp_level = (reserved >> 8) & 0xFF
    signed = bool(reserved & _SIGN_FLAG)

    if flag not in _FLAG_TO_MODE:
        raise ValueError(f"Unknown mode flag 0x{flag:02x}")
    if comp_id not in _COMP_ID_TO_NAME and comp_id not in _CUSTOM_CODECS and comp_id != 0:
        raise ValueError(f"Unknown compression ID {comp_id}")

    compression = (
        "none" if comp_id == 0 else _COMP_ID_TO_NAME.get(comp_id, f"custom_{comp_id}")
    )

    return {
        "token_mode": _FLAG_TO_MODE[flag],
        "compression": compression,
        "compression_level": comp_level if comp_id else 0,
        "signed": signed,
        "chunk_count": chunk_count,
        "max_tokens_per_chunk": max_tokens,
    }


def sha256_contents(filename: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of everything after the 32-byte header."""
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        f.seek(_HEADER_SIZE)
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()

# ------------------------------------------------------------------
# Writer
# ------------------------------------------------------------------


class TBF2Writer:
    __slots__ = (
        "_f",
        "_own_file",
        "_flag",
        "_chunk_count",
        "_max_tokens",
        "_bits",
        "_signed",
        "_use_numpy",
        "_anybit",
        "_compress_id",
        "_compress_level",
        "_compress_fn",
    )

    # ------------------------------------------------ initialisation
    def __init__(
        self,
        file: str | os.PathLike[str] | io.BufferedWriter,
        *,
        token_mode: Literal[
            "1bit",
            "2bit",
            "4bit",
            "8bit",
            "12bit",
            "16bit",
            "24bit",
            "32bit",
            "48bit",
            "64bit",
            "anybit",
        ] = "24bit",
        signed: bool = False,
        compression: str = "none",
        compression_level: Optional[int] = None,
        buffer_size: int = 4 << 20,
        use_numpy: bool | None = None,
    ) -> None:
        # Output stream
        self._own_file = isinstance(file, (str, os.PathLike))
        self._f: io.BufferedWriter = (
            open(file, "wb", buffering=buffer_size)
            if self._own_file
            else file  # type: ignore[arg-type]
        )

        # Header fields
        self._flag = _MODE_TO_FLAG[token_mode]
        self._chunk_count = 0
        self._max_tokens = 0
        self._signed = bool(signed)

        # Compression / level
        if compression not in _COMP_NAME_TO_ID:
            raise ValueError(f"Unsupported compression: {compression!r}")
        self._compress_id = _COMP_NAME_TO_ID[compression]

        # Header level
        if compression_level is None:
            header_level = 0
        else:
            if not (0 <= compression_level <= 255):
                raise ValueError("compression_level must be 0–255")
            header_level = compression_level

        self._compress_level = header_level

        if self._compress_id == 0:
            runtime_level = 0
        elif self._compress_id in (1, 2, 3):
            runtime_level = (
                _DEFAULT_LEVEL_FOR_ID[self._compress_id]
                if header_level == 0
                else header_level
            )
        else:
            default_lv = _CUSTOM_CODECS[self._compress_id].default_level
            runtime_level = default_lv if header_level == 0 else header_level

        if self._compress_id == 0:
            self._compress_fn = lambda b: b
        elif self._compress_id == 1:
            self._compress_fn = lambda b, lv=runtime_level: zlib.compress(
                b, level=lv
            )
        elif self._compress_id == 2:
            self._compress_fn = lambda b, lv=runtime_level: bz2.compress(
                b, compresslevel=lv
            )
        elif self._compress_id == 3:
            self._compress_fn = lambda b, lv=runtime_level: lzma.compress(
                b, preset=lv
            )
        else:
            codec = _CUSTOM_CODECS[self._compress_id]
            self._compress_fn = lambda data, lv=runtime_level, fn=codec.compress: fn(
                data, lv
            )

        # Token width bookkeeping
        self._anybit = token_mode == "anybit"
        self._bits = 0 if self._anybit else int(token_mode[:-3])

        # NumPy acceleration?
        want_np = _HAVE_NUMPY if use_numpy is None else bool(use_numpy) and _HAVE_NUMPY
        self._use_numpy = (
            not self._anybit and want_np and self._bits in (16, 24, 32)
        )

        # Reserve placeholder header
        self._f.write(b"\x00" * _HEADER_SIZE)

    # ------------------------------------------------ public helpers
    def signed(self) -> bool:
        """Return True if writer is in signed mode."""
        return self._signed

    def min_size(self) -> int:
        if self._anybit:
            raise ValueError("min_size() undefined for anybit mode")
        return -(1 << (self._bits - 1)) if self._signed else 0

    def max_size(self) -> int:
        if self._anybit:
            raise ValueError("max_size() undefined for anybit mode")
        return (
            (1 << (self._bits - 1)) - 1
            if self._signed
            else ((1 << self._bits) - 1)
        )

    # ------------------------------------------------ encoders
    def _encode_block_anybit(self, tokens: Sequence[int]) -> bytes:
        out = bytearray()
        for t in tokens:
            if self._signed:
                val = _zigzag_encode(int(t))
            else:
                if t < 0:
                    raise ValueError("Negative token not allowed in unsigned mode")
                val = int(t)
            while True:  # unsigned LEB128
                byte = val & 0x7F
                val >>= 7
                out.append(byte | 0x80 if val else byte)
                if not val:
                    break
        return bytes(out)

    def _encode_block_numpy(self, tokens: Sequence[int]) -> memoryview:
        arr = _np.asarray(tokens, dtype=_np.int64).ravel()
        mask = (1 << self._bits) - 1

        if self._signed:
            lo, hi = _signed_range(self._bits)
            if arr.min(initial=0) < lo or arr.max(initial=0) > hi:
                raise ValueError(f"Token out of signed {self._bits}-bit range")
            arr_u = (arr & mask).astype("u8", copy=False)
        else:
            if arr.min(initial=0) < 0 or arr.max(initial=0) >= (mask + 1):
                raise ValueError(f"Token exceeds {self._bits}-bit limit")
            arr_u = arr.astype("u8", copy=False)

        if self._bits == 16:
            return memoryview(arr_u.astype(">u2", copy=False))
        if self._bits == 24:
            arr32 = arr_u.astype("u4", copy=False).astype(">u4", copy=False)
            view = arr32.view("u1").reshape(-1, 4)[:, 1:]
            return memoryview(_np.ascontiguousarray(view))
        return memoryview(arr_u.astype(">u4", copy=False))

    def _encode_block_py(self, tokens: Sequence[int]) -> bytes:
        bits = self._bits
        n = len(tokens)
        mask = (1 << bits) - 1
        lo, hi = _signed_range(bits) if self._signed else (0, mask)

        if bits < 8 or bits % 8:
            out = bytearray((n * bits + 7) // 8)
            bitpos = 0
            for t in tokens:
                if not (lo <= t <= hi):
                    raise ValueError(f"{t} out of range for {bits}-bit")
                val = t & mask
                for i in range(bits):
                    if (val >> (bits - 1 - i)) & 1:
                        out[bitpos // 8] |= 1 << (7 - (bitpos % 8))
                    bitpos += 1
            return bytes(out)

        byte_len = bits // 8
        out = bytearray(n * byte_len)
        for i, t in enumerate(tokens):
            if not (lo <= t <= hi):
                raise ValueError(f"{t} out of range for {bits}-bit")
            out[i * byte_len : (i + 1) * byte_len] = (t & mask).to_bytes(
                byte_len, "big"
            )
        return bytes(out)

    # ------------------------------------------------ public API
    def write(self, tokens: Sequence[int]) -> None:
        n = len(tokens)
        if n >= 1 << 32:
            raise ValueError("Chunk has too many tokens (> 4 294 967 295)")

        # Encode
        if self._anybit:
            payload = self._encode_block_anybit(tokens)
        elif self._use_numpy:
            payload = self._encode_block_numpy(tokens).tobytes()
        else:
            payload = self._encode_block_py(tokens)

        # Compress
        if self._compress_id:
            payload = self._compress_fn(payload)

        # Chunk header
        self._f.write(_UINT32.pack(n))
        if self._compress_id:
            self._f.write(_UINT32.pack(len(payload)))
        self._f.write(payload)

        # Stats
        self._chunk_count += 1
        self._max_tokens = max(self._max_tokens, n)

    # ------------------------------------------------ finalisation
    def close(self) -> None:
        if self._f.closed:
            return
        self._f.flush()
        reserved = (
            (self._compress_level << 8)
            | self._compress_id
            | (_SIGN_FLAG if self._signed else 0)
        )
        header = _HEADER_PACKER.pack(
            self._flag, reserved, self._chunk_count, self._max_tokens
        )
        if _USE_OS_PWRITE:
            os.pwrite(self._f.fileno(), header + _HEADER_RESERVED_TAIL, 0)
        else:
            self._f.seek(0)
            self._f.write(header + _HEADER_RESERVED_TAIL)
        if self._own_file:
            self._f.close()

    # ------------------------------------------------ context manager
    def __enter__(self) -> "TBF2Writer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        if exc_type is None:
            self.close()
        elif self._own_file and not self._f.closed:
            self._f.close()
        return None

# ------------------------------------------------------------------
# Reader
# ------------------------------------------------------------------


class TBF2Reader(Iterator[List[int]]):
    __slots__ = (
        "_f",
        "_own_file",
        "_bits",
        "_anybit",
        "_signed",
        "_use_numpy",
        "_decoder",
        "_compress_id",
        "_decompress_fn",
        "token_mode",
        "_skip_level",
    )

    def __init__(
        self,
        file: str | os.PathLike[str] | io.BufferedReader,
        *,
        buffer_size: int = 4 << 20,
        use_numpy: bool | None = None,
    ) -> None:
        self._own_file = isinstance(file, (str, os.PathLike))
        self._f: io.BufferedReader = (
            open(file, "rb", buffering=buffer_size)
            if self._own_file
            else file  # type: ignore[arg-type]
        )

        try:
            header = self._f.read(_HEADER_SIZE)
            if len(header) != _HEADER_SIZE:
                raise ValueError("Invalid or truncated header.")
            flag, reserved, _, _ = _HEADER_PACKER.unpack(header[: _HEADER_PACKER.size])

            # Mode & compression
            mode = _FLAG_TO_MODE.get(flag)
            if mode is None:
                raise ValueError(f"Unknown mode flag 0x{flag:02x}")

            self.token_mode = mode
            self._compress_id = reserved & 0xFF
            self._skip_level = (reserved >> 8) & 0xFF
            self._signed = bool(reserved & _SIGN_FLAG)
            self._anybit = mode == "anybit"

            if self._compress_id not in (0, 1, 2, 3) and self._compress_id not in _CUSTOM_CODECS:
                raise ValueError(f"Unknown compression ID: {self._compress_id}")

            # Decompressor
            if self._compress_id == 0:
                self._decompress_fn = lambda b: b
            elif self._compress_id == 1:
                self._decompress_fn = zlib.decompress
            elif self._compress_id == 2:
                self._decompress_fn = bz2.decompress
            elif self._compress_id == 3:
                self._decompress_fn = lzma.decompress
            else:
                self._decompress_fn = _CUSTOM_CODECS[self._compress_id].decompress

            # Token width
            self._bits = 0 if self._anybit else int(mode[:-3])

            # NumPy?
            want_np = _HAVE_NUMPY if use_numpy is None else bool(use_numpy) and _HAVE_NUMPY
            self._use_numpy = not self._anybit and want_np and self._bits in (16, 24, 32)

            # Decoder
            if self._anybit:
                self._decoder = None
            elif self._use_numpy:
                if self._bits == 16:
                    self._decoder = lambda d, n: _np.frombuffer(d, dtype=">u2").tolist()
                elif self._bits == 24:
                    self._decoder = self._decode_24_numpy
                else:
                    self._decoder = lambda d, n: _np.frombuffer(d, dtype=">u4").tolist()
            else:
                self._decoder = self._make_py_decoder()

        except Exception:
            if self._own_file and not self._f.closed:
                self._f.close()
            raise

    # ------------------------------------------------ API helpers
    def signed(self) -> bool:
        return self._signed

    # ------------------------------------------------ decoder helpers
    def _make_py_decoder(self):
        bits = self._bits
        if bits in (16, 32):

            def fn(d: bytes, n: int) -> List[int]:
                if bits == 16:
                    return [x >> 8 | ((x & 0xFF) << 8) for x in memoryview(d).cast("H")]
                return [
                    ((x & 0xFF) << 24)
                    | ((x & 0xFF00) << 8)
                    | ((x >> 8) & 0xFF00)
                    | (x >> 24)
                    for x in memoryview(d).cast("I")
                ]

            return fn

        if bits == 24:

            def fn(d: bytes, n: int) -> List[int]:
                return [
                    (d[i] << 16) | (d[i + 1] << 8) | d[i + 2]
                    for i in range(0, len(d), 3)
                ]

            return fn

        if bits % 8 == 0:
            byte_len = bits // 8

            def fn(d: bytes, n: int) -> List[int]:
                return [
                    int.from_bytes(d[i * byte_len : (i + 1) * byte_len], "big")
                    for i in range(n)
                ]

            return fn

        return lambda d, n, b=bits: _unpack_bits_py(d, n, b)

    @staticmethod
    def _decode_24_numpy(buf: bytes, _: int) -> list[int]:
        arr = _np.frombuffer(buf, dtype="u1").reshape(-1, 3)
        return (
            (arr[:, 0].astype("u4") << 16)
            | (arr[:, 1].astype("u4") << 8)
            | arr[:, 2]
        ).tolist()

    @staticmethod
    def _decode_anybit_from_bytes(buf: bytes, n: int) -> List[int]:
        out: List[int] = []
        idx = 0
        for _ in range(n):
            val = 0
            shift = 0
            while True:
                if idx >= len(buf):
                    raise ValueError("Truncated anybit varint")
                byte = buf[idx]
                idx += 1
                val |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            out.append(val)
        return out

    # ------------------------------------------------ iterator
    def __iter__(self) -> "TBF2Reader":
        return self

    def __next__(self) -> List[int]:
        sz = self._f.read(4)
        if not sz:
            raise StopIteration
        if len(sz) != 4:
            raise ValueError("Corrupted file: incomplete chunk header")
        n = _UINT32.unpack(sz)[0]

        if self._compress_id:
            comp_size_raw = self._f.read(4)
            if len(comp_size_raw) != 4:
                raise ValueError("Corrupted file: missing compressed-size")
            comp_size = _UINT32.unpack(comp_size_raw)[0]
            comp_payload = self._f.read(comp_size)
            if len(comp_payload) != comp_size:
                raise ValueError("Corrupted file: truncated compressed payload")
            payload = self._decompress_fn(comp_payload)

            if self._anybit:
                vals_u = self._decode_anybit_from_bytes(payload, n)
                return [_zigzag_decode(v) for v in vals_u] if self._signed else vals_u
            vals = self._decoder(payload, n)  # type: ignore[arg-type]
            return self._apply_signed(vals)

        # Uncompressed path
        if self._anybit:
            vals_u = self._decode_anybit_stream(n)
            return [_zigzag_decode(v) for v in vals_u] if self._signed else vals_u

        need = (
            n * (self._bits // 8)
            if self._bits % 8 == 0
            else (n * self._bits + 7) // 8
        )
        payload = self._f.read(need)
        if len(payload) != need:
            raise ValueError("Corrupted file: truncated chunk")
        vals = self._decoder(payload, n)  # type: ignore[arg-type]
        return self._apply_signed(vals)

    # anybit (streaming)
    def _decode_anybit_stream(self, n: int) -> List[int]:
        out: List[int] = []
        for _ in range(n):
            val = 0
            shift = 0
            while True:
                b = self._f.read(1)
                if not b:
                    raise ValueError("Corrupted file: truncated anybit varint")
                byte = b[0]
                val |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            out.append(val)
        return out

    # sign helpers
    def _apply_signed(self, vals: List[int]) -> List[int]:
        if not self._signed:
            return vals
        mod = 1 << self._bits
        sign_bit = 1 << (self._bits - 1)
        return [(v - mod) if (v & sign_bit) else v for v in vals]

    # min/max helpers
    def min_size(self) -> int:
        if self._anybit:
            raise ValueError("min_size() undefined for anybit mode")
        return -(1 << (self._bits - 1)) if self._signed else 0

    def max_size(self) -> int:
        if self._anybit:
            raise ValueError("max_size() undefined for anybit mode")
        return (
            (1 << (self._bits - 1)) - 1
            if self._signed
            else ((1 << self._bits) - 1)
        )

    # housekeeping
    def close(self) -> None:
        if self._own_file and not self._f.closed:
            self._f.close()

    def __enter__(self) -> "TBF2Reader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

# ------------------------------------------------------------------
# Payload validator
# ------------------------------------------------------------------


def is_valid_payload(payload: Any, *, signed: bool = False) -> bool:
    """Lightweight structural validation of token payloads."""
    try:
        import numpy as np

        HAVE_NUMPY = True
    except ImportError:  # pragma: no cover
        HAVE_NUMPY = False
        np = None  # type: ignore

    def _is_int(x):
        return isinstance(x, int) and (signed or x >= 0)

    def _check_ints(seq) -> bool:
        if HAVE_NUMPY and isinstance(seq, np.ndarray):
            if not np.issubdtype(seq.dtype, np.integer) or seq.size == 0:
                return False
            return bool(signed or np.all(seq >= 0))
        if not seq:
            return False
        return all(_is_int(el) for el in seq)

    if HAVE_NUMPY and isinstance(payload, np.ndarray):
        if payload.ndim == 1:
            return _check_ints(payload)
        if payload.ndim == 2:
            return all(_check_ints(row) for row in payload)
        return False

    if isinstance(payload, list):
        if not payload:
            return False
        has_listlike = any(
            isinstance(el, (list, np.ndarray) if HAVE_NUMPY else list) for el in payload
        )
        has_nonlistlike = any(
            not isinstance(el, (list, np.ndarray) if HAVE_NUMPY else list)
            for el in payload
        )
        if has_listlike and has_nonlistlike:
            return False
        if has_listlike:
            return all(_check_ints(sub) for sub in payload)
        return _check_ints(payload)

    return False
