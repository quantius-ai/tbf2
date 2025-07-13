# TBF2 (Token Block Format 2) Documentation

**Version:** 2.0  
**Authors:** Darin Tanner, Elijah Tribhuwan, Sharad Sreekanth  
**Copyright:** © 2025 Quantius LLC  
**License:** MIT  

## Overview

TBF2 is a binary file format specifically designed for efficient storage and retrieval of token sequences. It supports various bit-width configurations, compression algorithms, and provides both streaming and batch processing capabilities.

## Key Features

- **Multiple bit-widths:** 1-bit through 64-bit, plus variable-width "anybit" mode
- **Compression support:** Built-in support for zlib, bz2, lzma, plus extensible custom codecs
- **Signed/unsigned integers:** Configurable integer representation
- **NumPy acceleration:** Automatic acceleration when NumPy is available
- **Streaming I/O:** Memory-efficient processing of large files
- **Extensible compression:** Register custom compression algorithms at runtime

## Installation

```python
# No installation required - this is a single-file library
# Simply import the module in your Python project
```

## Quick Start

```python
from tbf2 import write_tbf2, read_tbf2

# Write token sequences to a file
data = [
    [1, 2, 3, 4],
    [10, 20, 30],
    [100, 200, 300, 400, 500]
]
write_tbf2("tokens.tbf2", data)

# Read back the data
loaded_data = read_tbf2("tokens.tbf2")
print(loaded_data)  # [[1, 2, 3, 4], [10, 20, 30], [100, 200, 300, 400, 500]]
```

## API Reference

### Core Functions

#### `write_tbf2(filename, data, *, token_mode_16bit=False, signed=False, compression="none", compression_level=None, use_numpy=None)`

One-shot helper function to write token sequences directly to a TBF2 file.

**Parameters:**
- `filename` (str | os.PathLike): Output file path
- `data` (Sequence[Sequence[int]]): Token sequences to write
- `token_mode_16bit` (bool, optional): Force 16-bit token mode. Default: False
- `signed` (bool, optional): Enable signed integer support. Default: False
- `compression` (str, optional): Compression algorithm ("none", "zlib", "bz2", "lzma"). Default: "none"
- `compression_level` (int, optional): Compression level (0-255). Default: None (use algorithm default)
- `use_numpy` (bool, optional): Enable NumPy acceleration. Default: None (auto-detect)

**Example:**
```python
# Basic usage
data = [[1, 2, 3], [4, 5, 6]]
write_tbf2("output.tbf2", data)

# With compression
write_tbf2("compressed.tbf2", data, compression="zlib", compression_level=9)

# With signed integers
signed_data = [[-1, 0, 1], [-10, 5, 15]]
write_tbf2("signed.tbf2", signed_data, signed=True)

# Force 16-bit mode
write_tbf2("16bit.tbf2", data, token_mode_16bit=True)
```

#### `read_tbf2(filename, *, use_numpy=None)`

Read a TBF2 file and return all token sequences.

**Parameters:**
- `filename` (str | os.PathLike): Input file path
- `use_numpy` (bool, optional): Enable NumPy acceleration. Default: None (auto-detect)

**Returns:**
- `List[List[int]]`: List of token sequences

**Example:**
```python
# Basic reading
data = read_tbf2("input.tbf2")
print(f"Loaded {len(data)} sequences")

# With NumPy disabled
data = read_tbf2("input.tbf2", use_numpy=False)
```

#### `get_tbf2_info(filename)`

Get metadata about a TBF2 file without loading the actual token data.

**Parameters:**
- `filename` (str | os.PathLike): Input file path

**Returns:**
- `dict`: File metadata including token_mode, compression, signed flag, etc.

**Example:**
```python
info = get_tbf2_info("example.tbf2")
print(f"Token mode: {info['token_mode']}")
print(f"Compression: {info['compression']}")
print(f"Signed: {info['signed']}")
print(f"Chunk count: {info['chunk_count']}")
print(f"Max tokens per chunk: {info['max_tokens_per_chunk']}")
```

#### `sha256_contents(filename)`

Calculate SHA-256 hash of file contents (excluding the 32-byte header).

**Parameters:**
- `filename` (str | os.PathLike): Input file path

**Returns:**
- `str`: Hexadecimal SHA-256 hash

**Example:**
```python
hash_value = sha256_contents("data.tbf2")
print(f"Content hash: {hash_value}")
```

#### `is_valid_payload(payload, *, signed=False)`

Validate that a payload has the correct structure for TBF2 format.

**Parameters:**
- `payload` (Any): Data to validate
- `signed` (bool, optional): Whether negative integers are allowed. Default: False

**Returns:**
- `bool`: True if payload is valid

**Example:**
```python
# Valid payloads
print(is_valid_payload([[1, 2, 3], [4, 5]]))  # True
print(is_valid_payload([1, 2, 3]))  # True (single sequence)
print(is_valid_payload([[-1, 0, 1]], signed=True))  # True

# Invalid payloads
print(is_valid_payload([]))  # False (empty)
print(is_valid_payload([[-1, 0, 1]]))  # False (negative without signed=True)
print(is_valid_payload([[1, 2], "invalid"]))  # False (mixed types)
```

### Writer Class

#### `TBF2Writer(file, *, token_mode="24bit", signed=False, compression="none", compression_level=None, buffer_size=4<<20, use_numpy=None)`

Streaming writer for TBF2 files.

**Parameters:**
- `file` (str | os.PathLike | io.BufferedWriter): Output file or file-like object
- `token_mode` (str, optional): Token bit-width mode. Default: "24bit"
- `signed` (bool, optional): Enable signed integers. Default: False
- `compression` (str, optional): Compression algorithm. Default: "none"
- `compression_level` (int, optional): Compression level (0-255). Default: None
- `buffer_size` (int, optional): I/O buffer size. Default: 4MB
- `use_numpy` (bool, optional): Enable NumPy acceleration. Default: None

**Token Modes:**
- `"1bit"`, `"2bit"`, `"4bit"`, `"8bit"`, `"12bit"`, `"16bit"`, `"24bit"`, `"32bit"`, `"48bit"`, `"64bit"`: Fixed bit-width modes
- `"anybit"`: Variable-width encoding (most space-efficient for sparse data)

**Example:**
```python
# Basic streaming write
with TBF2Writer("output.tbf2", token_mode="16bit") as writer:
    writer.write([1, 2, 3, 4])
    writer.write([10, 20, 30])
    writer.write([100, 200])

# With compression
with TBF2Writer("compressed.tbf2", compression="zlib", compression_level=6) as writer:
    for i in range(1000):
        writer.write([i, i+1, i+2])

# Anybit mode for variable-width tokens
with TBF2Writer("variable.tbf2", token_mode="anybit") as writer:
    writer.write([1, 100, 10000, 1000000])  # Efficiently packed
```

**Methods:**

##### `write(tokens)`
Write a sequence of tokens to the file.

**Parameters:**
- `tokens` (Sequence[int]): Token sequence to write

**Example:**
```python
writer = TBF2Writer("output.tbf2")
writer.write([1, 2, 3])
writer.write([4, 5, 6, 7])
writer.close()
```

##### `signed()`
Check if writer is in signed mode.

**Returns:**
- `bool`: True if signed mode is enabled

##### `min_size()` / `max_size()`
Get the minimum/maximum token values for the current mode.

**Returns:**
- `int`: Minimum or maximum allowed token value

**Example:**
```python
writer = TBF2Writer("test.tbf2", token_mode="8bit", signed=True)
print(f"Range: {writer.min_size()} to {writer.max_size()}")  # Range: -128 to 127
```

### Reader Class

#### `TBF2Reader(file, *, buffer_size=4<<20, use_numpy=None)`

Streaming reader for TBF2 files.

**Parameters:**
- `file` (str | os.PathLike | io.BufferedReader): Input file or file-like object
- `buffer_size` (int, optional): I/O buffer size. Default: 4MB
- `use_numpy` (bool, optional): Enable NumPy acceleration. Default: None

**Example:**
```python
# Basic streaming read
with TBF2Reader("input.tbf2") as reader:
    for chunk in reader:
        print(f"Chunk: {chunk}")

# Process large files efficiently
with TBF2Reader("large_file.tbf2") as reader:
    total_tokens = 0
    for chunk in reader:
        total_tokens += len(chunk)
        # Process chunk without loading entire file
    print(f"Total tokens: {total_tokens}")
```

**Methods:**

##### `signed()`
Check if file uses signed integers.

**Returns:**
- `bool`: True if signed mode

##### `min_size()` / `max_size()`
Get the minimum/maximum token values for the file's mode.

**Returns:**
- `int`: Minimum or maximum token value

##### `token_mode`
Property containing the token mode string.

**Example:**
```python
with TBF2Reader("input.tbf2") as reader:
    print(f"Token mode: {reader.token_mode}")
    print(f"Signed: {reader.signed()}")
    if reader.token_mode != "anybit":
        print(f"Range: {reader.min_size()} to {reader.max_size()}")
```

### Compression Management

#### `register_compression(name, compress_fn, decompress_fn, *, default_level=6, comp_id=None)`

Register a custom compression codec.

**Parameters:**
- `name` (str): Codec name
- `compress_fn` (Callable[[bytes, int], bytes]): Compression function
- `decompress_fn` (Callable[[bytes], bytes]): Decompression function
- `default_level` (int, optional): Default compression level. Default: 6
- `comp_id` (int, optional): Compression ID (4-255). Default: auto-assign

**Returns:**
- `int`: Assigned compression ID

**Example:**
```python
import gzip

# Register gzip compression
def gzip_compress(data, level):
    return gzip.compress(data, compresslevel=level)

def gzip_decompress(data):
    return gzip.decompress(data)

comp_id = register_compression("gzip", gzip_compress, gzip_decompress, default_level=6)
print(f"Registered gzip with ID: {comp_id}")

# Use the custom compression
write_tbf2("gzip_compressed.tbf2", [[1, 2, 3]], compression="gzip")
```

#### `get_compression(identifier)`

Get information about a registered compression codec.

**Parameters:**
- `identifier` (str | int): Codec name or ID

**Returns:**
- `_CompressionCodec`: Codec object

**Example:**
```python
codec = get_compression("gzip")
print(f"Default level: {codec.default_level}")
```

#### `remove_compression(identifier)`

Remove a previously registered custom compression codec.

**Parameters:**
- `identifier` (str | int): Codec name or ID to remove

**Note:** Built-in codecs (IDs 0-3) cannot be removed.

**Example:**
```python
remove_compression("gzip")  # Remove by name
# or
remove_compression(4)  # Remove by ID
```

## Advanced Usage

### Working with NumPy Arrays

```python
import numpy as np

# NumPy arrays are automatically supported
data = [
    np.array([1, 2, 3, 4], dtype=np.int32),
    np.array([10, 20, 30], dtype=np.int32)
]
write_tbf2("numpy_data.tbf2", data, use_numpy=True)

# Mixed NumPy and regular lists
mixed_data = [
    [1, 2, 3],  # Regular list
    np.array([4, 5, 6])  # NumPy array
]
write_tbf2("mixed_data.tbf2", mixed_data)
```

### Handling Large Files

```python
# Stream processing for memory efficiency
def process_large_file(input_file, output_file):
    with TBF2Reader(input_file) as reader:
        with TBF2Writer(output_file, compression="zlib") as writer:
            for chunk in reader:
                # Process chunk (e.g., filter, transform)
                processed = [token * 2 for token in chunk]
                writer.write(processed)

process_large_file("large_input.tbf2", "processed_output.tbf2")
```

### Custom Compression Example

```python
import pickle
import zstd  # hypothetical zstandard library

def zstd_compress(data, level):
    return zstd.compress(data, level=level)

def zstd_decompress(data):
    return zstd.decompress(data)

# Register custom codec
register_compression("zstd", zstd_compress, zstd_decompress, default_level=3)

# Use it
write_tbf2("zstd_compressed.tbf2", [[1, 2, 3]], compression="zstd")
data = read_tbf2("zstd_compressed.tbf2")
```

### Token Mode Selection Guide

```python
# Choose appropriate token mode based on your data:

# For small values (0-1): use "1bit"
binary_data = [[0, 1, 1, 0], [1, 0, 0, 1]]
write_tbf2("binary.tbf2", binary_data, token_mode="1bit")

# For small integers (0-255): use "8bit"
small_ints = [[10, 20, 30], [100, 150, 200]]
write_tbf2("small.tbf2", small_ints, token_mode="8bit")

# For typical token IDs (0-16,777,215): use "24bit" (default)
tokens = [[1000, 2000, 3000], [10000, 20000, 30000]]
write_tbf2("tokens.tbf2", tokens)  # Uses 24bit by default

# For very large or sparse values: use "anybit"
sparse_data = [[1, 1000000, 2], [50000, 3, 9000000]]
write_tbf2("sparse.tbf2", sparse_data, token_mode="anybit")
```

## Error Handling

```python
try:
    # File operations
    data = read_tbf2("nonexistent.tbf2")
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid file format: {e}")

try:
    # Token validation
    with TBF2Writer("test.tbf2", token_mode="8bit") as writer:
        writer.write([256])  # Too large for 8bit
except ValueError as e:
    print(f"Token out of range: {e}")

try:
    # Compression errors
    register_compression("existing", lambda x, y: x, lambda x: x)
except ValueError as e:
    print(f"Compression registration failed: {e}")
```

## Performance Tips

1. **Use appropriate token modes:** Choose the smallest bit-width that fits your data
2. **Enable NumPy acceleration:** Install NumPy for better performance with 16bit, 24bit, and 32bit modes
3. **Batch processing:** Write multiple tokens per `write()` call rather than one at a time
4. **Compression trade-offs:** Higher compression levels reduce file size but increase processing time
5. **Buffer sizes:** Adjust buffer sizes based on your I/O patterns

## File Format Specification

The TBF2 file format consists of:

1. **Header (32 bytes):**
   - Mode flag (1 byte)
   - Reserved fields (8 bytes) - contains compression ID, level, and flags
   - Chunk count (8 bytes)
   - Max tokens per chunk (4 bytes)
   - Padding (11 bytes)

2. **Chunks:** Each chunk contains:
   - Token count (4 bytes)
   - Compressed size (4 bytes, if compression enabled)
   - Token data (variable length)

## License

MIT License - see source code for full license text.
