# TBF2 (Token Block Format 2)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/quantius-ai/tbf2/releases)
[![Language](https://img.shields.io/badge/language-Python-blue.svg)](https://python.org)

> An efficient data format optimized for language model training

## Overview

**Token Block Format 2 (TBF2)** is a highly efficient binary data format specifically designed for language model training workflows. It provides optimized storage and fast access patterns for tokenized text data, making it ideal for large-scale machine learning applications.

## Features

- 🚀 **High Performance** - Optimized for fast sequential and random access
- 💾 **Space Efficient** - Compact binary format reduces storage requirements
- 🔧 **Easy Integration** - Simple API for common ML frameworks
- 📊 **Scalable** - Works for large sizes
- 🛡️ **Robust** - Built-in data integrity checks

## Performance

Over encoding 16-bit and 24-bit integers, TBF2 guarantees a 277% improvement over JSON in file size, and a 64% improvement in read speed.

## Quick Start

```python
# Example usage
import tbf2

# Write tokenized data
with tbf2.Writer('dataset.tbf2') as writer:
    writer.write_tokens([1, 2, 3, 4, 5])
    writer.write_tokens([6, 7, 8, 9, 10])

# Read tokenized data
with tbf2.Reader('dataset.tbf2') as reader:
    for token_block in reader:
        print(token_block)
```

## Installation

```bash
pip install tbf2
```

## Documentation

📚 [Full Documentation](https://github.com/yourusername/tbf2/wiki)

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for the ML community</sub>
</p>
