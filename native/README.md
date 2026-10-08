# Optional native feature encoder

The neural network remains PyTorch. This optional CPython C extension accelerates
character feature extraction, preserving Unicode n-gram boundaries, digest size,
hash order and bucket indices. It uses no prediction cache and does not change
weights or calibration. Whole-word features and cleanup remain in Python.

A bounded direct-mapped cache stores exact digests for up to 32,768 fragments
(maximum 20 UTF-8 bytes each; approximately 1 MiB total). Cache hits compare
length and bytes; collisions only evict. This caches short text fragments in
process memory, never whole-text predictions. `import _windup_native;
_windup_native.cache_clear()` clears it. Repeated-input latency benefits from
warm fragments; varied and initially empty-cache results are reported separately.
See [measurements and protocol](../docs/native-performance.md).

Build from the repository root with a C compiler and Python development headers:

```bash
python -m pip install setuptools
python setup_native.py build_ext --inplace
python -m unittest discover -s tests
```

The module is used automatically when built for the current Python interpreter.
Without it, inference uses the original Python encoder. A present but broken
extension raises its import error instead of silently hiding installation faults.
Unsupported/custom encoding configurations use Python. Rebuild after changing
Python or platform; compiled modules are local artifacts excluded from Git.

## Upstream source

The unmodified files in `blake2/` come from the official BLAKE2 reference tree at
commit `ed1974ea83433eba7b2d95c5dcd9ac33cb847913`:
https://github.com/BLAKE2/BLAKE2/tree/ed1974ea83433eba7b2d95c5dcd9ac33cb847913/ref

Copyright 2012 Samuel Neves. This project selects the offered CC0 1.0 option;
see `blake2/COPYING`. The Python/C adapter is project-authored MIT software.
