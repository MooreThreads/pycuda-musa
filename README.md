# pycuda-musa

`pycuda-musa` is the Moore Threads MUSA maintenance line of
[PyCUDA](https://github.com/inducer/pycuda), providing Python access to GPU
memory, contexts, modules, arrays, and driver operations.

## Source and version

- Upstream project: PyCUDA
- Inherited version: `v2026.1`
- Upstream baseline: `c8448dca0d8853f2f84f0b48ff852c36b1aa63c5`
- Maintainer: Moore Threads

The MUSA line adds MUSA SDK build configuration and wrapper integration while
retaining the upstream Python API, CUDA-facing modules, examples, and tests.

## Requirements and build

Use a supported Python interpreter, a C++ compiler, Boost and NumPy as
required by the source, and the MUSA SDK. The MUSA setup entry point is
`setup_musa.py`; a typical editable installation is:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python setup_musa.py install
```

Review the setup options for the target SDK and compiler before installation.

## Layout and contribution

Python modules are under `pycuda/`, native wrappers under `src/`, and examples,
documentation, and tests retain the upstream organization. Keep MUSA and CUDA
driver paths distinct and preserve upstream attribution.

Moore Threads maintains this source line. See [LICENSE](LICENSE) for the
applicable declaration.
