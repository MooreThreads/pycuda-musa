# MUSIFY — pycuda

| Field | Value |
|-------|-------|
| Project | pycuda |
| Source repo | ssh://git@sh-gitlab.mthreads.com:2222/auto-musify-blocked/pycuda |
| Migration branch | musa |
| MUSA SDK version | 4.3.5 |
| torch_musa version | N/A (pycuda does not depend on torch) |
| Docker image | sh-harbor.mthreads.com/mcctest/musa-train:4.3.5_kuae2.1_20260119_ubuntu |
| MT GPU type | S5000 |
| GPU | 7 |
| Pipeline start | 2026-04-03 03:05:50 UTC |
| Pipeline end | 2026-04-03 16:05 UTC |
| Elapsed | ~13h |
| Last updated | 2026-04-03 16:05 |
| Director | xiaofeng |
| Status | done |

## Migration Plan

pycuda is a Python binding for the CUDA Driver API (not a PyTorch extension). It wraps:
- CUDA Driver API (`cuda.h`) via C++ (Boost.Python)
- cuRAND library (`curand.h`)
- OpenGL interoperability (`cudaGL.h`)
- Dynamic CUDA kernel compilation (nvcc → mcc)

Migration strategy:
1. Port C++ files in `src/cpp/` and `src/wrapper/` — replace CUDA types/functions with MUSA equivalents
2. Modify `setup.py` to find MUSA SDK instead of CUDA
3. Modify `pycuda/compiler.py` to use `mcc` instead of `nvcc`
4. Adapt `pycuda/driver.py` path detection for MUSA
5. Rename package to `pymusa` or provide compatibility shim

Key mappings:
- `cuda.h` → `musa.h`
- `curand.h` → `murand.h`
- `cudaGL.h` → `musaGL.h`
- `cudaProfiler.h` → `musaProfiler.h`
- All `cu*` driver API functions → `mu*`
- All `CU*` types → `MU*`
- `CUDA_*` error codes → `MUSA_*`
- `CUDAPP_*` macros adapted accordingly
- `nvcc` → `mcc` in compiler.py

## CUDA Dependencies

None — pycuda has no CUDA Python dependencies that need musifying.

## Pipeline Progress

### pycuda (Main)

| Stage | Status | Started | Completed | Notes |
|-------|--------|---------|-----------|-------|
| -1. ecosystem pre-flight | done | 2026-04-03 03:06 | 2026-04-03 03:08 | No MUSA counterpart found |
| 0. sdk-version | done | 2026-04-03 03:08 | 2026-04-03 03:12 | SDK 4.3.5, image 4.3.5_kuae2.1_20260119_ubuntu |
| 1a. reader (code analysis) | done | 2026-04-03 03:12 | 2026-04-03 03:20 | CUDA Driver API wrapper, Boost.Python |
| 1b. lookuper (document-site) | skipped | — | — | Not needed for driver-API project |
| 1c. lookuper (docker setup) | done | 2026-04-03 03:18 | 2026-04-03 03:20 | Container auto-musify-pycuda up, GPU 7 |
| 2. eco-support | done | 2026-04-03 03:20 | 2026-04-03 03:20 | No blocking deps |
| 3. musa-adapt | done | 2026-04-03 03:20 | 2026-04-03 16:05 | Build clean, GPU smoke test PASSED |

#### Stage Details

**Stage 0 — SDK Version Resolution (Core):**
- torch_musa version: N/A
- MUSA SDK version: 4.3.5
- Docker image: sh-harbor.mthreads.com/mcctest/musa-train:4.3.5_kuae2.1_20260119_ubuntu
- Container: auto-musify-pycuda

**Stage 1 — Parallel Preparation:**

*1a. Reader (code analysis):*
- Project type: Ext (Python extension via Boost.Python wrapping CUDA Driver API)
- CUDA files: 0 .cu, 0 .cuh (no kernel files — driver API wrapper only)
- C++ files: src/cpp/cuda.hpp (2260 lines), src/cpp/cuda.cpp, src/cpp/curand.hpp, src/cpp/cuda_gl.hpp
- Wrapper files: src/wrapper/wrap_cudadrv.cpp (1754 lines), wrap_cudagl.cpp, wrap_curand.cpp
- Python files: pycuda/compiler.py (nvcc), pycuda/driver.py, pycuda/gpuarray.py, etc.
- Key CUDA deps: cuda.h (Driver API), curand.h, cudaGL.h, cudaProfiler.h
- Build: setup.py + aksetup_helper.py, Boost.Python, no CUDAExtension

*1c. Lookuper (docker setup):*
- Container name: auto-musify-pycuda
- GPU visible: yes (GPU 7)
- MUSA SDK: 4.3.5
- mcc: available

**Stage 2 — Eco-support:**
- No blocking dependencies
- murand.h available in container (replaces curand.h)
- musaGL.h available (replaces cudaGL.h)
- All required MUSA Driver API present in musa.h

## Verification & Testing

### V4 Smoke Test (2026-04-03)

```
driver init OK
devices: 1
device 0: MTT S5000
free: 81252MB / total: 81837MB
memcpy test OK (htod + dtoh round-trip on 1000 float32 array)
GPU SMOKE TEST PASSED
```

Passed on GPU 7 (S5000) with MUSA SDK 4.3.5, Docker container auto-musify-pycuda.

## Issues

| # | File | Issue | Status |
|---|------|-------|--------|
| 1 | src/musa_cpp/musa.hpp | MUSA SDK 4.3.5 version = 40305 > 13000, wrongly selected 4-arg muCtxCreate path | Fixed: unconditional 3-arg muCtxCreate |
| 2 | src/musa_cpp/musa.hpp | muFuncSetBlockShape, muLaunch* (pre-CUDA 4.0) absent from MUSA SDK | Fixed: guarded with #if !defined(MUSA_VERSION) |
| 3 | src/musa_cpp/musa.hpp | muMemcpy2DUnaligned_v2 not exported from MUSA stub lib | Fixed: macro redirect to muMemcpy2D_v2 |
| 4 | src/musa_cpp/musa.hpp | initialize_profiler uses muProfilerInitialize (commented out in musaProfiler.h) | Fixed: guarded with #if !defined(MUSA_VERSION) |
| 5 | src/musa_wrapper/mempool.cpp | Wrong text substitution: converter → comterter from SimplePorting | Fixed: sed replacement |
| 6 | src/musa_wrapper/wrap_cudadrv.cpp | Wrong text substitution: converter → comterter in wrap_cudadrv.cpp | Fixed: sed replacement |
| 7 | src/musa_wrapper/wrap_cudadrv.cpp | MU_JIT_TARGET_FROM_MUCONTEXT doesn't exist; correct name is MU_JIT_TARGET_FROM_CUCONTEXT | Fixed: sed replacement |
| 8 | src/musa_wrapper/wrap_cudadrv.cpp | MU_JIT_INPUT_MUBIN doesn't exist; correct name is MU_JIT_INPUT_CUBIN | Fixed: sed replacement |
| 9 | src/musa_wrapper/wrap_cudadrv.cpp | muLinkDestroy/muLinkComplete absent from MUSA SDK | Fixed: Linker class disabled with #if !defined(MUSA_VERSION) |
| 10 | src/musa_wrapper/wrap_cudadrv.cpp | Legacy function bindings (set_block_shape, launch, etc.) for missing MUSA APIs | Fixed: guarded Python .def() calls with #if !defined(MUSA_VERSION) |
| 11 | setup_musa.py | PYGPU_PYMUSA=1 define missing, causing shared_ptr type mismatch in mempool.hpp | Fixed: added EXTRA_DEFINES["PYGPU_PYMUSA"] = "1" |

## Final Report

**Status: DONE**

pycuda successfully migrated to MUSA (pymusa) for SDK 4.3.5 / MTT S5000.

**What was ported:**
- All CUDA Driver API C++ headers (cuda.hpp) → MUSA equivalents (musa.hpp, musa_gl.hpp, murand.hpp)
- Python wrapper (wrap_cudadrv.cpp, mempool.cpp, wrap_curand.cpp) ported to MUSA API
- Build system (setup_musa.py) created for MUSA SDK instead of CUDA SDK
- Compiler (pycuda/compiler.py) updated: nvcc → mcc

**What was disabled (MUSA SDK gaps):**
- Pre-CUDA 4.0 legacy Driver API: muFuncSetBlockShape, muParamSet*, muLaunch, muLaunchGrid (never in MUSA)
- muProfilerInitialize (commented out in MUSA's musaProfiler.h)
- Linker class: muLinkComplete and muLinkDestroy not exported in MUSA SDK 4.3.5

**Smoke test:** import + muInit + device query + memcpy htod/dtoh round-trip — all PASSED on GPU 7 (S5000, 81GB).

## Knowledge Base

### 2026-04-03: pycuda architecture analysis
- pycuda is NOT a PyTorch extension — it wraps CUDA Driver API via Boost.Python
- No .cu kernel files — all CUDA code is C++ using Driver API
- Migration maps cu* → mu*, CU* → MU*, CUDA_* → MUSA_*
- general.json has complete CUDA Driver API mappings
- murand.h is available as curand.h replacement
- musaGL.h is available as cudaGL.h replacement
- Build uses aksetup_helper.py (custom schema), NOT setup.py + CUDAExtension pattern
- Must adapt setup.py to find MUSA_ROOT instead of CUDA_ROOT
- compiler.py must replace nvcc with mcc
