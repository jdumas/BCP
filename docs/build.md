# Build and run BCP

## One environment manager

All dependency/environment configuration is in **`pyproject.toml`**, managed by
Pixi. There is no separate `pixi.toml`, uv-managed environment, or uv task.
Pixi reads the standard Python dependencies and `pipeline` extra as PyPI
requirements, and installs the compiler, CMake, Ninja and native runtimes
from conda-forge. Linux/Windows also install MKL; macOS uses the system
Accelerate framework, with no MKL dependency. CPM still fetches pinned C++ source dependencies into the build
tree.

| Pixi environment | Platforms declared | Contents / runtime |
| --- | --- | --- |
| default | Linux x86-64, Windows x86-64, macOS arm64/x86-64 | CM/PT, platform sparse solver, core Python front-end |
| matlab | Linux x86-64, Windows x86-64, macOS arm64 | Full pipeline with direct MATLAB C engine |
| octave | Linux x86-64, macOS arm64 | Full pipeline with prebuilt Octave compatibility bridge |
| octave-source | macOS arm64 only, explicit opt-in | Full pipeline with private source-built, patched headless Octave |

The Linux native and complete MKL/Octave pipelines have been installed, built
and run successfully. Windows is a declared target, **not a claim of Windows
execution validation**. The Linux toolchain is pinned to GCC 13 because GCC 15 rejects
legacy iterator templates in CGAL 5.6; no diagnostic suppression is needed.

## Native CM/PT

From the repository root:

```sh
pixi run test
pixi run parameterize path/to/disk.obj outputs/disk_cm.obj
```

This configures a Release build (Accelerate on macOS, MKL elsewhere), runs the native and dependency-free Python
tests, and optionally writes an OBJ retaining the 3D geometry with `vt` texture
coordinates. Use your own input triangle disk; no sample data is required.

The task driver uses the active Pixi Python and passes dependencies under both
the environment prefix and its Windows `Library/` directory to CMake. It uses
argument arrays rather than shell-specific commands. On Windows, Visual Studio
C++ Build Tools must be installed/discoverable; the Conda compiler package does
not install the licensed MSVC compiler itself.

Build products are separated by solver/runtime, for example
`build/pixi-mkl-native/bin/`.

## macOS (Apple Silicon and Intel)

Install Xcode Command Line Tools (`xcode-select --install`) and Pixi, then use
the same commands above. Both CMake and the Pixi task driver default to
`ACCELERATE` on macOS, with separate outputs under
`build/pixi-accelerate-native/bin/` or `build/pixi-accelerate-octave/bin/`.
No MKL installation, license, or Homebrew BLAS replacement is needed.

For the MATLAB-free full CPU pipeline on Apple Silicon:

```sh
pixi run -e octave test
pixi run -e octave parameterize path/to/disk.obj outputs/disk_bcp.obj --device cpu --gif-device cpu
# Optional Torch-on-CPU energy evaluator:
pixi run -e octave test --gif-backend torch
```

The full environment installs Octave from conda-forge and Python packages
(including a source build of OpenMesh) through Pixi. It is not declared for
Intel Macs because current PyTorch >=2.5 wheels are Apple-Silicon-only; the
native CM/PT environment supports both architectures. CPU execution is the
macOS path: neither CUDA nor Metal/MPS is implemented by this pipeline.
`test-gpu` requires CUDA and is not a macOS validation command.

The GIF build uses CGAL 5.6.3, which fixes iterator code rejected by modern
Clang. A GIF-only `-Wno-register` permits GMM 4.2's legacy storage-class syntax
under Clang/C++17; it does not change the numerical algorithm.

**Validation on Apple Silicon:** `pixi run test` passes all 6 CTest checks,
including real Accelerate solves and the native CM/PT smoke test. The prebuilt
Octave environment installs and the full GIF executable compiles. However,
its live Octave numerical and full-pipeline tests fail at interpreter startup
in the current nono sandbox:
`fatal: unable to find current directory`. The Conda Octave 10.3 build uses
gnulib's replacement `getcwd()`, which opens/enumerates parent directories;
`nono why --self` confirms ancestors outside the workspace grant are denied.
The system `getcwd()` succeeds, but Octave's wrapper fails with `EPERM`.
No sandbox bypass or broad filesystem grant has been applied. The optional
source-build path below addresses this without changing permissions. Intel
Mac execution and MATLAB on macOS remain unverified.

### Opt-in patched Octave source build (macOS arm64)

The default `octave` environment **continues to use the prebuilt Conda package**.
No configure/build/test task in that environment triggers an Octave rebuild.
To explicitly select the source-built alternative:

```sh
# First run downloads, builds, installs privately, and checks the interpreter.
pixi run -e octave-source build-octave
# These also build it on first use, then reuse the validated installation:
pixi run -e octave-source test
pixi run -e octave-source parameterize path/to/disk.obj outputs/disk_bcp.obj --device cpu --gif-device cpu
# Optional Torch-on-CPU evaluator:
pixi run -e octave-source test --gif-backend torch
```

This builds the official Octave **10.3.0** release, checked against a pinned
SHA-256, with `scripts/patches/octave-10.3.0-macos-getcwd.patch`. The Apple-only
patch makes gnulib try native `getcwd()` even if configure classified it as
unusable. A native permission error is returned, not retried by traversing
ancestors. Other platforms' code is unchanged; this is not a loader shim,
modified `$PWD`, root-working-directory trick, or filesystem permission grant.

Pixi supplies the C/C++/Fortran compilers, make, patch, pkg-config, generation
tools and native library dependencies. The prebuilt Octave package remains
installed as a dependency provider; **none of its files are patched or
replaced**. The private interpreter is headless (no GUI, Java, audio or docs),
uses Conda BLAS/LAPACK and retains the sparse/complex numerical functionality
needed by GIF. BCP's native sparse solves still use Accelerate.

Builds and logs live under `build/octave-source/<fingerprint>/`. A build is
reused only after installation and real `pwd()`, sparse-solve and JSON
round-trip smoke tests in the original workspace pass. The fingerprint includes
the archive, patch, builder, configuration, compiler flags and Conda package set. Source download
and archive extraction are checked; a failed build never falls back to the
prebuilt interpreter. Interrupted builds can resume with the same inputs.

Compilation can take substantial time and disk space. The default is four
parallel jobs; use `pixi run -e octave-source build-octave --jobs 8` or set
`BCP_OCTAVE_BUILD_JOBS=8` to change it. Run only one source build at a time.
The private installation depends on its Pixi environment; do not move it or
remove that environment while using it. Remove `build/octave-source/` to
explicitly discard all source build artifacts.

The source environment passes its exact interpreter path to both CMake and
the GIF runtime, including numerical/reference tests, and uses a separate
BCP build directory: `build/pixi-accelerate-octave-source/`. Merely having a
source build on disk never changes the ordinary `octave` environment.

**Validated on Apple Silicon inside the unchanged nono sandbox:**
`pixi run -e octave-source test` passes **12/12 CTest checks**. This includes
live engine/error/timeout tests, both complete BCP variants (IDT disabled and
enabled) with the native Octave and Torch-on-CPU evaluators, Torch reference
comparisons, and the Accelerate CM/PT numerical tests. All **62 Python unit
tests pass with no skips**. The private interpreter's `pwd()`, JSON and sparse
solve smoke checks pass in the original workspace. Library linkage was checked
to use the private source-built Octave libraries, not the prebuilt ones.
A subsequent test invocation reused the validated interpreter without compiling
it again. No additional filesystem grants, cwd relocation or loader injection
were used. These are CPU correctness results, not a macOS performance-parity
or GPU-support claim.

## Full pipeline: preserve the original fast MATLAB path

MATLAB R2018a or newer must already be installed and licensed. Pixi cannot
distribute/install MATLAB. Set `Matlab_ROOT_DIR` to its installation directory
when automatic SDK discovery is insufficient. On Linux, the MATLAB C engine
also requires the C shell specified by MathWorks.

```sh
pixi run -e matlab test
pixi run -e matlab parameterize path/to/disk.obj outputs/disk_bcp.obj
```

PowerShell example for selecting MATLAB:

```powershell
$env:Matlab_ROOT_DIR = 'C:\Program Files\MATLAB\R2022b'
pixi run -e matlab test
```

This reuses the original in-memory MATLAB engine/matrix API. It does not impose
the Octave bridge's MAT-file transfers. The native CLI finds numerical scripts
by their configured absolute source path, not by the process working directory.
The task driver adds the MATLAB runtime paths for subprocesses.

The full `test` task runs CTest, the Python unit tests, and a live generated-mesh
integration test exercising simplification, CM, PT, prolongation and GIF, both
with and without IDT. It fails on missing dependencies/runtime; it does not
replace a failed solver with another method.

## Optional Linux Octave path

```sh
pixi run -e octave test
pixi run -e octave parameterize path/to/disk.obj outputs/disk_bcp.obj --device cpu
```

The conda-forge Octave recipe currently
[skips Windows](https://github.com/conda-forge/octave-feedstock/blob/main/recipe/recipe.yaml).
The compatibility adapter is also POSIX-only. The Windows full-pipeline route is
the direct MATLAB engine, not a purported all-Pixi Windows Octave environment.

Octave keeps the original Newton/line-search scripts. Its default evaluator is
the original CPU code; an optional PyTorch evaluator implements the analytic
energy/gradient/Hessian on CPU or CUDA. This alternative has numerical CPU
and CUDA correctness validation on Linux. CUDA speed and paper-scale memory
headroom have not been benchmarked.

The full Python dependency set includes OpenMesh 1.2.1, which may need a source
build. Its conda-forge recipe skips Windows Python >3.9, and the PyPI source build
on current Windows Python is **not validated**. The Windows Python pipeline must
pass the live gate before claiming Windows end-to-end support.

## GPU paths: preserve the original kernels

The original Python code hard-coded `cuda:0`. The original MATLAB Newton script
also set `use_GPU = true` unconditionally (the later face-count condition did not
turn it off). It used `gpuArray` for energy/gradient/Hessian evaluation, then
`gather` before the CPU Cholesky solve. Those numerical kernels remain unchanged.

Device choices are explicit and independent:

| Control | Applies to | Values |
| --- | --- | --- |
| `--device` / `BCP_DEVICE` | Python tensor calculations and prolongation | `auto`, `cpu`, `cuda:0` |
| `--gif-device` / `BCP_GIF_DEVICE` | GIF energy evaluator device | `auto`, `cpu`, `gpu` |
| `--gif-backend` / `BCP_GIF_BACKEND` | GIF energy evaluator implementation | `auto`, `native`, `torch` |

An explicit CPU tensor selection does not probe CUDA. Required CUDA/GIF GPU
selections fail if unavailable; they never silently become CPU computations.
The native code passes GIF's policy into the engine explicitly, even when a
Windows MATLAB COM server was started before the CLI. The device is selected
once per run. MATLAB reuses the existing `gpuArray` branch. Octave GPU requests
select the PyTorch evaluator; ordinary CPU/auto requests retain the native
Octave evaluator unless `--gif-backend torch` is explicitly selected.

Original-style GPU execution requires MATLAB with Parallel Computing Toolbox,
a working CUDA device, and a CUDA-enabled Torch installation:

```sh
pixi run -e matlab parameterize path/to/disk.obj outputs/disk_gpu.obj --device cuda:0 --gif-device gpu
pixi run -e matlab test-gpu
```

MATLAB-free CUDA evaluation is wired through the optional Torch backend:

```sh
pixi run -e octave parameterize path/to/disk.obj outputs/disk_gpu.obj --device cuda:0 --gif-device gpu
pixi run -e octave test-gpu
```

Both runtime environments' `test-gpu` tasks now require CUDA for the tensor and
GIF energy stages. Octave additionally compares its original CPU reference
against the CUDA Torch evaluator; no CPU fallback or GPU emulation is accepted.
The native Octave evaluator alone still has no `gpuArray` implementation.

To validate the same Torch formulas/bridge without requiring GPU access:

```sh
pixi run -e octave test --gif-backend torch
pixi run -e octave parameterize path/to/disk.obj outputs/disk_torch_cpu.obj --device cpu --gif-device cpu --gif-backend torch
```

`torch_energy.py` ports only the analytic E/G/H evaluator in float64/complex128;
it does not use autograd Hessians or change Newton, line searches, scaffold
remeshing, or the CPU Cholesky/MKL solves. In the Octave build,
`bcp_torch_energy_native.oct` embeds CPython + NumPy in the Octave process and
invokes `torch_energy.py` directly, with no worker subprocess, pipe IPC, or
JSON/byte serialization per iteration. Jacobians, area/fixed-variable data and
constant derivative blocks stay in one persistent device cache per scaffold
(prepared via Fortran-order NumPy views of the Octave input buffers). Inner
steps upload only the current free coordinate vector `x` and write requested
outputs directly into Octave-owned buffers after reading dynamic foldover shapes
from device metadata, avoiding any intermediate `evaluate() -> NumPy -> memcpy`
host pass on $O(F)$ outputs. Only the Hessian block consumed by the CPU solve
is downloaded (selected on-device after the full analytic calculation, so its
entries are unchanged). The native dispatcher retains only a prepared marker,
not host Jacobians from old scaffolds; the CPU reference fallback retains the
inputs it needs. Borrowed input views are read-only and output views are writable.
The native module stays locked for the Octave process lifetime so Python capsule
callbacks cannot outlive its code; `close` still releases the evaluator/device cache.

**Data movement and layout ledger (not a blanket zero-copy claim):**
- **Input wrapping vs device/cache copies:** `prepare` wraps Octave input
  matrices (`J_fz`, `J_fbz`, `area`, `fixed_indices`, `fixed_values`, optional
  `hessian_indices`) as Fortran-order NumPy views with `astype(np.float64, copy=False)`
  so already-`float64` real inputs (`area`, `x`, etc.) are not duplicated into
  an extra NumPy host buffer before `torch.tensor(...)` copies/uploads them into
  its own persistent device/cache storage.
- **Output downloads:** 1D/vector outputs (`foldovers`, `fz`, `fbz`, `G`) are
  contiguous and copy directly from device tensors into wrapped Octave output
  buffers without an intermediate `evaluate() -> NumPy -> memcpy` host pass;
  `1x1` scalars (`E`, or single-face outputs) use a 1-element staging cell
  because Octave stores scalars without a shareable matrix buffer.
- **H-only device packing:** the selected 2D Hessian block `H` is produced in
  row-major order on the device while Octave owns column-major (`Fortran`-order)
  matrix storage, so `_copy_device_into` performs a device-side transpose
  gather (`source.t()`) into a contiguous device temporary before issuing a
  single contiguous device-to-host copy into Octave's column-major buffer (no
  host transpose intermediate, but an explicit device-side layout-packing pass
  on `H`).
- **Unavoidable layout work in other stages:** Octave engine MAT v5 transfers
  in `OctaveInterface.cpp` borrow column-major `double` setter buffers directly
  (`MAT_F_DONT_COPY_DATA`) and decode getters straight into caller destinations,
  but MAT v5 still requires split real/imaginary staging for complex matrices,
  row/column-major conversion when caller layout is row-major, and `matio`'s
  internal file-read buffer in `Mat_VarRead`. Likewise, `PardisoLinearSolver`
  passes distinct `GMMDenseColMatrix` RHS/solution buffers directly to the
  compatibility solver backends while keeping an internal copy fallback only
  when RHS and solution alias (or on the Panua path).

MKL PARDISO and native PT remain CPU algorithms. Keeping the original division
of work avoids introducing a new GPU sparse solver or rewriting the optimizer.
Numerical correctness is not a licensed MATLAB/Panua runtime-speed parity claim;
use the explicit comparison modes below to separate license-free reference checks
from Linux/CUDA placement runs.

## Explicit comparison modes (`reference` vs `parity`)

Comparisons must not rely on implicit `auto` defaults (which select CPU for
Octave GIF when `--gif-device` is omitted). `scripts/compare.py` and the Pixi
tasks provide two named presets outside CTest's single-thread smoke limits:

- **`reference` (`pixi run -e octave compare-reference`)**: runs the unchanged
  original `.m` logic through Octave on CPU (`--device cpu --gif-device cpu --gif-backend native`)
  with the platform sparse solver (`MKL` on Linux/Windows, `ACCELERATE` on macOS).
  This is a license-free numerical reference, **not** an original GPU-placement
  benchmark.
- **`parity` (`pixi run -e octave compare-parity`)**: requires Linux x86-64,
  working `cuda:0` for both Python/Torch prolongation and the GIF Torch energy
  evaluator (`--device cuda:0 --gif-device gpu --gif-backend torch`), and `MKL`
  for CPU sparse solves. Missing CUDA or a non-MKL solver fails immediately;
  there is no automatic CPU fallback.
- **`both` (`pixi run -e octave compare --mode both`)**: runs both modes on a
  supplied `--input` mesh (or a deterministic curved disk fixture) and compares
  final UVs using **only proper 2D rigid gauge alignment** ($SO(2)$ rotation +
  translation, no rescaling and no reflection), alongside Symmetric Dirichlet
  energy, quasi-conformal distortion, and local injectivity.

All GIF stage and total timings use `CGAL::Real_timer` wall-clock semantics so
child interpreter and GPU work are included on POSIX. Untimed diagnostic flags
can be enabled for separate validation runs without adding copies or I/O to
normal timed runs. Reports label these as `validation_with_diagnostic_io`:
wall/stage timings in those runs include trace/dump I/O and are not pure solver
benchmarks. Use runs without these flags for performance measurements:

- `--trace` (sets `BCP_COMPARE_TRACE_PATH`): emits per-step Newton energy,
  gradient norm, step norm, line-search step size, foldover count, and per-remesh
  summaries. Because end-to-end `reference` vs `parity` runs differ in both
  preprocessing (`pred_M` float32 staging on CPU vs CUDA) and near-zero `|fbz|`
  Hessian conditioning on Tutte initializations, end-to-end trace comparisons are
  explicitly labeled `confounded_by_preprocessing: true` and do not claim strict
  step-by-step trajectory parity; instead, `tests/controller_reference.m` (also
  run by `compare.py --trace`) replays `iterationOfNewtonWithScaffold.m` on shared
  exact inputs under a strict `rtol=1e-7, atol=1e-9` envelope.
- `--kkt-diagnostics` (sets `BCP_COMPARE_KKT_DIR`): dumps each full KKT or
  nonsymmetric MVC repair matrix (`matrix.mtx`), solver solutions (`batch_*.mat`
  or `solution.mat`), and zero-based `metadata.json`, then delegates verification
  to `scripts/check_kkt.py` for normalized backward-error
  ($\|AX-B\|_F / (\|A\|_F \|X\|_F + \|B\|_F)$) and independent SuperLU checks.
- `--golden-uv path/to/golden.{obj,mat}`: compares outputs against an externally
  supplied baseline. Without golden baseline evidence from the licensed MATLAB/Panua
  runtime on the same hardware, structured JSON reports explicitly state that
  original MATLAB/Panua runtime parity is **not** certified.

GPU execution is now validated on an NVIDIA RTX 5000 Ada Generation with
Torch 2.15.0+cu132, CUDA runtime 13.2 and NVIDIA driver 595.91.07. Before GPU
activation, Torch initialization returned error 304 with a denied
`/proc/self/task/<tid>/comm` thread-naming write.
Nono 0.80.0 includes the scoped supervisor fix described in
[issue #924](https://github.com/nolabs-ai/nono/issues/924), but it is activated
by the **CLI `--allow-gpu` flag**, not just the profile's `allow_gpu: true`
permission. The initial session lacked that flag; restarting with it resolved
CUDA initialization and the strict GPU suite passed. Manually allowing NVIDIA
device files alone does not activate the procfs broker.

Restart with GPU activation before running the strict test:

```sh
nono run --profile pi --allow-gpu --allow-cwd -- pi
# Inside the new session:
pixi run -e octave test-gpu
```

No blanket `/proc` write grant or sandbox bypass is required for this fix.
No nono version upgrade or additional procfs permissions were needed.

A separate, non-blocking warning remains in the current sandbox: PyTorch cannot
create `~/.cache/torch/kernels`, so its persistent CUDA kernel cache is disabled.
The GPU correctness tests pass regardless; repeated-process warm-up performance
may benefit from a reviewed write grant to `~/.cache/torch`. The directory must
exist before launch. This cache grant has not been applied automatically.

The regular full-pipeline integration gate deliberately requests CPU for both
stages. Its passing results do not stand in for the strict GPU gate. MATLAB
script errors now propagate through the existing output buffer rather than
retrying with a reset/stale workspace; no extra per-command IPC roundtrip was
added for that check.

## Sparse solver backends

CMake selects one explicit backend:

| `BCP_LINEAR_SOLVER` | Behavior |
| --- | --- |
| `ACCELERATE` (macOS default) | Apple sparse Cholesky / pivoted LDLT; QR for general matrices |
| `MKL` (Linux/Windows default) | Separate Intel oneMKL PARDISO implementation, LP64 |
| `PANUA` | Original PARDISO library and selected-inversion path |
| `EIGEN` | Explicit portability/testing fallback, not a performance substitute |

A missing requested backend is a configuration error. There is **no silent
Eigen fallback**. MKL uses its real API and private controls/handles, not Panua's
incompatible `iparm` values or exported symbol names. CSR data is passed directly,
without Eigen conversion, and symbolic analysis is reused for numerical
refactorizations. GNU builds use MKL's GNU OpenMP layer; other non-Intel
compilers use parallel TBB to avoid loading a second OpenMP runtime alongside
MSVC/Clang's runtime. MKL threading remains configurable with `MKL_THREADING`.

Accelerate implements the same private compatibility ABI, not a PARDISO
library masquerading under Intel/Panua symbols. It translates one-based CSR
indices to the framework's zero-based CSC representation. Symmetric upper CSR
is lower CSC without copying/reordering numeric values; general matrices use
a cached CSR-to-CSC permutation. Symbolic analysis is reused for numerical
refactorization, with Cholesky for SPD systems and pivoted LDLT for indefinite
KKT systems. General real matrix types use sparse QR, which works on macOS
versions before the sparse LU API introduced in 15.5. Panua-specific controls
are not passed to Accelerate, and failed refactorizations invalidate old factors.

The Panua backend retains its selected-inverse algorithm and sparsity pattern.
Accelerate/MKL/Eigen instead reuse the KKT factors and solve bounded batches of the needed
cone columns; the batch size is capped by a per-buffer memory budget.
Mathematical equivalence does not establish performance parity: realistic mesh
benchmarks against the original backend are still required. Thread counts are
not forcibly reduced to one in production; tests explicitly constrain OpenMP to one thread.

Manual CMake configurations (normally Pixi tasks supply the environment):

```sh
# macOS (also selected by default there):
cmake -S . -B build/accelerate -G Ninja -DBCP_LINEAR_SOLVER=ACCELERATE
# Linux/Windows:
cmake -S . -B build/mkl -G Ninja -DBCP_LINEAR_SOLVER=MKL
cmake -S . -B build/matlab -G Ninja -DBCP_BUILD_GIF=ON -DBCP_SCRIPT_ENGINE=MATLAB -DBCP_LINEAR_SOLVER=MKL
cmake -S . -B build/octave -G Ninja -DBCP_BUILD_GIF=ON -DBCP_SCRIPT_ENGINE=OCTAVE -DBCP_LINEAR_SOLVER=MKL
```

For licensed Panua, use `-DBCP_LINEAR_SOLVER=PANUA` and provide
`BCP_PARDISO_LIBRARY` / `PARDISO_BIN`. MATLAB/CGAL may also need a
`CMAKE_PREFIX_PATH` or `Matlab_ROOT_DIR` hint.

An intentionally dependency-light fallback can be tested using the installed
system compiler:

```sh
cmake -S . -B build/eigen -G Ninja -DBCP_LINEAR_SOLVER=EIGEN
cmake --build build/eigen --parallel 4
ctest --test-dir build/eigen --output-on-failure
```

## Validation status and limits

macOS results and its current sandbox/runtime limitation are listed above.
The Linux results below were recorded before this macOS port; they have not
been rerun on Linux in the macOS session.

- `pixi run test` (Linux): **5/5 CTest checks pass**, including real MKL-backed CM/PT
  numerical tests and core Python tests. The optional Torch math and bridge tests skip in
  this lightweight environment because Torch/Octave are not installed there.
- `pixi run -e octave test`: **15/15 CTest checks pass, no skips**. The full
  environment runs all Python unit tests (including finite-difference gradient
  and unprojected-Hessian checks for the analytic Torch implementation, KKT checker
  tests, comparison harness tests, and live in-process Octave/Torch bridge tests),
  plus `pardiso_linear_solver`, `octave_transport`, `controller_reference`, and
  `compare_reference_diagnostics`.
- `pixi run -e octave test-gpu`: **15/15 CTest checks pass, no skips**, with
  `cuda:0` and the `inprocess` native bridge (`bcp_torch_energy_native.oct`)
  confirmed for both Python tensor processing and the GIF Torch energy
  evaluator. Both IDT variants produce finite, non-flipped UVs. The 65
  reference comparisons and 25-step matched controller replay pass with the
  Torch evaluator running on CUDA.
- Live Octave parity validates 65 evaluations in `torch_energy_reference.m` plus
  the dedicated bridge suite (`test_octave_torch_bridge.py`), including full-output
  comparisons and the selected Hessian block:
  fixed/all-fixed variables, foldover filtering, zero-fbz conventions and both
  Hessian modes. It verifies in-process evaluator reuse, exact upload/download
  byte ledgers, output buffer ownership, and that `evaluate` does not re-upload
  the Jacobians.
- Both complete BCP variants (IDT disabled/enabled) pass with the native Octave
  evaluator and with the in-process Torch evaluator on CPU and CUDA, producing
  finite UVs and no flipped/collapsed triangles.
- Validated versions include GCC 13, oneMKL 2026.1, Octave 10.3, Python 3.11,
  NumPy 1.26.4 and libigl 2.6.3. Runtime linkage was checked: MKL LP64/core/GNU
  threading and a single libgomp come from the project-local Pixi environment.
- The separate stub-MKL contract test checks API/control translation; it is not
  being substituted for the real MKL numerical tests above. Native tests also
  exercise `PardisoLinearSolver` direct multi-RHS/aliased solves, factor reuse
  and solver diagnostics (`pardiso_linear_solver`), and `OctaveInterface` borrowed
  MAT setter / direct getter transfers (`octave_transport`).
- Script-engine regressions cover chunked/split-sentinel output, large complex
  MAT roundtrips, no-op `bcp_torch_energy('close')` without initializing Python/CUDA,
  and explicit failure on unstarted `stats`.
- Manual compile-only Octave builds without an interpreter still report the
  live numerical test as **Skipped**. Full Pixi test environments require a real
  runtime and cannot silently skip their integration gate.
- The proprietary MATLAB SDK/engine, Windows execution, paper-scale GPU memory
  headroom and Panua-versus-MKL performance parity have not been validated here.
- Inputs must be connected, consistently oriented, nondegenerate triangle disks.
  Native indices are currently 32-bit/LP64; MAT v7.3/HDF5 is not enabled.
  CM retains basic vertex/index/area and boundary checks, not a separate
  O(F log F) full-topology validation pass.
- No paper-scale performance or numerical reproduction claim is made. CM-only
  execution is **not** a complete BCP result. The original academic-use restriction
  and third-party licenses still apply.

## Failure handling and timeouts

Native calls inherit the caller's OpenMP/MKL/Accelerate thread environment
without injecting a single-thread default. Correctness tests cap
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` explicitly.
On Linux/POSIX, each native invocation has its own process group; timeout or
interruption terminates that group, including the Octave/Torch descendants.
Windows retains direct-child termination; its process-tree behavior is not
validated by the Linux tests. MATLAB teardown on Windows detaches the output
buffer without quitting a potentially shared COM session; POSIX closes its
private engine process.

Failures abort with diagnostics rather than restarting an empty Octave workspace.
An explicit engine restart is supported, but callers must rebuild all algorithm
state. The Octave adapter implements the dense/complex and sparse-write subset
needed by GIF; unused sparse reads fail explicitly instead of carrying a second
untested sparse-conversion implementation. Its private numeric transfers use
uncompressed MAT v5 files, and output framing scans only newly received data
plus the possible split-sentinel overlap. Public result formats are unchanged.

Timeout controls cover different scopes:

- `BCP_NATIVE_TIMEOUT_SECONDS`: optional finite, positive override for the whole
  native subprocess. Defaults are CM 3600 s and GIF/PT 300 s; PT is newly bounded.
  This includes startup and every solver iteration.
- `BCP_OCTAVE_TIMEOUT_MS`: per-engine-evaluation deadline, default 600000 ms
  (10 min), maximum 86400000 ms.
- Retired worker-pipe read timeout (`BCP_TORCH_TIMEOUT_SECONDS`): removed
  together with `torch_energy_worker.py` when the in-process native Torch
  bridge (`bcp_torch_energy_native.oct`) replaced the worker subprocess.
  Timeout enforcement is governed by the outer `BCP_OCTAVE_TIMEOUT_MS` and
  `BCP_NATIVE_TIMEOUT_SECONDS` watchdogs, which terminate the Octave/native
  process group on timeout rather than promising asynchronous mid-kernel
  cancellation of an in-process CUDA call.

For a deliberately longer run, configure both the whole-process deadline and
`BCP_OCTAVE_TIMEOUT_MS`; a longer inner deadline does not override a shorter
outer timeout.

## Runtime overrides and command protocol

`BCP_BIN_DIR`, `BCP_CM_EXE`, `BCP_PT_EXE`, `BCP_GIF_EXE`,
`BCP_CACHE_DIR`, `BCP_DEVICE`, `BCP_GIF_DEVICE`, `BCP_GIF_BACKEND`,
`BCP_PYTHON_EXECUTABLE`, `BCP_OCTAVE_EXECUTABLE`, `BCP_TORCH_NATIVE_DIR`, and
`BCP_TORCH_MODULE_DIR` override discovery without source edits. CUDA is selected when available; the manifest no longer
forces CPU-only Torch wheels. A hardware-specific Torch index can be configured
through Pixi when needed.

```text
Parameterization_cmd input.obj output.obj output.mat metadata.txt [target_energy] [target_seconds] [initial_uv.obj]
PT source_matrices.mat result.mat 1
```

CM can omit OBJ output with an empty argument and writes MAT v5 `uvs` plus
metadata; its native OBJ output is flattened geometry. PT's dense-double MAT
inputs use zero-based mesh indices. Subprocess failures/timeouts propagate and
unique temporary work directories are cleaned up.
