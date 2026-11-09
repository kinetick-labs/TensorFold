# In-model runner image layers

Two layers over a serving image: the circuit frontend (`build.sh`) and the runner's assets (`build-assets.sh`).

## Circuit frontend

Build `zig build runner-frontend-check` with the repository's pinned Zig. Then run:

```sh
packaging/runner/build.sh BASE_IMAGE TAG zig-out/bin/tf-runner-frontend-check
packaging/runner/check-image.sh TAG
```

Use a derived serving base that already contains the engine and its lane Python layer. The layer adds Ubuntu 24.04's clang-18 and lld-18, with `/usr/bin/clang` and `/usr/bin/ld.lld` aliases. No cross libc, RISC-V emulator or Python dependency is added for compilation. Host Python is used only to prepare the build context and report image sizes. The builder preserves the base image's entrypoint arguments and CMD; it does not replace engine binaries. Source provenance can be set with `TF_RUNNER_FRONTEND_TREE`; `TF_RUNNER_BUILD_ID` labels task-owned build containers for the shared-node pause guard.

At every container boot, `tf-runner-frontend-check` compiles a tiny freestanding C program through the production frontend and validates its RV32IM+Zba ELF. It never runs that ELF on the host. The Docker build runs the same check with `--require` so an image without a functioning RISC-V toolchain is rejected.

An image that ships a circuit also proves the route the compiled programs take. `tf-runner-frontend-check --require --circuit PATH` compiles the cells in `zig/src/core/runner_probe.zig` (Python loops with `*`, `//` and `%`, and a C string read from ROM bytes) and runs them on that circuit; any output but the expected one, or a run past 100,000 steps, exits 1. `check-image.sh` runs it on `/opt/tensorfold/share/runner/calc.tfpc` when the image has one, and a serving image's Dockerfile should run it as a `RUN` step after copying its circuit. The engine runs the same probe when it opens the runner (`TF_RUNNER=1`) and refuses to start on a wrong output (about a second at boot).

At runtime, a missing or broken compiler/linker reports unavailable and allows ordinary boot when `TF_RUNNER` (or a family's alias, such as DeepSeek-V4.1's `TF_DSV41_RUNNER`) is unset, empty, 0, false, off or no. Any other runner value requires a successful check and fails boot otherwise. `TF_RUNNER_CLANG` and `TF_RUNNER_LLD` override compiler/linker paths for deployment and diagnostics. The production frontend also independently abstains if compilation is unavailable; boot does not turn a missing optional dependency into model output.

Runner threads (each cell's translation, which also starts clang and ld.lld, and each circuit run) leave the engine thread's CPU pin when they start. `TF_RUNNER_CPUS` sets where they run: `auto` (default) takes the container's cpuset, removes the CPUs the engine pins (the TP plan link's, `TF_TP_PLAN_PIN`, and the RoCE proxy's, `TF_TP_ROCE_CPU`) and, when core sizes differ (smallest capacity under 80% of the largest), keeps the small cores; on a GB10 rank with the plan link on CPU 19 that is 0-4,10-14. `inherit` keeps the previous behaviour (the engine thread's CPU); a list such as `0-4,10-14` is used as given. `TF_RUNNER_NICE` (default 10, 0-19) is the least niceness they run at; a higher inherited niceness is kept. The engine logs the resolved set at the first run (`in-model runner threads: CPUs ...`), and the threads are named `tf-runner`. `zig build test-runner-host` checks the placement; `zig build bench-runner-place` measures an engine host-loop stand-in beside the runner at full load (`runner-place-bench ENGINE_CPU SECONDS WORKERS ROUNDS`, `TF_RUNNER_CIRCUIT` for a real circuit).

Exact decimal/fraction arithmetic is separately gated by `TF_RUNNER_EXACT` (default off), through the engine's frontend context. Default Python output remains CPython-identical: unsupported floating, decimal and rational output abstains. Installing the toolchain does not enable either runner knob.

Build on a CPU node at nice 19. On shared nodes, stop the build session and pause its labelled Docker build containers while QUIET exists; stopping only the Docker client does not stop daemon-owned build work. No GPU access is needed. `check-image.sh` uses the runc runtime, no network and no GPU mounts, and checks normal/required boot, missing clang/LLD, successful compiler without ELF, and argument preservation. It never launches the engine.

The builder emits a JSON receipt with tag, immutable image/base IDs, sizes and size delta reported by `docker image inspect .Size`. Docker storage backends may report additional unpacked disk usage in `docker image ls`; record that separately when budgeting local storage. The base engine revision label is inherited; the frontend source tree is recorded separately.

## Assets

```sh
packaging/runner/build-assets.sh RUNNER_IMAGE TAG [ASSETS_DIR]
```

`RUNNER_IMAGE` is an image `build.sh` built. The layer adds the runner circuit at `/opt/tensorfold/share/runner/calc.tfpc` and the graph bank at `/opt/tensorfold/share/runner/graph-v7-powmod/bank/` (its MIT notice also under `/opt/tensorfold/licenses/runtime-deps/`); point `TF_RUNNER_CIRCUIT` and `TF_RUNNER_GRAPH_BANK` (`.../bank/bank.json`) at them. The build runs the circuit-route probe and fails on a wrong output.

`assets.sha256` pins exactly the files the layer takes, each at its sha256: the circuit and the graph-v7-powmod bank DeepSeek-V4.1's AB3 image served. `assets.release` names the archive they come from, DeepSeek-V4.1's `runner-assets-20261107` release (one deterministic tar.gz at its sha256); it is used where it is, not re-hosted. Without `ASSETS_DIR`, `assets.py fetch` downloads that archive, checks its sha256, unpacks only plain files and directories and checks every file against `assets.sha256`. With `ASSETS_DIR`, `assets.py check` compares the directory with `assets.sha256` before the build. `RUNNER_ASSETS_URL` names a mirror; it must serve the same bytes. `pytest packaging/runner` checks the fetch and the check offline.
