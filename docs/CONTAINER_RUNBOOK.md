# CALO-RPD Studio container runbook

The repository provides two mutually exclusive runtime profiles built from the same source:

- `cpu`: CPU PyTorch only. CUDA is hidden and startup fails if it is unexpectedly visible.
- `cuda`: NVIDIA CUDA PyTorch. Startup fails unless the assigned GPU is available to PyTorch.

**Phase 4 boundary:** container development and manual validation must prove clean empty-policy
startup and must not package, download, generate, train, evaluate, qualify, register, activate, or
delete any policy. Existing policy artifacts are non-final and excluded from images and release
state. A future newly qualified policy, if Phase 5 includes one, requires a separate immutable
manifest and is not part of the Phase 4 image contract.

The Linux/amd64 Python 3.11 slim-bookworm base is pinned by OCI manifest digest. Runtime dependency
graphs are separately resolved for CPU and CUDA 12.8 against Python 3.11/manylinux 2.28, with every
distribution protected by SHA-256 and installation forced through `--require-hashes`.

Both profiles expose the full Qt application through noVNC at <http://localhost:6080/vnc.html> and
use the same named `calo-rpd-studio-runtime` volume for databases, policies, checkpoints, results,
exports, and cross-container CUDA device leases. Set `CALO_RUNTIME_VOLUME` to an isolated name for
a qualification run; every container that can address the same GPU must use that same volume.

The browser port binds to `127.0.0.1` by default and is not exposed to the local network. The
underlying VNC server also listens only on the container loopback interface. The application runs
as unprivileged UID/GID 10001, drops Linux capabilities, prevents privilege
escalation, and uses a read-only root filesystem with a bounded temporary filesystem. The `/data`
volume is the only persistent writable location.

## CPU profile

```text
docker compose --profile cpu build cpu
docker compose --profile cpu up cpu
```

Open <http://localhost:6080/vnc.html>. The container cannot see an NVIDIA device and therefore uses
system RAM. The application admission governor budgets at most 80% of memory that is currently
available at admission time; it does not calculate 80% of installed RAM.

## NVIDIA CUDA profile

The host must have a current NVIDIA driver and working Docker GPU passthrough. On Windows, use
Docker Desktop with the WSL2 backend and verify GPU passthrough before starting CALO:

```text
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
docker compose --profile cuda build cuda
docker compose --profile cuda up cuda
```

The CUDA profile assigns device `0` by default. Select another single device explicitly with
`CALO_GPU_DEVICE=<device-id>`; do not expose additional GPUs to increase an individual run's
scientific budget.

Open <http://localhost:6080/vnc.html>. The CUDA image uses the official PyTorch 2.10 CUDA 12.8 wheel,
and Compose exposes exactly the selected NVIDIA device. CALO admits work against 80% of VRAM
currently free at admission time. A CUDA out-of-memory condition first reduces the active
microbatch and retries on CUDA. It never moves a partially executed numerical operation to CPU. A
not-yet-executed task may restart through the explicitly enabled, provenance-recorded staged-host or
CPU fallback path only after CUDA retry policy is exhausted.

The laptop RTX 4060 has 8 GiB physical VRAM. The usable admission budget is therefore dynamic. For
example, if other applications leave 6 GiB free, the default new-work budget is 4.8 GiB, less memory
already reserved by this CALO process.

## Select a non-default port

```text
CALO_PORT=7080 docker compose --profile cpu up cpu
```

Then open `http://localhost:7080/vnc.html`.

The default container host-memory limit is 24 GiB, suitable for the specified 32 GiB workstation.
Override it with `CALO_HOST_MEMORY_LIMIT` only after reserving enough memory for Windows, Docker/WSL,
and other applications. The live 80%-of-currently-available admission rule remains the stricter
application-level limit.

## Persistent data and switching profiles

Stop the active profile before starting the other profile:

```text
docker compose --profile cuda down
docker compose --profile cpu up cpu
```

Do not run CPU and CUDA services concurrently against the shared volume. Policies and experiment
records remain portable because the checkpoint/result formats are device-neutral; tensors are loaded
onto the selected runtime only when used.

Volumes created by an older root-running image may not be writable by UID 10001. Back up the volume
and correct its ownership explicitly before upgrading; do not make the application privileged as a
workaround. Newly created volumes inherit the image's `/data` ownership.

To inspect the volume without deleting it:

```text
docker volume inspect calo-rpd-studio-runtime
```

## Operational boundaries

- CPU remains responsible for the Qt event loop, orchestration, SQLite, files, logging, and host-side
  validation even in CUDA mode. VRAM replaces RAM for supported numerical tensors, not for desktop
  services or persistent files.
- CUDA mode is accelerator-first and fail-closed. CPU mode is an explicit deployment choice.
- Containerization makes the runtime reproducible; it does not establish scientific superiority or
  hardware performance. Those claims require the benchmark and qualification gates in the audit plan.
- Image SBOMs, vulnerability reports and successful CPU/CUDA build records are still required before
  this source tree qualifies as a reproducible release artifact.

## Release evidence

The locks are source-controlled inputs, not proof that an image built successfully. On a release host
with Buildx and Docker Compose installed, build each profile with BuildKit provenance and SBOM enabled,
record the resulting immutable image digest, and retain the attestations. First require a clean
tracked checkout and pass its full commit into the immutable in-image source declaration:

```text
$commit = git rev-parse HEAD
if (git status --porcelain --untracked-files=no) { throw "Tracked source is dirty" }
docker buildx build --platform linux/amd64 --build-arg RUNTIME_LOCK=requirements-lock-cpu-py311-linux.txt --build-arg SOURCE_COMMIT=$commit --build-arg SOURCE_TRACKED_CLEAN=true --provenance=mode=max --sbom=true --tag "calo-rpd-studio:cpu-$($commit.Substring(0,12))" --load .
docker buildx build --platform linux/amd64 --build-arg RUNTIME_LOCK=requirements-lock-cuda128-py311-linux.txt --build-arg SOURCE_COMMIT=$commit --build-arg SOURCE_TRACKED_CLEAN=true --provenance=mode=max --sbom=true --tag "calo-rpd-studio:cuda-$($commit.Substring(0,12))" --load .
```

The image contains `/opt/calo/.calo-source-identity.json`. Runtime evidence resolves a live Git
worktree first and only uses this declaration when Git is unavailable, so a clean image declaration
cannot bypass a dirty mounted checkout. Durable validators reject unavailable commits and dirty
tracked identities. The declaration is operator-supplied build metadata corroborated by retained
BuildKit provenance; it is not a signature.

`pip check` runs after the complete hash-locked graph is installed. The image then removes the
runtime-unneeded `pip`, `setuptools`, and `wheel` build tools, reducing package-management attack
surface and preventing vendored build-tool vulnerabilities from shipping in the runtime layer.

Run an image vulnerability scanner against both immutable digests and retain the scanner name,
database timestamp, policy and complete report. Docker Desktop, Compose and Buildx are now available
on the target workstation; successful source-bound builds and runtime reports remain required before
the images are qualified.

## Continuous validation

The repository workflow is a release harness, not evidence by itself. Its main lanes are:

- a hash-locked Linux/Python 3.11 source and scientific-test lane, including exact-pin/index/hash
  integrity checks, Ruff lint/format, generated-schema verification, scoped mypy and coverage;
- Python 3.11â€“3.13 Linux/Windows compatibility checks for historical configuration, database and
  policy loading. Populated version-0 databases are copied with SQLite's online-backup API before
  transactional migration; backup integrity, SHA-256 receipt, row preservation, idempotent reopen
  and fail-closed rejection of future schemas are regression tested;
- a Qt offscreen lane that exercises navigation and the scientist protocol, checks accessibility
  contracts, renders a real dashboard PNG and uploads it;
- a CPU image lane that builds with BuildKit SBOM/provenance, verifies UID/GID 10001, enforces a
  read-only root, performs a version-checked SQLite/config round-trip, creates a manifest from `/opt/calo`, emits a
  CycloneDX SBOM and fails on fixable high/critical vulnerabilities;
- a CUDA image build without a device on non-pull-request runs; and
- a manually requested physical-CUDA lane restricted to a trusted `calo-cuda` runner, covering
  visibility, FP64 parity, a bounded protection soak and graceful cancellation.

All external GitHub Actions references are full 40-character commit hashes rather than mutable
tags. This is especially important for the scanner action: Aqua documented a 2026 tag-rewriting
supply-chain incident and recommends SHA-pinned action references. See the official
[Trivy advisory](https://github.com/aquasecurity/trivy/security/advisories/GHSA-69fq-xp46-6x23),
[GitHub checkout guidance](https://github.com/actions/checkout), and
[Docker Buildx action documentation](https://github.com/docker/build-push-action).

Official references: [PyTorch 2.10 wheel matrix](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA GPU enumeration for containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
