# Native Windows and Docker operation

CALO-RPD Studio is a desktop application first. Docker is an optional reproducible CPU/CUDA runtime; ordinary GUI use does not require Docker Desktop.

## First native setup

Use a supported Python version and create the project environment once. Dependency setup is deliberately separate from routine launch so normal startup does not repeatedly install or upgrade packages.

```powershell
cd "C:\path\to\calo-rpd-studio"
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe .\bootstrap.py --setup
```

The bootstrap scans prerequisites and may offer installation or repair. Review its proposed actions. It is not the routine launcher.

## Normal native launch

For a repository checkout:

```powershell
cd "C:\path\to\calo-rpd-studio"
& .\Launch-CALO-RPD.ps1
```

The launcher uses only `.venv\Scripts\python.exe` and the shared application entry point. It does not install, repair, train, qualify, register, activate, or delete anything. If the environment is missing, it stops with first-setup instructions.

For an installed wheel in an activated environment, either command launches the same application:

```powershell
calo-rpd-native
python -m calo_rpd_studio.app.application
```

`calo-rpd-studio` remains the setup-aware packaged command. `calo-rpd-native` is the direct routine desktop command and never invokes the prerequisite installer.

## CUDA and CPU-only operation

CUDA-preferred is a configured intent, not proof of GPU execution. Check the Compute ribbon and activity/status surfaces for the actual runtime assignment.

To inspect the installed PyTorch runtime before launch:

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi
```

Use **Compute > Compute mode** to select **CUDA-preferred** or **CPU-only**, then validate and apply. CPU-only forces the requested device to CPU. Intel XPU is not executable in v12. Safe-80 values are admission ceilings based on currently free VRAM or available RAM; they are not measurements of current utilization.

## Policy training boundary

Policy-free CALO requires no trained policy. TSH-CALO training is a separate workflow. Fresh training creates an unqualified, inactive candidate; qualification and activation are separate explicit gates. Saved campaigns retain their own authenticated plan for exact resume or compatible finite extension. Do not weaken readiness checks to make a workflow appear ready.

## Activity, data, and shutdown

- Jobs, searchable display logs, warnings, device state, and provenance appear in the Activity surface. Clearing the display does not delete durable evidence.
- Scientist-facing dialogs intentionally stay concise; use Activity/Logs for detailed exception information and tracebacks.
- Application preferences use the Windows Qt settings store for `CALO-RPD / CALO-RPD Studio`.
- Repository-development result data default to local configured paths and are not source-controlled release evidence.
- Use the window close action for normal shutdown. Active work must follow its safe pause/recovery contract rather than being silently terminated.

## Docker

Docker uses the same source and policy boundaries as native execution. The authoritative commands and runtime constraints are documented in [`CONTAINER_RUNBOOK.md`](CONTAINER_RUNBOOK.md). Docker layer caching normally reuses dependency layers when lock files and earlier Dockerfile instructions are unchanged; `--no-cache`, changed dependency inputs, a pruned cache, or a different builder can require downloads again.

## Troubleshooting

- **Virtual environment missing:** run the one-time setup commands; do not point the launcher at an unrelated global interpreter.
- **CUDA unavailable:** verify `nvidia-smi`, the CUDA-capable PyTorch build, and `torch.cuda.is_available()`. Select CPU-only when CPU execution is intentional; do not label CPU execution as CUDA evidence.
- **Ribbon or docks misplaced:** use **View > Reset layout**. Incompatible or corrupt saved layout state should fall back to a safe default.
- **Policy not ready:** an empty policy store is supported. Rule-only CALO and non-policy workflows remain available; file presence alone never means a policy is qualified or active.
- **TSH-CALO readiness fails:** review the training output and retained plan/integrity diagnostics. Do not bypass source, plan, compatibility, or integrity gates.
