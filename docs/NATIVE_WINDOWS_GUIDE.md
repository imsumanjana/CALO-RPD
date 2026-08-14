# Native Windows and Docker operation

CALO-RPD Studio is a desktop application first. Docker remains an optional reproducible CPU/CUDA
runtime, but the ordinary GUI does not require Docker Desktop.

## First native setup

Use a supported Python version and create the project environment once. Dependency setup is
deliberately separate from normal launch because a routine start must not repeatedly download or
upgrade large packages.

```powershell
cd "C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio"
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe .\bootstrap.py --setup
```

The bootstrap path scans prerequisites and may offer installation or repair. Review its proposed
actions. It is not the routine launcher.

## Normal native launch

For a repository-development checkout:

```powershell
cd "C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio"
& .\Launch-CALO-RPD.ps1
```

The launcher uses only `.venv\Scripts\python.exe` and the shared application entry point. It does
not install, repair, train, qualify, register, activate, or delete anything. If the environment is
missing, it stops with first-setup instructions.

For an installed wheel in an activated environment, either command launches the same application:

```powershell
calo-rpd-native
python -m calo_rpd_studio.app.application
```

`calo-rpd-studio` remains the setup-aware packaged command. `calo-rpd-native` is the direct routine
desktop command and never invokes the prerequisite installer.

## CUDA and CPU-only operation

CUDA-preferred is a configured intent, not proof of GPU execution. Check the Compute ribbon's
Device status and the activity/status areas for the actual runtime assignment.

To inspect the installed PyTorch runtime before launch:

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
nvidia-smi
```

In the application, use **Compute > Compute mode** to select **CUDA-preferred** or **CPU-only**, then
validate and apply. CPU-only forces the requested device to CPU. Intel XPU is not executable.
Safe-80 resource values are admission ceilings based on currently free VRAM or available RAM; they
must not be interpreted as current consumption.

For fresh policy training, open **Policies > Train policy** once to display the scientific inputs.
Choose **New training**, the eligible cases and other visible parameters, and a new training
directory. The application constructs the full internal plan from those inputs plus safe fixed
defaults; there is no external settings-template or model-library-path control. Click **Check
readiness** in the input pane. The plan retains the checkout source identity even when the native
command was launched from a different working directory. Fresh plan-generation errors are labeled
separately from retained saved-plan load errors. If readiness passes, the action changes to **Start
training**; click it to confirm the explicit start. The output is an unqualified, inactive candidate
and is never activated automatically.

## Activity, logs, data, and shutdown

- Jobs, searchable display logs, warnings, device state, and provenance appear in the bottom
  Activity pane. **Clear display** only clears the in-memory view; it does not delete durable evidence.
- Error dialogs intentionally use short scientist-facing explanations. Use **Activity > Logs** for
  the exception type, detailed message, and traceback needed for diagnosis.
- Application preferences use the Windows Qt settings store for `CALO-RPD / CALO-RPD Studio`.
- The repository-development result database defaults to `calo_rpd_results.sqlite` in the working
  directory. Experiment output defaults to the configured `results_data` location.
- Use the window close action for normal shutdown. If an experiment is active, the application asks
  for a safe pause. If independent training is active, the GUI refuses silent termination and keeps
  the process visible until it completes or its authenticated recovery procedure is used.

## Docker launch remains available

After building the chosen profile, launch the GUI service using the existing Compose contract. Use
the exact commands documented by the current Compose file and retained release-development
instructions; Docker images and native execution use the same source and policy boundaries.

Docker layer caching normally reuses already downloaded dependency layers when lock files and
earlier Dockerfile instructions are unchanged. `--no-cache`, changed dependency inputs, a pruned
builder cache, or a different builder can require downloads again.

## Troubleshooting

- **Virtual environment missing:** run the one-time setup commands; do not point the launcher at an
  unrelated global interpreter.
- **CUDA unavailable:** verify `nvidia-smi`, the CUDA-capable PyTorch build, and
  `torch.cuda.is_available()`. Select CPU-only when CPU execution is intentional; do not label it
  CUDA fallback evidence.
- **Ribbon or docks are misplaced:** use **View > Reset layout**. Phase 6 layout state is versioned;
  incompatible or corrupt state falls back to a safe default.
- **Policy not ready:** an empty policy store is supported. Rule-only CALO and non-policy workflows
  remain available; file presence alone never means a policy is qualified or active.
- **New policy training readiness fails:** review the training-center output. Select **CALO** when
  using the built-in policy-free optimizer; it requires no training. Select **TSH-CALO** to prepare
  a new policy through the independent CLI. That CLI authenticates the generated plan, current
  application source identity, and clean tracked source before work starts. Saved campaigns load
  their own retained plan internally only for exact resume or compatible finite extension. Do not
  weaken readiness checks to make the GUI appear ready.
