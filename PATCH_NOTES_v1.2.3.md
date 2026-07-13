# CALO-RPD Studio 1.2.3

Version 1.2.3 improves the first-launch prerequisite experience by making long downloads and installations observable rather than leaving the user with only a scrolling pip log.

## Prerequisite progress telemetry

The setup wizard now presents two complementary progress views:

1. **Overall installation stage** — a coarse, deterministic stage bar covering installer preparation, core prerequisites, accelerator detection, PyTorch backend setup, application installation, and final verification.
2. **Current artifact download** — exact byte-level progress for the file that pip is currently downloading.

For active downloads the wizard displays:

- current artifact name;
- downloaded bytes / total bytes;
- percentage complete;
- measured transfer speed;
- estimated remaining time;
- total elapsed installation time.

During dependency resolution, wheel installation, backend verification, or another operation without a truthful byte total, the current-activity bar switches to indeterminate mode and explains the active task.

## Implementation details

The bootstrap invokes pip installation commands with `--progress-bar=raw`. pip emits machine-readable byte counters in the form `Progress <current> of <total>`. CALO-RPD Studio parses those records in the bootstrap process, calculates a rolling transfer speed and ETA, and forwards structured progress events to the Tkinter setup wizard through its thread-safe event queue.

The repetitive raw progress records are not copied into the text log. Normal pip diagnostics remain visible.

## Scope of the overall percentage

The overall percentage intentionally represents **installation stages**, not a fabricated aggregate byte count. pip can discover additional dependency artifacts during resolution, so the final total number of bytes is not always known before downloads begin. The current-artifact byte counter is exact whenever pip provides a file size.
