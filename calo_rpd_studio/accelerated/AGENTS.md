# Accelerated execution instructions

- CUDA is VRAM-first and device-resident when the admitted working set fits.
- Calculate admission from currently free VRAM, not total capacity; distinguish allocated, reserved, and global-free memory.
- Retry by bounded microbatch reduction before governed staging, clean CPU restart, or fail-closed behavior.
- Record every selection, admission, transfer, fallback, retry, and numerical parity boundary.
