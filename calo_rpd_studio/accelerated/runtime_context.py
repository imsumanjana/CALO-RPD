"""Process-local v3.1 accelerator runtime context.

A persistent accelerator worker installs one cross-run broker in this module.  Every accelerated
problem created in that worker automatically attaches to the broker, while ordinary single-run and
CPU-reference execution remain unaffected.
"""

from __future__ import annotations

import threading


_lock = threading.Lock()
_broker = None


def set_cross_run_broker(broker) -> None:
    global _broker
    with _lock:
        _broker = broker


def get_cross_run_broker():
    with _lock:
        return _broker


def clear_cross_run_broker() -> None:
    set_cross_run_broker(None)
