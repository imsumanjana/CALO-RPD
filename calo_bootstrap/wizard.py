"""Tkinter prerequisite setup wizard.

Tkinter is used deliberately because the wizard must be able to start before PyQt6 is installed.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

from .prerequisites import (
    EnvironmentReport,
    InstallProgress,
    install_or_repair,
    save_environment_state,
    scan_environment,
)


class PrerequisiteWizard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("CALO-RPD Studio — Prerequisite Setup")
        self.root.geometry("980x820")
        self.root.minsize(820, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.result = False
        self.report: EnvironmentReport | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.install_started_at: float | None = None
        self._download_bar_running = False

        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="CALO-RPD Studio Prerequisite Setup",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "This first-launch wizard checks the scientific Python environment, detects NVIDIA and Intel "
                "graphics hardware, provisions CUDA and Intel XPU compute runtimes when supported, and verifies "
                "real accelerator computations before the main application starts."
            ),
            wraplength=850,
        ).pack(anchor="w", pady=(4, 12))

        self.tree = ttk.Treeview(outer, columns=("status", "details"), show="headings", height=14)
        self.tree.heading("status", text="Status")
        self.tree.heading("details", text="Component / details")
        self.tree.column("status", width=110, anchor="center", stretch=False)
        self.tree.column("details", width=690, anchor="w")
        self.tree.pack(fill="x")

        self.summary = ttk.Label(outer, text="Scanning environment...", wraplength=850)
        self.summary.pack(anchor="w", pady=(10, 8))

        progress_box = ttk.LabelFrame(outer, text="Installation progress", padding=10)
        progress_box.pack(fill="x", pady=(0, 10))

        self.phase_label = ttk.Label(progress_box, text="Waiting for installation.")
        self.phase_label.pack(fill="x")
        self.overall_progress = ttk.Progressbar(
            progress_box, orient="horizontal", mode="determinate", maximum=100
        )
        self.overall_progress.pack(fill="x", pady=(5, 8))

        self.download_label = ttk.Label(progress_box, text="No active download.", wraplength=820)
        self.download_label.pack(fill="x")
        self.download_progress = ttk.Progressbar(
            progress_box, orient="horizontal", mode="determinate", maximum=100
        )
        self.download_progress.pack(fill="x", pady=(5, 5))

        metrics = ttk.Frame(progress_box)
        metrics.pack(fill="x")
        self.download_amount_label = ttk.Label(metrics, text="Downloaded: —")
        self.download_speed_label = ttk.Label(metrics, text="Speed: —")
        self.download_eta_label = ttk.Label(metrics, text="ETA: —")
        self.elapsed_label = ttk.Label(metrics, text="Elapsed: 00:00")
        self.download_amount_label.pack(side="left")
        self.download_speed_label.pack(side="left", padx=(18, 0))
        self.download_eta_label.pack(side="left", padx=(18, 0))
        self.elapsed_label.pack(side="right")

        self.log = tk.Text(outer, height=8, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(0, 10))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x")
        self.scan_button = ttk.Button(buttons, text="Scan System", command=self.scan)
        self.install_button = ttk.Button(
            buttons, text="Install / Repair Prerequisites", command=self.install
        )
        self.verify_button = ttk.Button(buttons, text="Verify Environment", command=self.scan)
        self.start_button = ttk.Button(buttons, text="Start CALO-RPD Studio", command=self.start)
        self.close_button = ttk.Button(buttons, text="Exit", command=self._close)
        self.scan_button.pack(side="left", padx=(0, 6))
        self.install_button.pack(side="left", padx=6)
        self.verify_button.pack(side="left", padx=6)
        self.close_button.pack(side="right")
        self.start_button.pack(side="right", padx=(0, 6))
        self.start_button.state(["disabled"])

        self.root.after(100, self.scan)
        self.root.after(100, self._poll_events)
        self.root.after(500, self._update_elapsed)

    def _set_busy(self, busy: bool) -> None:
        controls = (self.scan_button, self.install_button, self.verify_button, self.close_button)
        for widget in controls:
            if busy:
                widget.state(["disabled"])
            else:
                widget.state(["!disabled"])
        if busy or not (self.report and self.report.mandatory_ready):
            self.start_button.state(["disabled"])
        else:
            self.start_button.state(["!disabled"])

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    @staticmethod
    def _mark(ok: bool, warning: bool = False) -> str:
        if ok:
            return "READY"
        return "WARNING" if warning else "MISSING"

    def _render(self, report: EnvironmentReport) -> None:
        self.report = report
        self.tree.delete(*self.tree.get_children())
        self.tree.insert(
            "",
            "end",
            values=(
                self._mark(report.python_ok),
                f"Python {report.python_version} — {report.interpreter}",
            ),
        )
        self.tree.insert(
            "",
            "end",
            values=(
                "INFO",
                f"Virtual environment: {'Yes' if report.virtual_environment else 'No'}",
            ),
        )
        for name, version in report.core_packages.items():
            self.tree.insert(
                "",
                "end",
                values=(self._mark(bool(version)), f"{name}: {version or 'not installed'}"),
            )

        nvidia_detail = (
            f"NVIDIA: {report.nvidia.name} · driver {report.nvidia.driver_version} · "
            f"maximum CUDA {report.nvidia.max_cuda_version or 'unknown'}"
            if report.nvidia.detected
            else f"NVIDIA: not detected ({report.nvidia.error or 'not available'})"
        )
        self.tree.insert(
            "",
            "end",
            values=("READY" if report.nvidia.detected else "INFO", nvidia_detail),
        )

        intel_detail = (
            f"Intel graphics: {report.intel.name}"
            if report.intel.detected
            else f"Intel graphics: not detected ({report.intel.error or 'not available'})"
        )
        self.tree.insert(
            "",
            "end",
            values=("READY" if report.intel.detected else "INFO", intel_detail),
        )

        primary_cuda_ready = bool(report.torch.cuda_available and report.torch.gpu_test_passed)
        primary_xpu_ready = bool(report.torch.xpu_available and report.torch.xpu_test_passed)
        torch_status = bool(report.torch.installed)
        torch_detail = (
            f"Primary PyTorch {report.torch.version or 'not installed'} · "
            f"CUDA {'READY' if primary_cuda_ready else 'inactive'}"
            f"{f' ({report.torch.device_name})' if report.torch.device_name else ''} · "
            f"XPU {'READY' if primary_xpu_ready else 'inactive'}"
            f"{f' ({report.torch.xpu_device_name})' if report.torch.xpu_device_name else ''}"
        )
        self.tree.insert(
            "",
            "end",
            values=(self._mark(torch_status, warning=not torch_status), torch_detail),
        )

        if report.intel.detected:
            sidecar_ready = bool(
                report.xpu_sidecar.xpu_available and report.xpu_sidecar.gpu_test_passed
            )
            sidecar_detail = (
                f"Secondary Intel XPU runtime: "
                f"{'READY' if sidecar_ready else 'not verified'} · "
                f"PyTorch {report.xpu_sidecar.torch_version or '—'} · "
                f"device {report.xpu_sidecar.device_name or '—'}"
            )
            if report.xpu_sidecar.error and not sidecar_ready:
                sidecar_detail += f" · {report.xpu_sidecar.error}"
            self.tree.insert(
                "",
                "end",
                values=(self._mark(sidecar_ready, warning=True), sidecar_detail),
            )

        self.summary.configure(
            text=(
                f"{report.message} Recommended backend: {report.recommended_backend}. "
                "Scheduler priority: NVIDIA CUDA → Intel XPU → CPU. "
                "Backend device IDs may differ from Windows Task Manager GPU numbers."
            )
        )
        if report.mandatory_ready:
            self.start_button.state(["!disabled"])
        else:
            self.start_button.state(["disabled"])

    @staticmethod
    def _format_bytes(value: int | float) -> str:
        amount = float(max(0.0, value))
        units = ("B", "KB", "MB", "GB", "TB")
        for unit in units:
            if amount < 1024.0 or unit == units[-1]:
                return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
            amount /= 1024.0
        return f"{amount:.1f} TB"

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or seconds < 0 or seconds == float("inf"):
            return "—"
        total = int(round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def _set_download_indeterminate(self, active: bool) -> None:
        if active:
            if self._download_bar_running:
                return
            self.download_progress.stop()
            self.download_progress.configure(mode="indeterminate")
            self.download_progress.start(12)
            self._download_bar_running = True
        else:
            if self._download_bar_running:
                self.download_progress.stop()
                self._download_bar_running = False
            self.download_progress.configure(mode="determinate", maximum=100)

    def _render_progress(self, progress: InstallProgress) -> None:
        self.overall_progress["value"] = max(0.0, min(100.0, progress.overall_percent))
        phase_index = max(1, min(progress.phase_count, progress.phase_index or 1))
        self.phase_label.configure(
            text=(
                f"Step {phase_index}/{progress.phase_count} — {progress.phase} · "
                f"overall {progress.overall_percent:.0f}%"
            )
        )

        if progress.total_bytes > 0:
            self._set_download_indeterminate(False)
            self.download_progress["value"] = progress.download_percent
            item = progress.item or "Current package"
            self.download_label.configure(text=f"{item} — {progress.download_percent:.1f}%")
            self.download_amount_label.configure(
                text=(
                    f"Downloaded: {self._format_bytes(progress.current_bytes)} / "
                    f"{self._format_bytes(progress.total_bytes)}"
                )
            )
            speed = progress.speed_bytes_per_second
            self.download_speed_label.configure(
                text=f"Speed: {self._format_bytes(speed)}/s" if speed > 0 else "Speed: measuring..."
            )
            self.download_eta_label.configure(
                text=f"ETA: {self._format_duration(progress.eta_seconds)}"
            )
        elif progress.indeterminate:
            self._set_download_indeterminate(True)
            self.download_label.configure(text=progress.message or progress.item or "Working...")
            self.download_amount_label.configure(text="Downloaded: resolving / installing")
            self.download_speed_label.configure(text="Speed: —")
            self.download_eta_label.configure(text="ETA: —")
        else:
            self._set_download_indeterminate(False)
            self.download_progress["value"] = 100 if progress.overall_percent >= 100 else 0
            self.download_label.configure(text=progress.message or "Stage complete.")
            self.download_amount_label.configure(text="Downloaded: —")
            self.download_speed_label.configure(text="Speed: —")
            self.download_eta_label.configure(text="ETA: —")

    def _update_elapsed(self) -> None:
        if self.install_started_at is not None:
            self.elapsed_label.configure(
                text=f"Elapsed: {self._format_duration(time.monotonic() - self.install_started_at)}"
            )
        self.root.after(500, self._update_elapsed)

    def scan(self) -> None:
        self._set_busy(True)
        self.summary.configure(text="Scanning environment...")

        def work() -> None:
            try:
                self.events.put(("report", scan_environment()))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("busy", False))

        threading.Thread(target=work, daemon=True).start()

    def install(self) -> None:
        self._set_busy(True)
        self.install_started_at = time.monotonic()
        self.overall_progress["value"] = 0
        self._log("Starting prerequisite installation/repair...")

        def work() -> None:
            try:
                report = install_or_repair(
                    lambda line: self.events.put(("log", line)),
                    prefer_gpu=True,
                    progress_callback=lambda progress: self.events.put(("progress", progress)),
                )
                self.events.put(("report", report))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("busy", False))

        threading.Thread(target=work, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "report":
                    self._render(payload)  # type: ignore[arg-type]
                elif kind == "progress":
                    self._render_progress(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self._log(str(payload))
                    messagebox.showerror("Prerequisite setup", str(payload))
                elif kind == "busy":
                    self._set_busy(bool(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def start(self) -> None:
        if not self.report or not self.report.mandatory_ready:
            messagebox.showwarning("Prerequisite setup", "Verify the environment before starting.")
            return
        accepted_cpu_fallback = False
        accelerator_detected = bool(self.report.nvidia.detected or self.report.intel.detected)
        if accelerator_detected and not self.report.gpu_ready:
            accepted_cpu_fallback = messagebox.askyesno(
                "Start with CPU fallback?",
                "GPU hardware was detected, but no CUDA/XPU runtime passed a real computation test. "
                "Start CALO-RPD Studio with CPU fallback and remember this choice for this environment?",
            )
            if not accepted_cpu_fallback:
                return
        save_environment_state(self.report, accepted_cpu_fallback=accepted_cpu_fallback)
        self.result = True
        self.root.destroy()

    def _close(self) -> None:
        self.result = False
        self.root.destroy()

    def run(self) -> bool:
        self.root.mainloop()
        return self.result
