"""Consistent scientist-facing errors with technical details retained in Activity logs."""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import QMessageBox, QWidget


_LOG = logging.getLogger("calo_rpd_studio.gui")
_DETAILS_DIRECTION = "Review Activity > Logs for technical details."


def show_error(
    parent: QWidget | None,
    title: str,
    message: str,
    error: BaseException | str | None = None,
    *,
    source: str = "application",
) -> None:
    """Log complete technical context while keeping the modal short and actionable."""

    if isinstance(error, BaseException):
        _LOG.error(
            "%s failed: %s: %s",
            source,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    elif error:
        _LOG.error("%s failed: %s", source, error)
    QMessageBox.critical(parent, str(title), f"{str(message).rstrip()}\n\n{_DETAILS_DIRECTION}")


def show_warning(
    parent: QWidget | None,
    title: str,
    message: str,
    error: BaseException | str | None = None,
    *,
    source: str = "application",
) -> None:
    """Warning counterpart for unavailable optional operations."""

    if isinstance(error, BaseException):
        _LOG.warning(
            "%s unavailable: %s: %s",
            source,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    elif error:
        _LOG.warning("%s unavailable: %s", source, error)
    QMessageBox.warning(parent, str(title), f"{str(message).rstrip()}\n\n{_DETAILS_DIRECTION}")


def log_technical_error(source: str, error: BaseException | str) -> None:
    """Retain a non-modal technical failure in the same Activity log stream."""

    if isinstance(error, BaseException):
        _LOG.error(
            "%s failed: %s: %s",
            source,
            type(error).__name__,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )
    else:
        _LOG.error("%s failed: %s", source, error)
