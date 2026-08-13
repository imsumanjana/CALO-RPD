"""Theme-aware vector chevrons for Qt spin-box step controls."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPainter, QPalette, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QProxyStyle,
    QStyle,
    QStyleOptionComplex,
    QStyleOptionSpinBox,
    QWidget,
)


class ModernSpinBoxStyle(QProxyStyle):
    """Draw crisp spin arrows instead of low-contrast platform glyphs."""

    _BUTTONS = {
        QStyle.SubControl.SC_SpinBoxUp: (
            -1.0,
            QAbstractSpinBox.StepEnabledFlag.StepUpEnabled,
        ),
        QStyle.SubControl.SC_SpinBoxDown: (
            1.0,
            QAbstractSpinBox.StepEnabledFlag.StepDownEnabled,
        ),
    }

    def drawComplexControl(
        self,
        control: QStyle.ComplexControl,
        option: QStyleOptionComplex,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        # Let Fusion plus the active stylesheet paint the field, dividers, and
        # button states first. The arrowheads are then guaranteed to be the
        # final layer instead of depending on an optional native primitive.
        super().drawComplexControl(control, option, painter, widget)
        if control != QStyle.ComplexControl.CC_SpinBox or not isinstance(
            option, QStyleOptionSpinBox
        ):
            return

        for subcontrol, (direction, step_flag) in self._BUTTONS.items():
            if not option.subControls & subcontrol:
                continue
            button_rect = self.subControlRect(control, option, subcontrol, widget)
            if button_rect.isValid() and not button_rect.isEmpty():
                self._draw_arrowhead(
                    painter,
                    option,
                    button_rect,
                    direction=direction,
                    enabled=bool(option.stepEnabled & step_flag),
                    interactive=bool(
                        option.activeSubControls & subcontrol
                        and option.state
                        & (QStyle.StateFlag.State_MouseOver | QStyle.StateFlag.State_Sunken)
                    ),
                )

    @staticmethod
    def _draw_arrowhead(
        painter: QPainter,
        option: QStyleOptionSpinBox,
        button_rect,
        *,
        direction: float,
        enabled: bool,
        interactive: bool,
    ) -> None:
        enabled = enabled and bool(option.state & QStyle.StateFlag.State_Enabled)
        if not enabled:
            color = option.palette.color(
                QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText
            )
        elif interactive:
            color = option.palette.color(QPalette.ColorRole.HighlightedText)
        else:
            color = option.palette.color(QPalette.ColorRole.Highlight)

        center = button_rect.center()
        half_width = max(2.5, min(3.5, button_rect.width() * 0.18))
        half_height = max(1.5, min(2.0, button_rect.height() * 0.16))
        points = QPolygonF(
            (
                QPointF(center.x() - half_width, center.y() - direction * half_height),
                QPointF(center.x(), center.y() + direction * half_height),
                QPointF(center.x() + half_width, center.y() - direction * half_height),
            )
        )

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(
            QPen(
                color,
                1.7,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawPolyline(points)
        painter.restore()
