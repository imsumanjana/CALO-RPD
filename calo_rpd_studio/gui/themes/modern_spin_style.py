"""Theme-aware vector details for compact Qt input controls."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPalette, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QStyleOptionComplex,
    QStyleOptionSpinBox,
    QWidget,
)


class ModernSpinBoxStyle(QProxyStyle):
    """Draw crisp spin arrows and consistently bounded checkbox indicators."""

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

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option: QStyleOption,
        painter: QPainter,
        widget: QWidget | None = None,
    ) -> None:
        if element == QStyle.PrimitiveElement.PE_IndicatorCheckBox:
            self._draw_checkbox_indicator(painter, option)
            return
        super().drawPrimitive(element, option, painter, widget)

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
    def _draw_checkbox_indicator(painter: QPainter, option: QStyleOption) -> None:
        """Paint a visible box and state mark with colors from the active palette."""
        state = option.state
        enabled = bool(state & QStyle.StateFlag.State_Enabled)
        checked = bool(state & QStyle.StateFlag.State_On)
        partial = bool(state & QStyle.StateFlag.State_NoChange)
        interactive = bool(
            state
            & (
                QStyle.StateFlag.State_MouseOver
                | QStyle.StateFlag.State_HasFocus
                | QStyle.StateFlag.State_Sunken
            )
        )
        group = QPalette.ColorGroup.Active if enabled else QPalette.ColorGroup.Disabled
        palette = option.palette
        box_color = palette.color(group, QPalette.ColorRole.Base)
        border_color = palette.color(group, QPalette.ColorRole.ButtonText)
        mark_color = palette.color(group, QPalette.ColorRole.HighlightedText)

        if checked or partial:
            box_color = palette.color(group, QPalette.ColorRole.Highlight)
        elif state & QStyle.StateFlag.State_Sunken:
            box_color = palette.color(group, QPalette.ColorRole.Button)

        if enabled and interactive:
            border_color = palette.color(group, QPalette.ColorRole.Highlight)
        elif enabled:
            border_color = QColor(border_color)
            # Keep the idle outline fully opaque: partial alpha made the
            # outer antialiased pixels disappear against light surfaces.
            border_color.setAlpha(255)
        else:
            border_color = QColor(border_color)
            border_color.setAlpha(150)

        box = QRectF(option.rect).adjusted(0.75, 0.75, -0.75, -0.75)
        border_width = 2.0 if state & QStyle.StateFlag.State_HasFocus else 1.5

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(box_color)
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(box, 3.0, 3.0)

        if checked:
            points = QPolygonF(
                (
                    QPointF(
                        box.left() + box.width() * 0.22,
                        box.top() + box.height() * 0.52,
                    ),
                    QPointF(
                        box.left() + box.width() * 0.43,
                        box.top() + box.height() * 0.73,
                    ),
                    QPointF(
                        box.left() + box.width() * 0.80,
                        box.top() + box.height() * 0.30,
                    ),
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(
                QPen(
                    mark_color,
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawPolyline(points)
        elif partial:
            y = box.center().y()
            painter.setPen(
                QPen(
                    mark_color,
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(
                QPointF(box.left() + box.width() * 0.25, y),
                QPointF(box.right() - box.width() * 0.25, y),
            )
        painter.restore()

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
