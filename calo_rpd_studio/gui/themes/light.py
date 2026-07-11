"""Light appearance stylesheet."""

LIGHT_STYLESHEET = r"""
QWidget {
    font-family: "Segoe UI";
    font-size: 10pt;
    color: #0f172a;
}
QMainWindow, #WorkspaceStack, #WorkspacePage, #ScrollableWorkspace,
#ScrollableViewport, #ScrollableContent {
    background: #f4f7fb;
}
QDialog {
    background: #f4f7fb;
}
QSplitter::handle {
    background: #dfe6ef;
    width: 1px;
}
QStatusBar {
    background: #ffffff;
    color: #64748b;
    border-top: 1px solid #e5eaf1;
}

/* Sidebar */
#Sidebar {
    background: #ffffff;
    border-right: 1px solid #e3e9f1;
}
#BrandMark {
    background: #2563eb;
    color: #ffffff;
    border-radius: 11px;
    font-size: 15pt;
    font-weight: 800;
}
#BrandTitle {
    color: #0f172a;
    font-size: 13.5pt;
    font-weight: 750;
    background: transparent;
}
#BrandSubtitle {
    color: #64748b;
    font-size: 9pt;
    background: transparent;
}
#NavSectionLabel {
    color: #94a3b8;
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4px 7px 5px 7px;
    background: transparent;
}
#NavButton {
    text-align: left;
    padding: 8px 11px;
    border: 1px solid transparent;
    border-radius: 9px;
    color: #526174;
    background: transparent;
    font-weight: 500;
}
#NavButton:hover {
    color: #0f172a;
    background: #f1f5fb;
}
#NavButton:checked {
    color: #1d4ed8;
    background: #eaf1ff;
    border-color: #d6e3ff;
    font-weight: 700;
}
#SidebarFooter {
    background: #f7f9fc;
    border: 1px solid #e6ebf2;
    border-radius: 10px;
}
#SidebarFooterTitle {
    color: #334155;
    font-size: 9pt;
    font-weight: 650;
    background: transparent;
}
#SidebarFooterText {
    color: #94a3b8;
    font-size: 8pt;
    background: transparent;
}

/* Page headings */
#PageHeader {
    background: transparent;
}
#PageTitle {
    color: #0f172a;
    font-size: 23pt;
    font-weight: 750;
    background: transparent;
}
#PageSubtitle {
    color: #64748b;
    font-size: 9.7pt;
    background: transparent;
}

/* Modern cards */
#SectionCard, #MetricCard {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
#MetricCard:hover {
    border-color: #c8d7ef;
}
#CardTitle {
    color: #0f172a;
    font-size: 11.5pt;
    font-weight: 700;
    background: transparent;
}
#CardSubtitle {
    color: #64748b;
    background: transparent;
}
#MetricLabel {
    color: #64748b;
    font-size: 8.8pt;
    font-weight: 600;
    background: transparent;
}
#MetricValue {
    color: #0f172a;
    font-size: 17pt;
    font-weight: 750;
    background: transparent;
}
#MetricDetail {
    color: #94a3b8;
    font-size: 8.5pt;
    background: transparent;
}

/* Legacy group boxes retained for long technical forms */
QGroupBox {
    color: #1e293b;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 11px;
    margin-top: 11px;
    padding: 18px 14px 14px 14px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 13px;
    top: 1px;
    padding: 0 7px;
    color: #334155;
    background: #ffffff;
}

#ContextValue {
    color: #1e293b;
    font-weight: 650;
    background: transparent;
}
#InfoText {
    color: #475569;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    padding: 9px 11px;
}
#ResultBanner {
    color: #1e3a8a;
    background: #eff6ff;
    border: 1px solid #cfe0ff;
    border-radius: 9px;
    padding: 9px 11px;
}
#ToolbarContext {
    color: #64748b;
    background: transparent;
    padding: 0 8px;
}

/* Inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QTextEdit, QPlainTextEdit, QListView, QTreeView {
    color: #0f172a;
    background: #ffffff;
    border: 1px solid #cfd8e6;
    border-radius: 7px;
    padding: 6px 8px;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #4f83ef;
}
QComboBox QAbstractItemView {
    color: #0f172a;
    background: #ffffff;
    border: 1px solid #d6deea;
    selection-background-color: #eaf1ff;
    selection-color: #1d4ed8;
    outline: 0;
}
QCheckBox, QRadioButton, QLabel {
    background: transparent;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
}

/* Buttons */
QPushButton {
    color: #334155;
    background: #ffffff;
    border: 1px solid #d4dce8;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 20px;
    font-weight: 600;
}
QPushButton:hover {
    color: #0f172a;
    background: #f6f8fb;
    border-color: #bcc8d8;
}
QPushButton:pressed {
    background: #eef2f7;
}
QPushButton:disabled {
    color: #a6b0bf;
    background: #f5f7fa;
    border-color: #e6ebf1;
}
QPushButton#PrimaryButton {
    color: #ffffff;
    background: #2563eb;
    border-color: #2563eb;
    font-weight: 700;
}
QPushButton#PrimaryButton:hover {
    background: #1d4ed8;
    border-color: #1d4ed8;
}

/* Tables */
QTableView, QTableWidget {
    color: #0f172a;
    background: #ffffff;
    alternate-background-color: #f8fafc;
    border: 1px solid #e1e7ef;
    border-radius: 9px;
    gridline-color: #edf1f6;
    selection-background-color: #e7efff;
    selection-color: #153b91;
}
QHeaderView::section {
    color: #475569;
    background: #f5f7fa;
    padding: 8px 9px;
    border: none;
    border-bottom: 1px solid #e1e7ef;
    border-right: 1px solid #edf1f5;
    font-weight: 700;
}
QTableCornerButton::section {
    background: #f5f7fa;
    border: none;
}

/* Tabs */
QTabWidget::pane {
    background: #ffffff;
    border: 1px solid #e1e7ef;
    border-radius: 9px;
    top: -1px;
}
QTabBar::tab {
    color: #64748b;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 13px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #334155;
}
QTabBar::tab:selected {
    color: #1d4ed8;
    border-bottom-color: #2563eb;
}

/* Progress */
QProgressBar {
    color: #475569;
    background: #edf2f7;
    border: none;
    border-radius: 6px;
    text-align: center;
    min-height: 12px;
}
QProgressBar::chunk {
    background: #2563eb;
    border-radius: 6px;
}

/* Top application toolbar */
QToolBar#TopToolbar {
    color: #334155;
    background: #ffffff;
    border: none;
    border-bottom: 1px solid #e3e9f1;
    spacing: 5px;
    padding: 5px 10px;
}
QToolBar#TopToolbar QToolButton {
    color: #475569;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 9px;
    font-weight: 600;
}
QToolBar#TopToolbar QToolButton:hover {
    color: #0f172a;
    background: #f2f5f9;
    border-color: #e3e9f1;
}
QMenu {
    color: #0f172a;
    background: #ffffff;
    border: 1px solid #dce3ed;
    padding: 5px;
}
QMenu::item {
    padding: 6px 22px 6px 10px;
    border-radius: 5px;
}
QMenu::item:selected {
    color: #1d4ed8;
    background: #eaf1ff;
}
QToolTip {
    color: #0f172a;
    background: #ffffff;
    border: 1px solid #cfd8e6;
    padding: 4px;
}

/* Scrollbars remain visually quiet and only appear where scrolling is required */
QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #c2ccda;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #98a7ba;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 9px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: #c2ccda;
    min-width: 30px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""

# Guided workflow and persistent task-status additions.
LIGHT_STYLESHEET += r"""
#WorkspaceContainer {
    background: #f4f7fb;
}
#WorkflowGuide {
    background: #ffffff;
    border-bottom: 1px solid #e3e9f1;
}
#WorkflowStep {
    color: #1d4ed8;
    font-size: 9pt;
    font-weight: 750;
}
#WorkflowInstruction {
    color: #475569;
    font-size: 9.4pt;
}
#WorkflowNextButton {
    color: #1d4ed8;
    background: #eef4ff;
    border: 1px solid #cfe0ff;
    font-weight: 700;
}
#WorkflowNextButton:hover {
    background: #e2ecff;
    border-color: #b9d0ff;
}
#NavButton[workflowState="locked"] {
    color: #aeb8c7;
    background: transparent;
}
#NavButton[workflowState="completed"] {
    color: #087f5b;
}
#NavButton[workflowState="recommended"] {
    color: #1d4ed8;
    background: #f1f6ff;
    border-color: #dce8ff;
    font-weight: 700;
}
#NavButton[workflowState="optional"] {
    color: #7c5c16;
}
#GlobalTaskState {
    min-width: 66px;
    font-weight: 800;
    color: #334155;
}
#GlobalTaskState[taskState="busy"] { color: #1d4ed8; }
#GlobalTaskState[taskState="completed"] { color: #087f5b; }
#GlobalTaskState[taskState="failed"] { color: #b42318; }
#GlobalTaskDetail {
    color: #64748b;
}
#GlobalTaskElapsed {
    color: #64748b;
    font-variant-numeric: tabular-nums;
}
#GlobalTaskProgress {
    min-height: 10px;
    max-height: 16px;
}
#StatusCancelButton {
    padding: 3px 9px;
    min-height: 16px;
    color: #b42318;
    border-color: #f0b8b3;
    background: #fff5f4;
}
"""
LIGHT_STYLESHEET += r"""
#GlobalTaskState[taskState="cancelled"] { color: #7c5c16; }
"""
