"""Dark appearance stylesheet."""

DARK_STYLESHEET = r"""
QWidget {
    font-family: "Segoe UI";
    font-size: 10pt;
    color: #e5eaf2;
}
QMainWindow, #WorkspaceStack, #WorkspacePage, #ScrollableWorkspace,
#ScrollableViewport, #ScrollableContent {
    background: #0f1520;
}
QDialog {
    background: #0f1520;
}
QSplitter::handle {
    background: #273244;
    width: 1px;
}
QStatusBar {
    background: #111a28;
    color: #8f9caf;
    border-top: 1px solid #253044;
}

#Sidebar {
    background: #111a28;
    border-right: 1px solid #263247;
}
#BrandMark {
    background: #4f7cff;
    color: #ffffff;
    border-radius: 11px;
    font-size: 15pt;
    font-weight: 800;
}
#BrandTitle {
    color: #f8fafc;
    font-size: 13.5pt;
    font-weight: 750;
    background: transparent;
}
#BrandSubtitle {
    color: #8290a6;
    font-size: 9pt;
    background: transparent;
}
#NavSectionLabel {
    color: #66758c;
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
    color: #9aa8bc;
    background: transparent;
    font-weight: 500;
}
#NavButton:hover {
    color: #f8fafc;
    background: #192537;
}
#NavButton:checked {
    color: #b9ccff;
    background: #1c315b;
    border-color: #29477d;
    font-weight: 700;
}
#SidebarFooter {
    background: #151f2f;
    border: 1px solid #263247;
    border-radius: 10px;
}
#SidebarFooterTitle {
    color: #d4dbe7;
    font-size: 9pt;
    font-weight: 650;
    background: transparent;
}
#SidebarFooterText {
    color: #718099;
    font-size: 8pt;
    background: transparent;
}

#PageHeader {
    background: transparent;
}
#PageTitle {
    color: #f8fafc;
    font-size: 23pt;
    font-weight: 750;
    background: transparent;
}
#PageSubtitle {
    color: #8d9aaf;
    font-size: 9.7pt;
    background: transparent;
}

#SectionCard, #MetricCard {
    background: #151e2c;
    border: 1px solid #2a3548;
    border-radius: 12px;
}
#MetricCard:hover {
    border-color: #3a4b65;
}
#CardTitle {
    color: #f4f7fb;
    font-size: 11.5pt;
    font-weight: 700;
    background: transparent;
}
#CardSubtitle {
    color: #8d9aaf;
    background: transparent;
}
#MetricLabel {
    color: #8d9aaf;
    font-size: 8.8pt;
    font-weight: 600;
    background: transparent;
}
#MetricValue {
    color: #f8fafc;
    font-size: 17pt;
    font-weight: 750;
    background: transparent;
}
#MetricDetail {
    color: #66758c;
    font-size: 8.5pt;
    background: transparent;
}

QGroupBox {
    color: #e3e9f2;
    background: #151e2c;
    border: 1px solid #2a3548;
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
    color: #d8e0eb;
    background: #151e2c;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QTextEdit, QPlainTextEdit, QListView, QTreeView {
    color: #e8edf5;
    background: #101824;
    border: 1px solid #344258;
    border-radius: 7px;
    padding: 6px 8px;
    selection-background-color: #4f7cff;
    selection-color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #6b92ff;
}
QComboBox QAbstractItemView {
    color: #e8edf5;
    background: #151e2c;
    border: 1px solid #344258;
    selection-background-color: #243a66;
    selection-color: #dbe6ff;
    outline: 0;
}
QCheckBox, QRadioButton, QLabel {
    background: transparent;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
}

QPushButton {
    color: #d9e1ec;
    background: #1b2636;
    border: 1px solid #344258;
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 20px;
    font-weight: 600;
}
QPushButton:hover {
    color: #ffffff;
    background: #243246;
    border-color: #475873;
}
QPushButton:pressed {
    background: #182334;
}
QPushButton:disabled {
    color: #627086;
    background: #151d29;
    border-color: #273245;
}
QPushButton#PrimaryButton {
    color: #ffffff;
    background: #4f7cff;
    border-color: #4f7cff;
    font-weight: 700;
}
QPushButton#PrimaryButton:hover {
    background: #3f6be6;
    border-color: #3f6be6;
}

QTableView, QTableWidget {
    color: #e8edf5;
    background: #111923;
    alternate-background-color: #151e2c;
    border: 1px solid #2b374a;
    border-radius: 9px;
    gridline-color: #202b3b;
    selection-background-color: #243a66;
    selection-color: #ffffff;
}
QHeaderView::section {
    color: #aeb9ca;
    background: #182230;
    padding: 8px 9px;
    border: none;
    border-bottom: 1px solid #2b374a;
    border-right: 1px solid #202b3b;
    font-weight: 700;
}
QTableCornerButton::section {
    background: #182230;
    border: none;
}

QTabWidget::pane {
    background: #151e2c;
    border: 1px solid #2b374a;
    border-radius: 9px;
    top: -1px;
}
QTabBar::tab {
    color: #8d9aaf;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 13px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:hover {
    color: #dce4ef;
}
QTabBar::tab:selected {
    color: #aFC4ff;
    border-bottom-color: #5f88ff;
}

QProgressBar {
    color: #aeb9ca;
    background: #202b3b;
    border: none;
    border-radius: 6px;
    text-align: center;
    min-height: 12px;
}
QProgressBar::chunk {
    background: #4f7cff;
    border-radius: 6px;
}

QToolBar#TopToolbar {
    color: #d7deea;
    background: #111a28;
    border: none;
    border-bottom: 1px solid #263247;
    spacing: 5px;
    padding: 5px 10px;
}
QToolBar#TopToolbar QToolButton {
    color: #aeb9ca;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 9px;
    font-weight: 600;
}
QToolBar#TopToolbar QToolButton:hover {
    color: #ffffff;
    background: #1b2738;
    border-color: #2c394d;
}
QMenu {
    color: #e8edf5;
    background: #151e2c;
    border: 1px solid #344258;
    padding: 5px;
}
QMenu::item {
    padding: 6px 22px 6px 10px;
    border-radius: 5px;
}
QMenu::item:selected {
    color: #dbe6ff;
    background: #243a66;
}
QToolTip {
    color: #e8edf5;
    background: #1a2433;
    border: 1px solid #3a485e;
    padding: 4px;
}

QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #435168;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #5c6d87;
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
    background: #435168;
    min-width: 30px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
"""

# Guided workflow and persistent task-status additions.
DARK_STYLESHEET += r"""
#WorkspaceContainer {
    background: #0f1520;
}
#WorkflowGuide {
    background: #111a28;
    border-bottom: 1px solid #263247;
}
#WorkflowStep {
    color: #8fb0ff;
    font-size: 9pt;
    font-weight: 750;
}
#WorkflowInstruction {
    color: #a2aec0;
    font-size: 9.4pt;
}
#WorkflowNextButton {
    color: #dbe6ff;
    background: #1c315b;
    border: 1px solid #29477d;
    font-weight: 700;
}
#WorkflowNextButton:hover {
    background: #243f72;
    border-color: #3b5f9d;
}
#NavButton[workflowState="locked"] {
    color: #59667a;
    background: transparent;
}
#NavButton[workflowState="completed"] {
    color: #6fd7b0;
}
#NavButton[workflowState="recommended"] {
    color: #b9ccff;
    background: #1a2a48;
    border-color: #29477d;
    font-weight: 700;
}
#NavButton[workflowState="optional"] {
    color: #d6b978;
}
#GlobalTaskState {
    min-width: 66px;
    font-weight: 800;
    color: #c9d2df;
}
#GlobalTaskState[taskState="busy"] { color: #9bb6ff; }
#GlobalTaskState[taskState="completed"] { color: #73d7b4; }
#GlobalTaskState[taskState="failed"] { color: #ff9b91; }
#GlobalTaskDetail {
    color: #8f9caf;
}
#GlobalTaskElapsed {
    color: #8f9caf;
}
#GlobalTaskProgress {
    min-height: 10px;
    max-height: 16px;
}
#StatusCancelButton {
    padding: 3px 9px;
    min-height: 16px;
    color: #ffb4ac;
    border-color: #74423e;
    background: #3a2424;
}
"""
DARK_STYLESHEET += r"""
#GlobalTaskState[taskState="cancelled"] { color: #d6b978; }
"""

# Compact plot editing tools and focused popup editors.
DARK_STYLESHEET += r"""
#PlotToolStrip {
    background: #151e2c;
    border: 1px solid #2a3548;
    border-radius: 10px;
}
#PlotToolStripLabel {
    color: #8d9aaf;
    font-size: 8.8pt;
    font-weight: 700;
    padding: 0 5px 0 2px;
}
#PlotToolStripSeparator {
    color: #2a3548;
    max-height: 24px;
}
QToolButton#PlotToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px;
}
QToolButton#PlotToolButton:hover {
    background: #1b2738;
    border-color: #344258;
}
QToolButton#PlotToolButton:pressed {
    background: #1c315b;
    border-color: #36598f;
}
#PlotToolPopup {
    background: #151e2c;
    border: 1px solid #344258;
    border-radius: 12px;
}
#PlotToolPopupTitle {
    color: #f4f7fb;
    font-size: 12.5pt;
    font-weight: 750;
}
#PlotToolPopupDescription {
    color: #8d9aaf;
    font-size: 9pt;
}
#PlotToolPopupDivider {
    color: #2a3548;
}
#PlotToolSectionTitle {
    color: #d8e0eb;
    font-size: 9pt;
    font-weight: 750;
    padding-top: 2px;
}
#PlotToolPopupNote {
    color: #aeb9ca;
    background: #111923;
    border: 1px solid #2b374a;
    border-radius: 7px;
    padding: 7px 9px;
}
"""
