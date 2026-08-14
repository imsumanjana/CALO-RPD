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
QMainWindow::separator {
    background: #263247;
    width: 1px;
    height: 1px;
}
QMainWindow::separator:hover {
    background: #344258;
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
QSpinBox, QDoubleSpinBox {
    padding-right: 30px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    width: 24px;
    background: #182843;
    border: 0;
    border-left: 1px solid #36598f;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-position: top right;
    border-bottom: 1px solid #2d4771;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #2853a3;
    border-left-color: #8fb0ff;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #1f4387;
}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
    background: #151f2d;
    border-left-color: #27364b;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 12px;
    height: 7px;
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
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator {
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
#GlobalTaskState[taskState="paused"] { color: #d6b978; }
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

# Phase 6 ribbon workspace, contextual panes, activity, and disabled-primary correction.
DARK_STYLESHEET += r"""
QPushButton#PrimaryButton:disabled {
    color: #657287;
    background: #182231;
    border-color: #2a3548;
}
#RibbonBar {
    background: #111923;
    border-bottom: 1px solid #2a3548;
}
#RibbonIdentityBar {
    background: #121c29;
    border: 0;
    border-bottom: 1px solid #344258;
}
#RibbonNavigationArea {
    background: #111923;
    border: 0;
}
#RibbonProduct {
    color: #dce6ff;
    font-size: 13pt;
    font-weight: 800;
}
#RibbonVersion {
    color: #9fb4d7;
    font-size: 9pt;
    font-weight: 600;
}
#RibbonState {
    color: #b9ccff;
    background: #182843;
    border: 1px solid #36598f;
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 8.8pt;
    font-weight: 650;
}
#RibbonTabs, #RibbonPageStack {
    background: #111923;
    border: 0;
}
QWidget#RibbonPage {
    background: #111923;
}
#RibbonCategoryRow {
    background: #111923;
    border: 0;
}
QPushButton#RibbonCategoryButton {
    color: #9aa8bc;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 5px 14px 4px 14px;
    border-radius: 0;
    min-height: 18px;
}
QPushButton#RibbonCategoryButton:checked {
    color: #dce6ff;
    background: transparent;
    border-bottom: 2px solid #6b92ff;
}
QPushButton#RibbonCategoryButton:hover:!checked {
    color: #c8d2e0;
    background: #151f2d;
}
QGroupBox#RibbonGroup {
    background: #151e2c;
    border: 1px solid #2a3548;
    border-radius: 6px;
    margin-top: 0;
    padding: 0;
}
#RibbonGroupCommands {
    background: transparent;
}
QLabel#RibbonGroupCaption {
    color: #8d9aaf;
    background: transparent;
    border: 0;
    padding: 0 4px 1px 4px;
    min-height: 12px;
    max-height: 12px;
    font-size: 7.8pt;
    font-weight: 650;
}
QToolButton#RibbonButton, QToolButton#RibbonPrimaryButton {
    color: #c8d2e0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 7px 8px 7px;
    min-width: 48px;
}
QToolButton#RibbonButton:hover, QToolButton#RibbonPrimaryButton:hover {
    background: #1c315b;
    border-color: #36598f;
}
QToolButton#RibbonPrimaryButton {
    color: #dce6ff;
    font-weight: 700;
}
QToolButton#RibbonButton:disabled, QToolButton#RibbonPrimaryButton:disabled {
    color: #657287;
    background: transparent;
}
QDockWidget {
    color: #c8d2e0;
    font-weight: 700;
}
QDockWidget::title {
    background: #111923;
    border: 0;
    border-bottom: 1px solid #2a3548;
    padding: 6px 9px;
    text-align: left;
}
#ContextPane, #ContextEditor, #ActivityCenter, #DocumentWorkspace {
    background: #151e2c;
}
#ContextTitle {
    color: #f8fafc;
    font-size: 15pt;
    font-weight: 750;
}
#ContextDescription, #ContextHelp {
    color: #8d9aaf;
}
#DocumentWorkspace::pane, #ActivityCenter::pane {
    border: 1px solid #2a3548;
    background: #151e2c;
}
QStatusBar #StatusCompute, QStatusBar #StatusDevice, QStatusBar #StatusMemory,
QStatusBar #StatusPolicy, QStatusBar #StatusVersion {
    color: #8d9aaf;
    padding: 0 5px;
    border-left: 1px solid #2a3548;
}
"""
DARK_STYLESHEET += r"""
#GlobalTaskState[taskState="cancelled"] { color: #d6b978; }
"""

# Professional shell refinement: concise headers, compact workspace palette, and quiet controls.
DARK_STYLESHEET += r"""
#PageTitle {
    font-size: 18pt;
}
#ContextTitle {
    font-size: 13pt;
}
QToolButton#WorkspaceRibbonButton {
    color: #c8d2e0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 4px 7px;
    min-width: 82px;
    min-height: 24px;
    text-align: left;
}
QToolButton#WorkspaceRibbonButton:hover {
    color: #dce6ff;
    background: #1c315b;
    border-color: #36598f;
}
QToolButton#WorkspaceRibbonButton:disabled {
    color: #657287;
    background: transparent;
}
QToolButton#DocumentCloseButton {
    color: #8d9aaf;
    background: transparent;
    border: 0;
    border-radius: 4px;
    padding: 0;
    font-size: 12pt;
}
QToolButton#DocumentCloseButton:hover {
    color: #f8fafc;
    background: #263247;
}
#ActivityCenter QTabBar::tab, #DocumentWorkspace QTabBar::tab {
    min-height: 24px;
    padding: 5px 12px;
}
#ActivityCenter::pane {
    border: 0;
    border-top: 1px solid #2a3548;
    border-radius: 0;
}
#ActivityCenter QTabBar {
    background: #111923;
}
#TrainingCasePicker {
    background: transparent;
}
#TrainingCaseBoundary {
    color: #8d9aaf;
    font-size: 8.6pt;
}
#TrainingInputLabel {
    background: transparent;
}
#TrainingInputCaption {
    color: #c8d2e0;
}
#TrainingActionBar {
    background: #111a28;
    border-top: 1px solid #2a3548;
}
QToolButton#TrainingInfoButton {
    color: #b9ccff;
    background: #182843;
    border: 1px solid #36598f;
    border-radius: 9px;
    padding: 0;
    font-size: 8.5pt;
    font-weight: 800;
}
QToolButton#TrainingInfoButton:hover,
QToolButton#TrainingInfoButton:focus {
    color: #ffffff;
    background: #2853a3;
    border-color: #8fb0ff;
}
#DocumentWorkspace QTabBar {
    min-height: 48px;
    background: #111923;
}
#DocumentWorkspace QTabBar::tab {
    min-height: 34px;
    min-width: 178px;
    padding: 7px 18px;
    margin: 5px 3px 0 5px;
    border: 1px solid transparent;
    border-radius: 9px 9px 0 0;
    font-size: 10pt;
    font-weight: 700;
}
#DocumentWorkspace QTabBar::tab:selected {
    color: #dce6ff;
    background: #151e2c;
    border-color: #344258;
    border-bottom-color: #151e2c;
}
#DocumentBrand {
    color: #dce6ff;
    background: #1c315b;
    border: 1px solid #36598f;
    border-radius: 9px;
    margin: 5px 8px 4px 4px;
}
#DocumentBrandName {
    color: #dce6ff;
    font-size: 10.5pt;
    font-weight: 850;
}
#MainPreviewScroll, #MainPreviewScroll > QWidget > QWidget {
    background: #101721;
}
#WorkspaceContainer {
    min-width: 920px;
    min-height: 650px;
}
#WorkspaceContainer QGroupBox, #WorkspaceContainer #SectionCard {
    min-height: 48px;
}
#WorkspaceContainer QTableView, #WorkspaceContainer QTableWidget {
    min-height: 220px;
}
#WorkspaceContainer QTextEdit, #WorkspaceContainer QPlainTextEdit {
    min-height: 150px;
}
#WorkspaceContainer QTabWidget {
    min-height: 320px;
}
QDockWidget::title {
    padding: 5px 9px;
}
"""

# Phase 3 semantic workspace, navigation, disclosure, and density tokens.
DARK_STYLESHEET += r"""
#WorkspaceSearch {
    background: #101824;
    border-color: #344258;
    min-height: 28px;
}
#NavigationCompactButton, #NavigationGroupHeader, #DisclosureToggle {
    color: #9aa8bc;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 7px;
    font-weight: 700;
    text-align: left;
}
#NavigationCompactButton:hover, #NavigationGroupHeader:hover, #DisclosureToggle:hover {
    color: #f8fafc;
    background: #192537;
    border-color: #344258;
}
#NavigationGroupHeader {
    color: #8290a6;
    font-size: 8.5pt;
}
#BlockedWorkspaceSummary {
    color: #d6b978;
    background: #2b2619;
    border: 1px solid #5f5130;
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 8.5pt;
}
#DisclosurePanel, #StudySetupWorkflow, #StudyLinkedStep {
    background: #151e2c;
    border: 1px solid #2a3548;
    border-radius: 12px;
}
#DisclosureSummary, #StudyStepProgress {
    color: #8d9aaf;
}
#StudySetupTitle {
    color: #f8fafc;
    font-size: 16pt;
    font-weight: 750;
}
#StudyStepButton {
    color: #9aa8bc;
    background: #111923;
    border-color: #2a3548;
    min-height: 36px;
    padding: 6px 8px;
}
#StudyStepButton:checked {
    color: #b9ccff;
    background: #1c315b;
    border-color: #36598f;
}
#NextActionStatus {
    color: #b9ccff;
    font-size: 11pt;
    font-weight: 700;
}
#InputChip {
    color: #b9ccff;
    background: #1c315b;
    border: 1px solid #36598f;
    border-radius: 8px;
    padding: 2px 7px;
    font-size: 8.5pt;
}
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus,
QTableView:focus, QTabBar::tab:focus {
    border: 2px solid #6b92ff;
}
#WorkspaceStack[interfaceDensity="compact"] QPushButton,
#WorkspaceStack[interfaceDensity="compact"] QLineEdit,
#WorkspaceStack[interfaceDensity="compact"] QComboBox,
#WorkspaceStack[interfaceDensity="compact"] QSpinBox,
#WorkspaceStack[interfaceDensity="compact"] QDoubleSpinBox {
    padding-top: 4px;
    padding-bottom: 4px;
}
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
