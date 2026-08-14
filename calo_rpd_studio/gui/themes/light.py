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
QMainWindow::separator {
    background: #dce4ee;
    width: 1px;
    height: 1px;
}
QMainWindow::separator:hover {
    background: #cbd5e1;
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
QSpinBox, QDoubleSpinBox {
    padding-right: 30px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    width: 24px;
    background: #edf4ff;
    border: 0;
    border-left: 1px solid #a9c5f5;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-position: top right;
    border-bottom: 1px solid #c7daf8;
    border-top-right-radius: 6px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-position: bottom right;
    border-bottom-right-radius: 6px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #2563eb;
    border-left-color: #1d4ed8;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #1d4ed8;
}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {
    background: #f2f4f7;
    border-left-color: #dce3ed;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 12px;
    height: 7px;
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
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator {
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
#GlobalTaskState[taskState="paused"] { color: #7c5c16; }
#GlobalTaskDetail {
    color: #64748b;
}
#GlobalTaskElapsed {
    color: #64748b;
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

# Phase 6 ribbon workspace, contextual panes, activity, and disabled-primary correction.
LIGHT_STYLESHEET += r"""
QPushButton#PrimaryButton:disabled {
    color: #9aa6b6;
    background: #e9edf3;
    border-color: #d8e0ea;
}
#RibbonBar {
    background: #ffffff;
    border-bottom: 1px solid #dce4ee;
}
#RibbonIdentityBar {
    background: #f8fbff;
    border: 0;
    border-bottom: 1px solid #cbdcf7;
}
#RibbonNavigationArea {
    background: #ffffff;
    border: 0;
}
#RibbonProduct {
    color: #123a82;
    font-size: 13pt;
    font-weight: 800;
}
#RibbonVersion {
    color: #526f9f;
    font-size: 9pt;
    font-weight: 600;
}
#RibbonState {
    color: #1d4ed8;
    background: #edf4ff;
    border: 1px solid #a9c5f5;
    border-radius: 9px;
    padding: 3px 9px;
    font-size: 8.8pt;
    font-weight: 650;
}
#RibbonTabs, #RibbonPageStack {
    background: #ffffff;
    border: 0;
}
QWidget#RibbonPage {
    background: #ffffff;
}
#RibbonCategoryRow {
    background: #ffffff;
    border: 0;
}
QPushButton#RibbonCategoryButton {
    color: #526174;
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    padding: 5px 14px 4px 14px;
    border-radius: 0;
    min-height: 18px;
}
QPushButton#RibbonCategoryButton:checked {
    color: #1d4ed8;
    background: transparent;
    border-bottom: 2px solid #2563eb;
}
QPushButton#RibbonCategoryButton:hover:!checked {
    color: #0f172a;
    background: #f1f5fb;
}
QGroupBox#RibbonGroup {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-top: 0;
    padding: 0;
}
#RibbonGroupCommands {
    background: transparent;
}
QLabel#RibbonGroupCaption {
    color: #64748b;
    background: transparent;
    border: 0;
    padding: 0 4px 1px 4px;
    min-height: 12px;
    max-height: 12px;
    font-size: 7.8pt;
    font-weight: 650;
}
QToolButton#RibbonButton, QToolButton#RibbonPrimaryButton {
    color: #334155;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 7px 8px 7px;
    min-width: 48px;
}
QToolButton#RibbonButton:hover, QToolButton#RibbonPrimaryButton:hover {
    background: #eaf1ff;
    border-color: #cfe0ff;
}
QToolButton#RibbonPrimaryButton {
    color: #143d91;
    font-weight: 700;
}
QToolButton#RibbonButton:disabled, QToolButton#RibbonPrimaryButton:disabled {
    color: #9aa6b6;
    background: transparent;
}
QDockWidget {
    color: #334155;
    font-weight: 700;
}
QDockWidget::title {
    background: #edf2f8;
    border: 0;
    border-bottom: 1px solid #dce4ee;
    padding: 6px 9px;
    text-align: left;
}
#ContextPane, #ContextEditor, #ActivityCenter, #DocumentWorkspace {
    background: #ffffff;
}
#ContextTitle {
    color: #0f172a;
    font-size: 15pt;
    font-weight: 750;
}
#ContextDescription, #ContextHelp {
    color: #64748b;
}
#DocumentWorkspace::pane, #ActivityCenter::pane {
    border: 1px solid #dce4ee;
    background: #ffffff;
}
QStatusBar #StatusCompute, QStatusBar #StatusDevice, QStatusBar #StatusMemory,
QStatusBar #StatusPolicy, QStatusBar #StatusVersion {
    color: #526174;
    padding: 0 5px;
    border-left: 1px solid #e2e8f0;
}
"""
LIGHT_STYLESHEET += r"""
#GlobalTaskState[taskState="cancelled"] { color: #7c5c16; }
"""

# Professional shell refinement: concise headers, compact workspace palette, and quiet controls.
LIGHT_STYLESHEET += r"""
#PageTitle {
    font-size: 18pt;
}
#ContextTitle {
    font-size: 13pt;
}
QToolButton#WorkspaceRibbonButton {
    color: #334155;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 4px 7px;
    min-width: 82px;
    min-height: 24px;
    text-align: left;
}
QToolButton#WorkspaceRibbonButton:hover {
    color: #1d4ed8;
    background: #edf4ff;
    border-color: #cbdcf7;
}
QToolButton#WorkspaceRibbonButton:disabled {
    color: #a6b0bf;
    background: transparent;
}
QToolButton#DocumentCloseButton {
    color: #64748b;
    background: transparent;
    border: 0;
    border-radius: 4px;
    padding: 0;
    font-size: 12pt;
}
QToolButton#DocumentCloseButton:hover {
    color: #0f172a;
    background: #e7edf6;
}
#ActivityCenter QTabBar::tab, #DocumentWorkspace QTabBar::tab {
    min-height: 24px;
    padding: 5px 12px;
}
#ActivityCenter::pane {
    border: 0;
    border-top: 1px solid #dce4ee;
    border-radius: 0;
}
#ActivityCenter QTabBar {
    background: #f8fafc;
}
#TrainingCasePicker {
    background: transparent;
}
#TrainingCaseBoundary {
    color: #64748b;
    font-size: 8.6pt;
}
#TrainingInputLabel {
    background: transparent;
}
#TrainingInputCaption {
    color: #334155;
}
#TrainingActionBar {
    background: #f8fafc;
    border-top: 1px solid #dce4ee;
}
QToolButton#TrainingInfoButton {
    color: #1d4ed8;
    background: #edf4ff;
    border: 1px solid #a9c5f5;
    border-radius: 9px;
    padding: 0;
    font-size: 8.5pt;
    font-weight: 800;
}
QToolButton#TrainingInfoButton:hover,
QToolButton#TrainingInfoButton:focus {
    color: #ffffff;
    background: #2563eb;
    border-color: #1d4ed8;
}
#DocumentWorkspace QTabBar {
    min-height: 48px;
    background: #f8fafc;
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
    color: #123a82;
    background: #ffffff;
    border-color: #dce4ee;
    border-bottom-color: #ffffff;
}
#DocumentBrand {
    color: #123a82;
    background: #edf4ff;
    border: 1px solid #cbdcf7;
    border-radius: 9px;
    margin: 5px 8px 4px 4px;
}
#DocumentBrandName {
    color: #123a82;
    font-size: 10.5pt;
    font-weight: 850;
}
#MainPreviewScroll, #MainPreviewScroll > QWidget > QWidget {
    background: #f6f8fb;
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
LIGHT_STYLESHEET += r"""
#WorkspaceSearch {
    background: #f8fafc;
    border-color: #dce4ef;
    min-height: 28px;
}
#NavigationCompactButton, #NavigationGroupHeader, #DisclosureToggle {
    color: #475569;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 7px;
    font-weight: 700;
    text-align: left;
}
#NavigationCompactButton:hover, #NavigationGroupHeader:hover, #DisclosureToggle:hover {
    color: #0f172a;
    background: #f1f5f9;
    border-color: #e2e8f0;
}
#NavigationGroupHeader {
    color: #64748b;
    font-size: 8.5pt;
}
#BlockedWorkspaceSummary {
    color: #7c5c16;
    background: #fffbeb;
    border: 1px solid #fde7a7;
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 8.5pt;
}
#DisclosurePanel, #StudySetupWorkflow, #StudyLinkedStep {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}
#DisclosureSummary, #StudyStepProgress {
    color: #64748b;
}
#StudySetupTitle {
    color: #0f172a;
    font-size: 16pt;
    font-weight: 750;
}
#StudyStepButton {
    color: #64748b;
    background: #f8fafc;
    border-color: #e2e8f0;
    min-height: 36px;
    padding: 6px 8px;
}
#StudyStepButton:checked {
    color: #1d4ed8;
    background: #eaf1ff;
    border-color: #bfd3ff;
}
#NextActionStatus {
    color: #1e3a8a;
    font-size: 11pt;
    font-weight: 700;
}
#InputChip {
    color: #1d4ed8;
    background: #eaf1ff;
    border: 1px solid #cfe0ff;
    border-radius: 8px;
    padding: 2px 7px;
    font-size: 8.5pt;
}
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus,
QTableView:focus, QTabBar::tab:focus {
    border: 2px solid #2563eb;
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
LIGHT_STYLESHEET += r"""
#PlotToolStrip {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}
#PlotToolStripLabel {
    color: #64748b;
    font-size: 8.8pt;
    font-weight: 700;
    padding: 0 5px 0 2px;
}
#PlotToolStripSeparator {
    color: #e2e8f0;
    max-height: 24px;
}
QToolButton#PlotToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px;
}
QToolButton#PlotToolButton:hover {
    background: #f1f5f9;
    border-color: #dbe4ef;
}
QToolButton#PlotToolButton:pressed {
    background: #eaf1ff;
    border-color: #cbdcff;
}
#PlotToolPopup {
    background: #ffffff;
    border: 1px solid #d9e2ee;
    border-radius: 12px;
}
#PlotToolPopupTitle {
    color: #0f172a;
    font-size: 12.5pt;
    font-weight: 750;
}
#PlotToolPopupDescription {
    color: #64748b;
    font-size: 9pt;
}
#PlotToolPopupDivider {
    color: #e7ecf3;
}
#PlotToolSectionTitle {
    color: #334155;
    font-size: 9pt;
    font-weight: 750;
    padding-top: 2px;
}
#PlotToolPopupNote {
    color: #475569;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 7px;
    padding: 7px 9px;
}
"""
