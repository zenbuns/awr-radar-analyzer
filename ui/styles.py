"""
Styles module for modern PyQt styling of the Radar Point Cloud Analyzer.
This module provides styling configurations for creating a modern UI experience.
"""

# Main stylesheet for the application with modern dark theme
DARK_STYLESHEET = """
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #1e1e2e;
}

/* Splitter styling */
QSplitter::handle {
    background-color: #45475a;
    width: 2px;
    height: 2px;
}

QSplitter::handle:horizontal:hover,
QSplitter::handle:vertical:hover {
    background-color: #89b4fa;
}

/* Menu styling */
QMenuBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 10px;
}

QMenuBar::item:selected {
    background-color: #313244;
    color: #89b4fa;
}

QMenu {
    background-color: #181825;
    border: 1px solid #313244;
}

QMenu::item {
    padding: 6px 20px 6px 20px;
}

QMenu::item:selected {
    background-color: #313244;
    color: #89b4fa;
}

/* Toolbar styling */
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 6px;
}

QToolBar::separator {
    background-color: #313244;
    width: 1px;
    height: 1px;
}

QToolButton {
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 6px;
}

QToolButton:hover {
    background-color: #313244;
}

QToolButton:pressed {
    background-color: #45475a;
}

/* GroupBox styling */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    padding-top: 4px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 6px;
}

/* Button styling */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #45475a;
    border: 1px solid #89b4fa;
}

QPushButton:pressed {
    background-color: #181825;
}

QPushButton:disabled {
    background-color: #181825;
    color: #6c7086;
    border: 1px solid #313244;
}

/* Primary action buttons */
QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    font-weight: bold;
}

QPushButton#primary:hover {
    background-color: #74c7ec;
}

QPushButton#primary:pressed {
    background-color: #89dceb;
}

/* Danger/warning buttons */
QPushButton#danger {
    background-color: #f38ba8;
    color: #1e1e2e;
    font-weight: bold;
}

QPushButton#danger:hover {
    background-color: #f5c2e7;
}

/* Checkboxes & Radio buttons */
QCheckBox, QRadioButton {
    spacing: 8px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
}

QCheckBox::indicator:unchecked,
QRadioButton::indicator:unchecked {
    border: 1px solid #45475a;
    background-color: #1e1e2e;
}

QCheckBox::indicator:checked,
QRadioButton::indicator:checked {
    border: 1px solid #89b4fa;
    background-color: #89b4fa;
}

QRadioButton::indicator:checked {
    border-radius: 8px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 5px;
    background: #313244;
    margin: 0px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #89b4fa;
    border: 1px solid #89b4fa;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #74c7ec;
    border: 1px solid #74c7ec;
}

/* Combo boxes and Spin boxes */
QComboBox, QSpinBox, QDoubleSpinBox {
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: #181825;
    selection-background-color: #313244;
}

QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    border-left: 1px solid #313244;
    width: 20px;
}

QComboBox::down-arrow {
    image: url('icons/down-arrow.png');
    width: 12px;
    height: 12px;
}

QComboBox:on {
    padding-top: 3px;
    padding-left: 4px;
}

QComboBox QAbstractItemView {
    border: 1px solid #313244;
    selection-background-color: #313244;
}

/* LineEdit */
QLineEdit {
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 4px 8px;
    background-color: #181825;
    selection-background-color: #89b4fa;
}

QLineEdit:focus {
    border: 1px solid #89b4fa;
}

/* StatusBar */
QStatusBar {
    background-color: #181825;
    border-top: 1px solid #313244;
}

/* ProgressBar */
QProgressBar {
    border: 1px solid #313244;
    border-radius: 4px;
    background-color: #1e1e2e;
    text-align: center;
    color: #cdd6f4;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    width: 20px;
}

/* TabWidget */
QTabWidget::pane {
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 6px;
}

QTabWidget::tab-bar {
    left: 14px;
}

QTabBar::tab {
    background-color: #181825;
    border: 1px solid #313244;
    border-bottom-color: #313244;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    min-width: 8ex;
    padding: 6px 10px;
}

QTabBar::tab:selected {
    background-color: #313244;
    border-bottom-color: #313244;
}

QTabBar::tab:hover {
    background-color: #45475a;
}

/* Matplotlib specific styling */
.QFrame {
    background-color: #1e1e2e;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #181825;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #89b4fa;
}

QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background-color: #181825;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: #45475a;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #89b4fa;
}

QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
    width: 0px;
}
"""

# Color palette
class Colors:
    DARK_BACKGROUND = "#252540"
    LIGHT_BACKGROUND = "#1a1a33"
    TEXT = "#ffffff"
    TEXT_MUTED = "#d0d0e0"
    BORDER = "#5555aa"
    BORDER_HIGHLIGHT = "#7777cc"
    ACCENT_BLUE = "#69c3ff"
    ACCENT_BLUE_HOVER = "#99ddff"
    ACCENT_GREEN = "#a6e3a1"
    ACCENT_RED = "#ff8ea2"
    ACCENT_YELLOW = "#ffe066"
    ACCENT_LAVENDER = "#ccbbff"

# Modern icon paths (to be added if custom icons are used)
class Icons:
    START = "icons/start.png"
    STOP = "icons/stop.png"
    RESET = "icons/reset.png"
    EXPORT = "icons/export.png"
    REPORT = "icons/report.png"
    SAVE = "icons/save.png"
    SETTINGS = "icons/settings.png"
    ADD = "icons/add.png"
    REMOVE = "icons/remove.png"
    EXIT = "icons/exit.png"
    
# Apply dark styling to matplotlib figures
def apply_mpl_style():
    """Apply enhanced eye-friendly dark style to matplotlib figures."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    plt.style.use('dark_background')
    
    # Customize style for better readability
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['grid.color'] = '#5555aa'
    plt.rcParams['grid.linewidth'] = 1.0
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.facecolor'] = '#252540'
    plt.rcParams['figure.facecolor'] = '#252540'
    plt.rcParams.update({
        'figure.facecolor': Colors.DARK_BACKGROUND,
        'axes.facecolor': Colors.DARK_BACKGROUND,
        'axes.edgecolor': Colors.BORDER,
        'axes.labelcolor': Colors.TEXT,
        'axes.grid': True,
        'grid.color': Colors.BORDER,
        'grid.alpha': 0.3,
        'text.color': Colors.TEXT,
        'xtick.color': Colors.TEXT_MUTED,
        'ytick.color': Colors.TEXT_MUTED,
        'savefig.facecolor': Colors.DARK_BACKGROUND,
        'savefig.edgecolor': 'none',
        'figure.figsize': (10, 8),
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })