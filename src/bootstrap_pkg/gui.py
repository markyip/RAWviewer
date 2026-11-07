"""InstallerGUI — the setup wizard window."""
from __future__ import annotations

import contextlib
import os
import subprocess

from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .helpers import (
    _apply_installer_process_icon,
    _installer_icon_path,
)
from .worker import InstallWorker

# Darkroom palette — keep in sync with src/theme.py
_VOID = "#14120f"
_SURFACE = "#1d1a16"
_RAISED = "#272219"
_RAISED_HI = "#302a1f"
_LINE = "#3a332a"
_LINE_SOFT = "#2a251d"
_INK = "#ede7dd"
_INK_MUTED = "#96897a"
_INK_FAINT = "#665d50"
_EMBER = "#d9691e"
_EMBER_HOVER = "#e87f39"
_HIST_R = "#e5484d"
_HIST_G = "#3dd68c"

class InstallerGUI(QMainWindow):
    _SUCCESS_DESC_BASE = (
        "Installation complete.\n\n"
        "• Click Launch, or open RAWviewer from the Desktop shortcut\n"
        "• Re-run RAWviewer_Setup.exe in the install folder to repair\n"
        "• Uninstall anytime from Windows Settings → Apps → RAWviewer"
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAWviewer Setup")
        self.setFixedSize(650, 560)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._install_in_progress = False
        self._awaiting_cancel = False
        self.worker = None
        self.thread = None
        self._close_after_failed_install = False

        self.init_ui()
        self.load_styles()
        _apply_installer_process_icon()
        icon_path = _installer_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

    def init_ui(self):
        self.main_container = QFrame()
        self.main_container.setObjectName("main_container")
        self.setCentralWidget(self.main_container)
        layout = QVBoxLayout(self.main_container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Custom Title Bar
        title_bar = QWidget()
        title_bar.setFixedHeight(40)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(20, 0, 10, 0)

        win_title = QLabel("RAWVIEWER SETUP")
        win_title.setStyleSheet(f"font-size: 11px; font-weight: 800; letter-spacing: 1px; color: {_INK_MUTED};")
        icon_path = _installer_icon_path()
        if icon_path:
            title_icon = QLabel()
            title_icon.setFixedSize(18, 18)
            title_icon.setPixmap(QIcon(icon_path).pixmap(18, 18))
            title_icon.setStyleSheet("background: transparent; border: none;")
            title_layout.addWidget(title_icon)
        title_layout.addWidget(win_title)
        title_layout.addStretch()

        btn_close = QPushButton("✕")
        btn_close.setFixedSize(30, 30)
        btn_close.setObjectName("btn_close")
        btn_close.clicked.connect(self.request_close)
        title_layout.addWidget(btn_close)
        layout.addWidget(title_bar)

        self.stack = QStackedWidget()

        # Page 1: Welcome
        self.page_welcome = QWidget()
        self.init_welcome_page()
        self.stack.addWidget(self.page_welcome)

        # Page 2: Progress
        self.page_progress = QWidget()
        self.init_progress_page()
        self.stack.addWidget(self.page_progress)

        # Page 3: Success
        self.page_success = QWidget()
        self.init_success_page()
        self.stack.addWidget(self.page_success)

        layout.addWidget(self.stack)

        # Bottom Bar
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("bottom_bar")
        self.bottom_bar.setFixedHeight(80)
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(40, 0, 40, 0)

        self.btn_cancel = QPushButton("CANCEL")
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_cancel.setFixedSize(120, 40)
        self.btn_cancel.clicked.connect(self.request_close)

        self.btn_next = QPushButton("INSTALL")
        self.btn_next.setObjectName("btn_next")
        self.btn_next.setFixedSize(140, 40)
        self.btn_next.clicked.connect(self.start_install)

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_cancel)
        bottom_layout.addWidget(self.btn_next)

        layout.addWidget(self.bottom_bar)

    def load_styles(self):
        self.setStyleSheet(f"""
            QFrame#main_container {{
                background-color: {_VOID};
                border-radius: 12px;
                border: 1px solid {_LINE};
            }}
            QLabel {{ color: {_INK}; font-family: 'Segoe UI', Arial; }}
            QLabel#title {{ font-size: 32px; font-weight: bold; }}
            QLabel#desc {{ color: {_INK_MUTED}; font-size: 14px; line-height: 1.5; }}
            QLineEdit {{
                background-color: {_SURFACE};
                border: 1px solid {_LINE};
                padding: 10px;
                border-radius: 6px;
                color: {_INK};
            }}
            QLineEdit:focus {{ border: 1px solid {_EMBER}; }}
            QPushButton {{
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton#btn_close {{
                background: transparent;
                color: {_INK_MUTED};
                font-size: 16px;
                border: none;
            }}
            QPushButton#btn_close:hover {{ color: {_HIST_R}; }}
            QPushButton#btn_cancel {{
                background: transparent;
                border: 1px solid {_LINE};
                color: {_INK};
            }}
            QPushButton#btn_cancel:hover {{ background: {_SURFACE}; border: 1px solid {_INK_FAINT}; }}
            QPushButton#btn_next {{
                background-color: {_EMBER};
                color: {_INK};
                border: none;
            }}
            QPushButton#btn_next:hover {{ background-color: {_EMBER_HOVER}; }}
            QPushButton#btn_next:disabled {{ background-color: {_RAISED_HI}; color: {_INK_FAINT}; }}
            QFrame#bottom_bar {{
                background-color: {_SURFACE};
                border-top: 1px solid {_LINE_SOFT};
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }}
            QProgressBar {{
                background-color: {_RAISED};
                border-radius: 6px;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background-color: {_EMBER};
                border-radius: 6px;
            }}
            QPlainTextEdit {{
                background-color: {_SURFACE};
                color: {_INK_MUTED};
                border: 1px solid {_LINE_SOFT};
                border-radius: 6px;
                font-family: Consolas, monospace;
                padding: 10px;
            }}
            QRadioButton {{
                color: {_INK};
                font-family: 'Segoe UI', Arial;
                font-size: 13px;
                min-height: 22px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 1px solid {_LINE};
                background: {_SURFACE};
            }}
            QRadioButton::indicator:checked {{
                border-radius: 8px;
                border: 1px solid {_EMBER};
                background: {_EMBER};
            }}
            QCheckBox {{
                color: {_INK};
                font-family: 'Segoe UI', Arial;
                font-size: 13px;
                min-height: 22px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {_LINE};
                background: {_SURFACE};
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid {_EMBER};
                background: {_EMBER};
            }}
        """)

    def init_welcome_page(self):
        layout = QVBoxLayout(self.page_welcome)
        layout.setContentsMargins(40, 20, 40, 20)

        title = QLabel("Ready to Install")
        title.setObjectName("title")
        layout.addWidget(title)

        desc_text = (
            "RAWviewer helps you review and cull RAW and JPEG photos quickly.\n\n"
            "Choose your edition below. "
            "Plus editions require internet access to download dependencies and AI models."
        )
        desc = QLabel(desc_text)
        desc.setObjectName("desc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        options_label = QLabel("Edition:")
        options_label.setStyleSheet(f"color: {_INK_MUTED}; font-weight: bold; margin-top: 15px;")
        layout.addWidget(options_label)

        self.radio_lite = QRadioButton(
            "Standard — Browse & Adjust (~0.8–1 GB; no AI search)"
        )
        self.radio_directml = QRadioButton(
            "Plus — DirectML (AI search; ~1.5–2 GB + ~600 MB models; CPU demosaic)"
        )
        self.radio_cuda = QRadioButton(
            "Plus — NVIDIA CUDA (AI search + CuPy GPU demosaic; ~1.5–2 GB + ~600 MB models)"
        )

        # Default Standard: lean install for most users.
        self.radio_lite.setChecked(True)

        options_layout = QVBoxLayout()
        options_layout.setSpacing(6)
        options_layout.addWidget(self.radio_lite)
        options_layout.addWidget(self.radio_directml)
        options_layout.addWidget(self.radio_cuda)
        layout.addLayout(options_layout)

        self.clear_cache_cb = QCheckBox(
            "Clear existing cache (recommended when upgrading from an older version)"
        )
        self.clear_cache_cb.setToolTip(
            "Removes the local photo cache (~/.rawviewer_cache) and session settings "
            "so v3 can use faster search/index defaults.\n"
            "Does not delete your photos or XMP sidecars.\n"
            "Plus builds may re-download AI models on first search."
        )
        self.clear_cache_cb.setChecked(False)
        layout.addWidget(self.clear_cache_cb)

        path_label = QLabel("Installation Directory:")
        path_label.setStyleSheet(f"color: {_INK_MUTED}; font-weight: bold; margin-top: 15px;")
        layout.addWidget(path_label)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        app_data = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        default_path = os.path.join(app_data, "RAWviewer")
        self.path_edit.setText(default_path)

        btn_browse = QPushButton("Browse...")
        btn_browse.setFixedSize(80, 36)
        btn_browse.setStyleSheet(f"background: {_RAISED}; border: 1px solid {_LINE}; color: {_INK};")
        btn_browse.clicked.connect(self.browse_path)

        row.addWidget(self.path_edit)
        row.addWidget(btn_browse)
        layout.addLayout(row)
        layout.addStretch()

    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", self.path_edit.text())
        if path:
            self.path_edit.setText(os.path.normpath(path))

    def init_progress_page(self):
        layout = QVBoxLayout(self.page_progress)
        layout.setContentsMargins(40, 20, 40, 20)

        title = QLabel("Installing...")
        title.setObjectName("title")
        layout.addWidget(title)

        self.install_step_label = QLabel("")
        self.install_step_label.setStyleSheet(f"color: {_INK_MUTED}; font-size: 13px;")
        layout.addWidget(self.install_step_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

    def init_success_page(self):
        layout = QVBoxLayout(self.page_success)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Success!")
        title.setObjectName("title")
        title.setStyleSheet(f"font-size: 40px; color: {_EMBER};")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        desc = QLabel(self._SUCCESS_DESC_BASE)
        desc.setObjectName("desc")
        desc.setWordWrap(True)
        self.success_desc = desc
        layout.addWidget(desc, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_install(self):
        target_dir = self.path_edit.text().strip()
        if not target_dir:
            return

        if self.radio_cuda.isChecked():
            mode = "cuda"
        elif self.radio_lite.isChecked():
            mode = "lite"
        else:
            mode = "directml"

        self._install_in_progress = True
        self._awaiting_cancel = False
        self.stack.setCurrentIndex(1)
        self.btn_next.setEnabled(False)
        self.btn_next.setText("INSTALLING...")
        self.btn_cancel.setText("CANCEL")
        self.log_box.clear()
        self.install_step_label.setText("")

        self.thread = QThread()
        self.worker = InstallWorker(
            target_dir, mode, clear_cache=self.clear_cache_cb.isChecked()
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.log_signal.connect(self.log_box.appendPlainText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.progress_label_signal.connect(self.install_step_label.setText)
        self.thread.start()

    def request_close(self):
        if self._install_in_progress:
            if self._awaiting_cancel:
                return
            self._awaiting_cancel = True
            self._close_after_failed_install = True
            self.btn_cancel.setEnabled(False)
            self.log_box.appendPlainText("Cancelling installation...")
            if self.worker is not None:
                self.worker.stop()
            return
        self.close()

    def on_finished(self, success, note=""):
        close_after_failure = self._close_after_failed_install
        self._install_in_progress = False
        self._awaiting_cancel = False
        self._close_after_failed_install = False
        self.btn_cancel.setEnabled(True)
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        self.thread = None
        self.worker = None

        if success:
            text = self._SUCCESS_DESC_BASE
            if note:
                text += f"\n\nNote: {note}"
            self.success_desc.setText(text)
            self.stack.setCurrentIndex(2)
            self.btn_next.setText("LAUNCH")
            self.btn_next.setEnabled(True)
            with contextlib.suppress(TypeError):
                self.btn_next.clicked.disconnect()
            self.btn_next.clicked.connect(self.launch_app)
            self.btn_cancel.hide()
        else:
            self.btn_next.setText("RETRY")
            self.btn_next.setEnabled(True)
            with contextlib.suppress(TypeError):
                self.btn_next.clicked.disconnect()
            self.btn_next.clicked.connect(self.retry_install)

        if not success and close_after_failure:
            self.close()

    def retry_install(self):
        self.stack.setCurrentIndex(0)
        self.btn_next.setText("INSTALL")
        self.btn_cancel.setText("CANCEL")
        self.btn_cancel.show()
        with contextlib.suppress(TypeError):
            self.btn_next.clicked.disconnect()
        self.btn_next.clicked.connect(self.start_install)
        self.progress_bar.setValue(0)

    def closeEvent(self, event):
        if self._install_in_progress:
            self.request_close()
            event.ignore()
            return
        super().closeEvent(event)

    def launch_app(self):
        install_dir = self.path_edit.text()
        app_exe = os.path.join(install_dir, "RAWviewer.exe")
        try:
            if os.path.isfile(app_exe):
                # Fully detach so Setup's onefile _MEI* cleanup is not blocked
                # by inherited handles from the child process.
                os.startfile(app_exe)  # noqa: S606 — intended Windows shell launch
            else:
                launcher_vbs = os.path.join(install_dir, "launcher.vbs")
                if os.path.isfile(launcher_vbs):
                    os.startfile(launcher_vbs)  # noqa: S606
                else:
                    return
        except OSError:
            if os.path.isfile(app_exe):
                flags = (
                    getattr(subprocess, "DETACHED_PROCESS", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
                subprocess.Popen(
                    [app_exe],
                    cwd=install_dir,
                    creationflags=flags,
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        # Let Qt unwind briefly before the onefile bootloader deletes _MEI*.

        QTimer.singleShot(400, self.close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_pos'):
            self.move(event.globalPosition().toPoint() - self.drag_pos)
