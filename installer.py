import os
import shutil
import sys
import subprocess
import time
import ctypes
import winreg
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar, QPlainTextEdit,
    QStackedWidget, QFileDialog, QFrame, QDialog,
)
from PySide6.QtGui import QIcon, QFont, QColor, QPalette
from PySide6.QtCore import Qt, Signal, QObject, QThread

# --- Versioning ---
APP_VERSION = "2.0.0"
APP_NAME = "SkySpotter"

# Paths
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = sys._MEIPASS
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = EXE_DIR

class InstallWorker(QObject):
    finished = Signal(bool)
    log_signal = Signal(str)
    progress_signal = Signal(int)

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = os.path.normpath(target_dir)
        self.cancelled = False

    def stop(self):
        self.cancelled = True

    def run(self):
        try:
            if self.cancelled: return
            target_dir = self.target_dir
            target_exe = os.path.join(target_dir, f"{APP_NAME}.exe")
            
            self.log_signal.emit(f"Installing to {target_dir}...")
            self.progress_signal.emit(10)

            # 1. Prepare directory
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            if self.cancelled: return
            self.progress_signal.emit(20)

            # 2. Copy Files (Assuming installer is next to 'SkySpotter' folder or includes it)
            # In our case, the installer will be built with --add-data "dist/SkySpotter;SkySpotter"
            src_folder = os.path.join(BUNDLE_DIR, APP_NAME)
            if not os.path.exists(src_folder):
                # Fallback for dev testing
                src_folder = os.path.join(EXE_DIR, "dist", APP_NAME)

            if os.path.exists(src_folder):
                self.log_signal.emit("Copying application files...")
                # Copy contents of src_folder to target_dir
                for item in os.listdir(src_folder):
                    if self.cancelled: return
                    s = os.path.join(src_folder, item)
                    d = os.path.join(target_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                self.log_signal.emit("Files copied successfully.")
            else:
                self.log_signal.emit(f"Error: Source folder not found at {src_folder}")
                self.finished.emit(False)
                return

            if self.cancelled: return
            self.progress_signal.emit(70)

            # 3. Create Shortcuts
            self.log_signal.emit("Creating shortcuts...")
            self.create_shortcuts(target_exe, target_dir)

            if self.cancelled: return
            self.progress_signal.emit(90)

            # 4. Register Uninstaller
            self.log_signal.emit("Registering uninstaller...")
            self.register_uninstaller(target_dir, target_exe)

            self.progress_signal.emit(100)
            self.log_signal.emit("Installation complete!")
            self.finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Fatal Error: {e}")
            self.finished.emit(False)

    def create_shortcuts(self, target_exe, target_dir):
        try:
            # Start Menu
            start_menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
            if not os.path.exists(start_menu): os.makedirs(start_menu)
            self.create_lnk(target_exe, target_dir, os.path.join(start_menu, f"{APP_NAME}.lnk"))
            
            # Desktop
            desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
            self.create_lnk(target_exe, target_dir, os.path.join(desktop, f"{APP_NAME}.lnk"))
        except Exception as e:
            self.log_signal.emit(f"Shortcut error: {e}")

    def create_lnk(self, target_exe, target_dir, lnk_path):
        vbs_script = f'Set oWS = WScript.CreateObject("WScript.Shell")\nsLinkFile = "{lnk_path}"\nSet oLink = oWS.CreateShortcut(sLinkFile)\noLink.TargetPath = "{target_exe}"\noLink.WorkingDirectory = "{target_dir}"\noLink.Save'
        vbs_file = os.path.join(os.environ['TEMP'], f"mkshortcut_{os.getpid()}.vbs")
        with open(vbs_file, "w") as f: f.write(vbs_script)
        subprocess.call(["cscript", "//nologo", vbs_file], creationflags=subprocess.CREATE_NO_WINDOW)
        if os.path.exists(vbs_file): os.remove(vbs_file)

    def register_uninstaller(self, install_dir, exe_path):
        try:
            key_path = fr"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}"
            uninst_path = os.path.join(install_dir, "uninstall.bat")
            silent_cmd = f'powershell.exe -WindowStyle Hidden -Command "& \'{uninst_path}\' __CLEANUP__ \'{install_dir}\'"'
            
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
                winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, exe_path)
                winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
                winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, silent_cmd)
                winreg.SetValueEx(key, "QuietUninstallString", 0, winreg.REG_SZ, silent_cmd)
                winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, install_dir)
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "SkySpotter Team")
                winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)

            # Refresh shell
            ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 1000, ctypes.byref(ctypes.wintypes.DWORD()))
        except Exception as e:
            self.log_signal.emit(f"Registry error: {e}")

class ModernDialog(QDialog):
    def __init__(self, parent=None, title=APP_NAME, message="", subtext="", buttons=["OK"]):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.container = QFrame()
        self.container.setStyleSheet("QFrame { background-color: #121212; border: 1px solid #333; border-radius: 12px; }")
        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(20, 20, 20, 20)
        
        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet("color: #666; font-weight: 900; font-size: 10px; letter-spacing: 2px;")
        c_layout.addWidget(t_lbl)
        
        m_lbl = QLabel(message)
        m_lbl.setStyleSheet("color: white; font-size: 16px; font-weight: 600;")
        m_lbl.setWordWrap(True)
        c_layout.addWidget(m_lbl)
        
        if subtext:
            s_lbl = QLabel(subtext)
            s_lbl.setStyleSheet("color: #999; font-size: 13px;")
            s_lbl.setWordWrap(True)
            c_layout.addWidget(s_lbl)
            
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        for b_txt in buttons:
            btn = QPushButton(b_txt)
            btn.setMinimumSize(80, 36)
            if b_txt in ["OK", "INSTALL", "Yes"]:
                btn.setStyleSheet("QPushButton { background: white; color: black; font-weight: 800; border-radius: 6px; }")
                btn.clicked.connect(lambda: self.done(1))
            else:
                btn.setStyleSheet("QPushButton { background: #333; color: white; border-radius: 6px; }")
                btn.clicked.connect(self.reject)
            btn_layout.addWidget(btn)
        c_layout.addLayout(btn_layout)
        layout.addWidget(self.container)

class InstallerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} Setup")
        self.setFixedSize(600, 450)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.init_ui()
        
    def init_ui(self):
        self.main_container = QWidget()
        self.main_container.setObjectName("main")
        self.main_container.setStyleSheet("QWidget#main { background-color: #121212; border-radius: 12px; border: 1px solid #333; }")
        self.setCentralWidget(self.main_container)
        layout = QVBoxLayout(self.main_container)
        
        # Title Bar
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(20, 10, 10, 0)
        t_lbl = QLabel(f"{APP_NAME.upper()} SETUP")
        t_lbl.setStyleSheet("color: #666; font-weight: 900; font-size: 10px; letter-spacing: 2px;")
        title_bar.addWidget(t_lbl)
        title_bar.addStretch()
        close_btn = QPushButton("×")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("QPushButton { background: transparent; color: #666; font-size: 20px; border: none; } QPushButton:hover { color: white; }")
        close_btn.clicked.connect(self.close)
        title_bar.addWidget(close_btn)
        layout.addLayout(title_bar)
        
        self.stack = QStackedWidget()
        
        # Page 1: Welcome & Path
        p1 = QWidget()
        p1_l = QVBoxLayout(p1)
        p1_l.setContentsMargins(40, 20, 40, 20)
        title = QLabel(f"Install {APP_NAME}")
        title.setStyleSheet("font-size: 32px; font-weight: 800; color: white;")
        p1_l.addWidget(title)
        desc = QLabel("SkySpotter is a professional RAW image viewer specialized for aviation photography.")
        desc.setStyleSheet("color: #999; font-size: 14px;")
        desc.setWordWrap(True)
        p1_l.addWidget(desc)
        p1_l.addSpacing(20)
        
        path_label = QLabel("INSTALLATION DIRECTORY")
        path_label.setStyleSheet("color: #666; font-weight: 800; font-size: 10px;")
        p1_l.addWidget(path_label)
        
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setText(os.path.join(os.environ.get('LOCALAPPDATA', 'C:'), APP_NAME))
        self.path_edit.setStyleSheet("QLineEdit { background: #1a1a1a; border: 1px solid #333; padding: 10px; color: white; border-radius: 6px; }")
        path_row.addWidget(self.path_edit)
        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet("QPushButton { background: #333; color: white; padding: 10px; border-radius: 6px; }")
        btn_browse.clicked.connect(self.browse_path)
        path_row.addWidget(btn_browse)
        p1_l.addLayout(path_row)
        p1_l.addStretch()
        
        self.btn_install = QPushButton("INSTALL")
        self.btn_install.setFixedHeight(44)
        self.btn_install.setStyleSheet("QPushButton { background: white; color: black; font-weight: 900; border-radius: 6px; } QPushButton:hover { background: #eee; }")
        self.btn_install.clicked.connect(self.start_install)
        p1_l.addWidget(self.btn_install)
        
        self.stack.addWidget(p1)
        
        # Page 2: Progress
        p2 = QWidget()
        p2_l = QVBoxLayout(p2)
        p2_l.setContentsMargins(40, 20, 40, 20)
        p2_title = QLabel("Installing...")
        p2_title.setStyleSheet("font-size: 24px; font-weight: 800; color: white;")
        p2_l.addWidget(p2_title)
        self.progress = QProgressBar()
        self.progress.setStyleSheet("QProgressBar { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; height: 10px; text-align: center; } QProgressBar::chunk { background: white; border-radius: 6px; }")
        p2_l.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("QPlainTextEdit { background: #000; color: #666; font-family: Consolas; font-size: 11px; border: 1px solid #222; }")
        p2_l.addWidget(self.log)
        self.stack.addWidget(p2)
        
        # Page 3: Success
        p3 = QWidget()
        p3_l = QVBoxLayout(p3)
        p3_l.setContentsMargins(40, 20, 40, 20)
        p3_l.setAlignment(Qt.AlignCenter)
        s_title = QLabel("Success!")
        s_title.setStyleSheet("font-size: 48px; font-weight: 800; color: white;")
        p3_l.addWidget(s_title)
        s_desc = QLabel(f"{APP_NAME} has been installed.")
        s_desc.setStyleSheet("color: #999; font-size: 16px;")
        p3_l.addWidget(s_desc)
        p3_l.addSpacing(20)
        btn_launch = QPushButton("LAUNCH")
        btn_launch.setFixedSize(200, 44)
        btn_launch.setStyleSheet("QPushButton { background: white; color: black; font-weight: 900; border-radius: 6px; }")
        btn_launch.clicked.connect(self.launch_app)
        p3_l.addWidget(btn_launch)
        self.stack.addWidget(p3)
        
        layout.addWidget(self.stack)

    def browse_path(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder", self.path_edit.text())
        if p: self.path_edit.setText(os.path.normpath(p))
        
    def start_install(self):
        self.stack.setCurrentIndex(1)
        self.thread = QThread()
        self.worker = InstallWorker(self.path_edit.text())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.log_signal.connect(self.log.appendPlainText)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.thread.start()
        
    def on_finished(self, ok):
        self.thread.quit()
        if ok: self.stack.setCurrentIndex(2)
        else:
            ModernDialog(self, "Error", "Installation failed.").exec()
            self.stack.setCurrentIndex(0)
            
    def launch_app(self):
        exe = os.path.join(self.path_edit.text(), f"{APP_NAME}.exe")
        if os.path.exists(exe): subprocess.Popen([exe])
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = InstallerGUI()
    gui.show()
    sys.exit(app.exec())
