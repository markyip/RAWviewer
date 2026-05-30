import os
import sys
import shutil
import subprocess
import zipfile
import urllib.request
import winreg

# IMPORTANT: Keep PyInstaller bundle dir
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EXE_DIR = BUNDLE_DIR

# Check if we should just run the app
if "--run" in sys.argv:
    # We are in the install directory. Find the pixi environment.
    cwd = EXE_DIR
    pixi_exe = os.path.join(cwd, "_internal", "pixi", "pixi.exe")
    if os.path.exists(pixi_exe):
        # Run using pixi
        subprocess.Popen([pixi_exe, "run", "start-windowless"], cwd=cwd, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        # Fallback if pixi is somehow missing
        pass
    sys.exit(0)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar, QPlainTextEdit,
    QStackedWidget, QFileDialog, QFrame, QDialog
)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

class InstallWorker(QObject):
    finished = pyqtSignal(bool)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = os.path.normpath(target_dir)
        self.cancelled = False

    def stop(self):
        self.cancelled = True

    def run(self):
        try:
            target_dir = self.target_dir
            os.makedirs(target_dir, exist_ok=True)
            self.progress_signal.emit(10)
            self.log_signal.emit(f"Installing to {target_dir}...")

            # Copy application files
            self.log_signal.emit("Copying core files...")
            shutil.copy2(os.path.join(BUNDLE_DIR, "pixi.toml"), target_dir)
            
            src_dir = os.path.join(target_dir, "src")
            if os.path.exists(src_dir): shutil.rmtree(src_dir)
            shutil.copytree(
                os.path.join(BUNDLE_DIR, "src"),
                src_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("logs", "*.log"),
            )
            
            assets_dir = os.path.join(target_dir, "icons")
            if os.path.exists(assets_dir): shutil.rmtree(assets_dir)
            shutil.copytree(os.path.join(BUNDLE_DIR, "icons"), assets_dir, dirs_exist_ok=True)
            
            scripts_dir = os.path.join(target_dir, "scripts")
            if os.path.exists(scripts_dir): shutil.rmtree(scripts_dir)
            if os.path.exists(os.path.join(BUNDLE_DIR, "scripts")):
                shutil.copytree(os.path.join(BUNDLE_DIR, "scripts"), scripts_dir, dirs_exist_ok=True)
            
            uninst_src = os.path.join(BUNDLE_DIR, "uninstall.bat")
            if os.path.exists(uninst_src):
                shutil.copy2(uninst_src, target_dir)

            # Copy the executable itself
            target_exe = os.path.join(target_dir, "RAWviewer.exe")
            if getattr(sys, 'frozen', False):
                shutil.copy2(sys.executable, target_exe)

            self.progress_signal.emit(30)

            # Download Pixi
            internal_dir = os.path.join(target_dir, "_internal")
            pixi_dir = os.path.join(internal_dir, "pixi")
            os.makedirs(pixi_dir, exist_ok=True)
            pixi_exe = os.path.join(pixi_dir, "pixi.exe")

            if not os.path.exists(pixi_exe):
                self.log_signal.emit("Downloading Pixi environment manager...")
                url = "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.zip"
                zip_path = os.path.join(pixi_dir, "pixi.zip")
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(pixi_dir)
                os.remove(zip_path)

            self.progress_signal.emit(50)

            # Install environment using Pixi
            self.log_signal.emit("Downloading AI Models & Python Dependencies. This may take a few minutes...")
            process = subprocess.Popen(
                [pixi_exe, "install", "-v"], cwd=target_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in process.stdout:
                if self.cancelled:
                    process.terminate()
                    self.log_signal.emit("Installation cancelled.")
                    return
                msg = line.strip()
                if msg:
                    self.log_signal.emit(msg)
            process.wait()
            
            if process.returncode != 0:
                self.log_signal.emit("Failed to install dependencies.")
                self.finished.emit(False)
                return

            self.progress_signal.emit(75)

            # Download AI Models
            self.log_signal.emit("Downloading MobileCLIP ONNX Models...")
            process_models = subprocess.Popen(
                [pixi_exe, "run", "python", "scripts/download_mobileclip_onnx.py"], cwd=target_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in process_models.stdout:
                if self.cancelled:
                    process_models.terminate()
                    self.log_signal.emit("Installation cancelled.")
                    return
                msg = line.strip()
                if msg:
                    self.log_signal.emit(msg)
            process_models.wait()
            
            if process_models.returncode != 0:
                self.log_signal.emit("Failed to download AI models.")
                self.finished.emit(False)
                return

            self.progress_signal.emit(90)

            # Create fast launcher
            self.log_signal.emit("Creating launcher...")
            launcher_script = f'''Set oWS = WScript.CreateObject("WScript.Shell")
oWS.CurrentDirectory = "{target_dir}"
oWS.Run "{pixi_exe} run start-windowless", 0, False
'''
            launcher_vbs_path = os.path.join(target_dir, "launcher.vbs")
            with open(launcher_vbs_path, "w", encoding="utf-8") as f:
                f.write(launcher_script)

            # Create Shortcuts
            self.log_signal.emit("Creating shortcuts...")
            try:
                desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
                programs = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
                
                vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{os.path.join(desktop, 'RAWviewer.lnk')}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{launcher_vbs_path}"
oLink.WorkingDirectory = "{target_dir}"
oLink.IconLocation = "{target_exe}"
oLink.Save

sLinkFile2 = "{os.path.join(programs, 'RAWviewer.lnk')}"
Set oLink2 = oWS.CreateShortcut(sLinkFile2)
oLink2.TargetPath = "{launcher_vbs_path}"
oLink2.WorkingDirectory = "{target_dir}"
oLink2.IconLocation = "{target_exe}"
oLink2.Save
"""
                vbs_path = os.path.join(target_dir, "create_shortcuts.vbs")
                with open(vbs_path, "w", encoding="utf-8") as f:
                    f.write(vbs_script)
                
                subprocess.run(["cscript.exe", "//Nologo", vbs_path], creationflags=subprocess.CREATE_NO_WINDOW)
                os.remove(vbs_path)
            except Exception as e:
                self.log_signal.emit(f"Could not create shortcut: {e}")
                
            self.progress_signal.emit(95)
            
            # Register Uninstaller
            self.log_signal.emit("Registering uninstaller...")
            try:
                import ctypes
                import time
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer"
                icon_path = target_exe
                install_date = time.strftime("%Y%m%d")
                uninst_path = os.path.join(target_dir, "uninstall.bat")
                silent_cmd = f'powershell.exe -WindowStyle Hidden -Command "& \'{uninst_path}\' __CLEANUP__ \'{target_dir}\'"'
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "RAWviewer")
                    winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, icon_path)
                    winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, "2.2.2")
                    winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "QuietUninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, target_dir)
                    winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Mark Yip")
                    winreg.SetValueEx(key, "InstallDate", 0, winreg.REG_SZ, install_date)
                    winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "WindowsInstaller", 0, winreg.REG_DWORD, 0)
                # Broadcast environment change
                result = ctypes.wintypes.DWORD()
                ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 1000, ctypes.byref(result))
            except Exception as e:
                self.log_signal.emit(f"Failed to register uninstaller: {e}")

            self.progress_signal.emit(100)
            self.finished.emit(True)

        except Exception as e:
            self.log_signal.emit(f"Error: {e}")
            self.finished.emit(False)

class InstallerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAWviewer Setup")
        self.setFixedSize(650, 500)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.init_ui()
        self.load_styles()

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
        win_title.setStyleSheet("font-size: 11px; font-weight: 800; letter-spacing: 1px; color: #888;")
        title_layout.addWidget(win_title)
        title_layout.addStretch()
        
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(30, 30)
        btn_close.setObjectName("btn_close")
        btn_close.clicked.connect(self.close)
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
        self.btn_cancel.clicked.connect(self.close)
        
        self.btn_next = QPushButton("INSTALL")
        self.btn_next.setObjectName("btn_next")
        self.btn_next.setFixedSize(140, 40)
        self.btn_next.clicked.connect(self.start_install)
        
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_cancel)
        bottom_layout.addWidget(self.btn_next)
        
        layout.addWidget(self.bottom_bar)

    def load_styles(self):
        self.setStyleSheet("""
            QFrame#main_container { 
                background-color: #121212; 
                border-radius: 12px;
                border: 1px solid #333;
            }
            QLabel { color: #ffffff; font-family: 'Segoe UI', Arial; }
            QLabel#title { font-size: 32px; font-weight: bold; }
            QLabel#desc { color: #aaa; font-size: 14px; line-height: 1.5; }
            QLineEdit { 
                background-color: #1e1e1e; 
                border: 1px solid #444; 
                padding: 10px; 
                border-radius: 6px; 
                color: #fff;
            }
            QPushButton { 
                border-radius: 6px; 
                font-weight: bold;
            }
            QPushButton#btn_close {
                background: transparent;
                color: #888;
                font-size: 16px;
                border: none;
            }
            QPushButton#btn_close:hover { color: #ff5555; }
            QPushButton#btn_cancel { 
                background: transparent; 
                border: 1px solid #555; 
                color: #fff; 
            }
            QPushButton#btn_cancel:hover { background: #1e1e1e; border: 1px solid #777; }
            QPushButton#btn_next { 
                background-color: #3b82f6; 
                color: #fff; 
                border: none; 
            }
            QPushButton#btn_next:hover { background-color: #60a5fa; }
            QFrame#bottom_bar {
                background-color: #1a1a1a;
                border-top: 1px solid #2a2a2a;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
            }
            QProgressBar {
                background-color: #1e1e1e;
                border-radius: 6px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 6px;
            }
            QPlainTextEdit {
                background-color: #0a0a0a;
                color: #888;
                border: 1px solid #222;
                border-radius: 6px;
                font-family: Consolas, monospace;
                padding: 10px;
            }
        """)

    def init_welcome_page(self):
        layout = QVBoxLayout(self.page_welcome)
        layout.setContentsMargins(40, 20, 40, 20)
        
        title = QLabel("Ready to Install")
        title.setObjectName("title")
        layout.addWidget(title)
        
        desc = QLabel("RAWviewer is a professional photo culling tool powered by AI.\nThis setup will download the required AI models and optimized environments.")
        desc.setObjectName("desc")
        layout.addWidget(desc)
        
        path_label = QLabel("Installation Directory:")
        path_label.setStyleSheet("color: #888; font-weight: bold; margin-top: 20px;")
        layout.addWidget(path_label)
        
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        default_path = os.path.join(os.environ.get('LOCALAPPDATA', 'C:'), "RAWviewer")
        self.path_edit.setText(default_path)
        
        btn_browse = QPushButton("Browse...")
        btn_browse.setFixedSize(80, 36)
        btn_browse.setStyleSheet("background: #2a2a2a; border: 1px solid #444; color: #fff;")
        btn_browse.clicked.connect(self.browse_path)
        
        row.addWidget(self.path_edit)
        row.addWidget(btn_browse)
        layout.addLayout(row)
        layout.addStretch()

    def browse_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", self.path_edit.text())
        if path: self.path_edit.setText(os.path.normpath(path))

    def init_progress_page(self):
        layout = QVBoxLayout(self.page_progress)
        layout.setContentsMargins(40, 20, 40, 20)
        
        title = QLabel("Installing...")
        title.setObjectName("title")
        layout.addWidget(title)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        layout.addWidget(self.progress_bar)
        
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

    def init_success_page(self):
        layout = QVBoxLayout(self.page_success)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("Success!")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 40px; color: #4ade80;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        desc = QLabel("RAWviewer is now installed and ready to launch.")
        desc.setObjectName("desc")
        layout.addWidget(desc, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_install(self):
        target_dir = self.path_edit.text()
        self.stack.setCurrentIndex(1)
        self.btn_next.setEnabled(False)
        self.btn_next.setText("INSTALLING...")
        
        self.thread = QThread()
        self.worker = InstallWorker(target_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.log_signal.connect(self.log_box.appendPlainText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.thread.start()

    def on_finished(self, success):
        if success:
            self.stack.setCurrentIndex(2)
            self.btn_next.setText("LAUNCH")
            self.btn_next.setEnabled(True)
            self.btn_next.clicked.disconnect()
            self.btn_next.clicked.connect(self.launch_app)
            self.btn_cancel.hide()
        else:
            self.btn_next.setText("FAILED")

    def launch_app(self):
        launcher_vbs = os.path.join(self.path_edit.text(), "launcher.vbs")
        subprocess.Popen(["wscript.exe", launcher_vbs], creationflags=subprocess.CREATE_NO_WINDOW)
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_pos'):
            self.move(event.globalPosition().toPoint() - self.drag_pos)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InstallerGUI()
    window.show()
    sys.exit(app.exec())
