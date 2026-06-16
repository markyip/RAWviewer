import errno
import os
import re
import sys
import shutil
import subprocess
import zipfile
import urllib.request
import urllib.error
import ssl
import time
import winreg
PIXI_DOWNLOAD_URL = (
    "https://github.com/prefix-dev/pixi/releases/latest/download/"
    "pixi-x86_64-pc-windows-msvc.zip"
)
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SEC = 3
MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024  # ~3 GiB for pixi env + models (full)
MIN_FREE_BYTES_LITE = 1024 * 1024 * 1024  # ~1 GiB for lite (no AI models)
INSTALLER_PROGRESS_PREFIX = "@RAWVIEWER_PROGRESS"
MODEL_DOWNLOAD_PROGRESS_START = 5
MODEL_DOWNLOAD_PROGRESS_END = 100


def _is_lite_installer() -> bool:
    try:
        from build_profile import PROFILE

        return str(PROFILE).strip().lower() == "lite"
    except Exception:
        return False


def _min_free_bytes_for_install() -> int:
    return MIN_FREE_BYTES_LITE if _is_lite_installer() else MIN_FREE_BYTES


def _parse_installer_progress_line(line: str) -> tuple[int, str] | None:
    """Parse @RAWVIEWER_PROGRESS pct=N message=... from child download scripts."""
    idx = line.find(INSTALLER_PROGRESS_PREFIX)
    if idx < 0:
        return None
    payload = line[idx:]
    match = re.search(r"pct=(\d+)\s+message=(.*)", payload)
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def _map_model_download_progress(pct: int) -> int:
    span = MODEL_DOWNLOAD_PROGRESS_END - MODEL_DOWNLOAD_PROGRESS_START
    return MODEL_DOWNLOAD_PROGRESS_START + int(max(0, min(100, pct)) * span / 100)


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _check_disk_space(path: str, min_bytes: int = MIN_FREE_BYTES) -> str | None:
    """Return a user-facing error when the target drive is too full."""
    try:
        abs_path = os.path.abspath(path)
        drive, _ = os.path.splitdrive(abs_path)
        check_path = f"{drive}\\" if drive else abs_path
        free = shutil.disk_usage(check_path).free
        if free < min_bytes:
            return (
                f"Not enough free disk space on {check_path}: "
                f"{_human_bytes(free)} available, about {_human_bytes(min_bytes)} recommended."
            )
    except OSError as exc:
        return f"Could not check disk space: {exc}"
    return None


def _describe_download_error(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} from server"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLError):
            return "SSL/TLS error — check HTTPS proxy or antivirus scanning"
        text = str(reason).lower()
        if "timed out" in text or "timeout" in text:
            return "Connection timed out — check network or VPN"
        if "proxy" in text:
            return "Proxy error — verify HTTP/HTTPS proxy settings"
        if "getaddrinfo" in text or "name or service not known" in text:
            return "DNS lookup failed — check internet connection"
        return f"Network error: {reason}"
    if isinstance(exc, OSError):
        if exc.errno in (errno.ENOSPC, 28):  # type: ignore[name-defined]
            return "Disk full while writing the download"
        if exc.errno in (errno.EACCES, errno.EPERM, 13, 5):
            return "Permission denied while writing files"
    return str(exc)


def _download_file_with_retry(url: str, dest_path: str, log) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RAWviewer-Setup/1.0"})
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        disk_err = _check_disk_space(dest_path, _min_free_bytes_for_install())
        if disk_err:
            log(disk_err)
            return False
        try:
            log(f"Downloading (attempt {attempt}/{DOWNLOAD_RETRIES})...")
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
            return True
        except Exception as exc:
            log(f"Download failed: {_describe_download_error(exc)}")
            if attempt < DOWNLOAD_RETRIES:
                log(f"Retrying in {RETRY_DELAY_SEC}s...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                log(
                    "Tip: check firewall, VPN, or corporate proxy settings "
                    "and ensure GitHub is reachable."
                )
    return False


def _run_logged_subprocess(cmd, cwd, log, cancelled, progress_hook=None) -> int | None:
    """Run a subprocess, streaming stdout. Returns None if cancelled."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW,
        env=env,
    )
    try:
        for line in process.stdout:
            if cancelled():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return None
            msg = line.strip()
            if not msg:
                continue
            parsed = _parse_installer_progress_line(msg)
            if parsed is not None:
                pct, progress_msg = parsed
                if progress_hook is not None:
                    progress_hook(pct, progress_msg)
                continue
            # Skip leaked tqdm bar lines (progress uses @RAWVIEWER_PROGRESS only).
            if re.search(r"\d+%\|", msg) or re.search(r"\|\s*\d+%", msg):
                continue
            log(msg)
        process.wait()
        return process.returncode
    finally:
        if process.stdout:
            process.stdout.close()


def _run_subprocess_with_retry(cmd, cwd, log, label, cancelled, progress_hook=None) -> int | None:
    last_code = 1
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if cancelled():
            return None
        log(f"{label} (attempt {attempt}/{DOWNLOAD_RETRIES})...")
        code = _run_logged_subprocess(cmd, cwd, log, cancelled, progress_hook)
        if code is None:
            return None
        if code == 0:
            return 0
        last_code = code
        log(f"{label} exited with code {code}.")
        if attempt < DOWNLOAD_RETRIES:
            log(f"Retrying in {RETRY_DELAY_SEC}s...")
            time.sleep(RETRY_DELAY_SEC)
    log(
        f"{label} failed after {DOWNLOAD_RETRIES} attempts. "
        "Check internet, proxy/VPN, and antivirus; conda-forge and PyPI must be reachable."
    )
    return last_code


FILE_ASSOC_PROGID = "RAWviewer.Image"


def _file_association_extensions() -> list[str]:
    try:
        src_dir = os.path.join(BUNDLE_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from raw_file_extensions import get_supported_extensions

        return get_supported_extensions()
    except Exception:
        return [
            ".cr2",
            ".cr3",
            ".nef",
            ".arw",
            ".dng",
            ".orf",
            ".rw2",
            ".pef",
            ".srw",
            ".x3f",
            ".raf",
            ".3fr",
            ".fff",
            ".iiq",
            ".cap",
            ".erf",
            ".mef",
            ".mos",
            ".nrw",
            ".rwl",
            ".srf",
            ".jpeg",
            ".jpg",
            ".png",
            ".webp",
            ".heif",
            ".heic",
            ".tif",
            ".tiff",
        ]


def _register_file_associations(target_exe: str, log) -> None:
    """Register HKCU Open With entries (does not set default handler / UserChoice)."""
    if not os.path.isfile(target_exe):
        log("Skipping file associations: RAWviewer.exe not found.")
        return

    exe_path = os.path.normpath(target_exe)
    open_cmd = f'"{exe_path}" "%1"'
    extensions = _file_association_extensions()

    try:
        app_key = r"Software\Classes\Applications\RAWviewer.exe"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"{app_key}\\shell\\open\\command") as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, open_cmd)

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"{app_key}\\SupportedTypes") as key:
            for ext in extensions:
                winreg.SetValueEx(key, ext.lower(), 0, winreg.REG_SZ, "")

        progid_key = f"Software\\Classes\\{FILE_ASSOC_PROGID}"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, progid_key) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "RAWviewer Image")
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, f"{progid_key}\\shell\\open\\command"
        ) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, open_cmd)

        for ext in extensions:
            ext_key = ext.lower()
            if not ext_key.startswith("."):
                ext_key = f".{ext_key}"
            owp_key = f"Software\\Classes\\{ext_key}\\OpenWithProgids"
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, owp_key) as key:
                winreg.SetValueEx(key, FILE_ASSOC_PROGID, 0, winreg.REG_SZ, "")

        caps_key = r"Software\RAWviewer\Capabilities"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, caps_key) as key:
            winreg.SetValueEx(key, "ApplicationName", 0, winreg.REG_SZ, "RAWviewer")
            winreg.SetValueEx(
                key, "ApplicationDescription", 0, winreg.REG_SZ, "RAW and photo viewer"
            )
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"{caps_key}\\FileAssociations") as key:
            for ext in extensions:
                ext_key = ext.lower()
                if not ext_key.startswith("."):
                    ext_key = f".{ext_key}"
                winreg.SetValueEx(key, ext_key, 0, winreg.REG_SZ, "RAWviewer Image")

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\RegisteredApplications") as key:
            winreg.SetValueEx(key, "RAWviewer", 0, winreg.REG_SZ, r"Software\RAWviewer\Capabilities")

        import ctypes

        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
        log("Registered file associations (Open With).")
    except Exception as exc:
        log(f"Failed to register file associations: {exc}")


def _cleanup_partial_install(target_dir: str, log=None) -> None:
    if not target_dir:
        return
    if not os.path.isdir(target_dir):
        return
    try:
        shutil.rmtree(target_dir)
        if log:
            log("Removed incomplete installation folder.")
    except Exception as exc:
        if log:
            log(f"Could not fully remove incomplete installation: {exc}")


# IMPORTANT: Keep PyInstaller bundle dir
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    EXE_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EXE_DIR = BUNDLE_DIR

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar, QPlainTextEdit,
    QStackedWidget, QFileDialog, QFrame, QDialog, QRadioButton
)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

class InstallWorker(QObject):
    finished = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    progress_label_signal = pyqtSignal(str)

    def __init__(self, target_dir, install_mode):
        super().__init__()
        self.target_dir = os.path.normpath(target_dir)
        self.install_mode = install_mode
        self.cancelled = False

    def stop(self):
        self.cancelled = True

    def _is_cancelled(self) -> bool:
        return self.cancelled

    def _abort_cancelled(self) -> None:
        self.log_signal.emit("Installation cancelled.")
        _cleanup_partial_install(self.target_dir, self.log_signal.emit)
        self.finished.emit(False, "")

    def run(self):
        target_dir = self.target_dir
        try:
            min_bytes = MIN_FREE_BYTES_LITE if self.install_mode == "lite" else MIN_FREE_BYTES
            disk_err = _check_disk_space(target_dir, min_bytes)
            if disk_err:
                self.log_signal.emit(disk_err)
                self.finished.emit(False, "")
                return

            os.makedirs(target_dir, exist_ok=True)
            self.progress_signal.emit(2)
            self.log_signal.emit(f"Installing to {target_dir}...")

            if self._is_cancelled():
                self._abort_cancelled()
                return

            self.log_signal.emit("Copying core files...")
            # Copy the selected variant of pixi.toml to the target folder
            manifest_source = os.path.join(BUNDLE_DIR, f"pixi-{self.install_mode}.toml")
            if not os.path.isfile(manifest_source):
                manifest_source = os.path.join(BUNDLE_DIR, "pixi.toml")
            shutil.copy2(manifest_source, os.path.join(target_dir, "pixi.toml"))

            src_dir = os.path.join(target_dir, "src")
            if os.path.exists(src_dir):
                shutil.rmtree(src_dir)
            shutil.copytree(
                os.path.join(BUNDLE_DIR, "src"),
                src_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("logs", "*.log"),
            )

            # Dynamically configure build_profile.py based on install mode
            profile_val = "lite" if self.install_mode == "lite" else "full"
            build_profile_path = os.path.join(src_dir, "build_profile.py")
            try:
                with open(build_profile_path, "w", encoding="utf-8") as f:
                    f.write(f'PROFILE = "{profile_val}"\n')
                self.log_signal.emit(f"Configured build profile: {profile_val}")
            except Exception as exc:
                self.log_signal.emit(f"Warning: could not write build_profile.py: {exc}")

            assets_dir = os.path.join(target_dir, "icons")
            if os.path.exists(assets_dir):
                shutil.rmtree(assets_dir)
            shutil.copytree(os.path.join(BUNDLE_DIR, "icons"), assets_dir, dirs_exist_ok=True)

            scripts_dir = os.path.join(target_dir, "scripts")
            if os.path.exists(scripts_dir):
                shutil.rmtree(scripts_dir)
            if os.path.exists(os.path.join(BUNDLE_DIR, "scripts")):
                shutil.copytree(
                    os.path.join(BUNDLE_DIR, "scripts"),
                    scripts_dir,
                    dirs_exist_ok=True,
                )

            uninst_src = os.path.join(BUNDLE_DIR, "uninstall.bat")
            if os.path.exists(uninst_src):
                shutil.copy2(uninst_src, target_dir)

            target_exe = os.path.join(target_dir, "RAWviewer.exe")
            launcher_stub_src = os.path.join(BUNDLE_DIR, "RAWviewer.exe")
            if os.path.isfile(launcher_stub_src):
                shutil.copy2(launcher_stub_src, target_exe)
            else:
                self.log_signal.emit(
                    "Warning: launcher stub missing from installer; creating launcher.vbs fallback."
                )

            setup_exe = os.path.join(target_dir, "RAWviewer_Setup.exe")
            if getattr(sys, "frozen", False):
                shutil.copy2(sys.executable, setup_exe)

            self.progress_signal.emit(4)

            internal_dir = os.path.join(target_dir, "_internal")
            pixi_dir = os.path.join(internal_dir, "pixi")
            os.makedirs(pixi_dir, exist_ok=True)
            pixi_exe = os.path.join(pixi_dir, "pixi.exe")

            if not os.path.exists(pixi_exe):
                self.log_signal.emit("Downloading Pixi environment manager...")
                zip_path = os.path.join(pixi_dir, "pixi.zip")
                if not _download_file_with_retry(PIXI_DOWNLOAD_URL, zip_path, self.log_signal.emit):
                    _cleanup_partial_install(target_dir, self.log_signal.emit)
                    self.finished.emit(False, "")
                    return
                if self._is_cancelled():
                    self._abort_cancelled()
                    return
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(pixi_dir)
                    os.remove(zip_path)
                except (OSError, zipfile.BadZipFile) as exc:
                    self.log_signal.emit(f"Could not unpack Pixi: {_describe_download_error(exc)}")
                    _cleanup_partial_install(target_dir, self.log_signal.emit)
                    self.finished.emit(False, "")
                    return

            self.progress_signal.emit(5)

            self.log_signal.emit(
                "Downloading Python dependencies. This may take several minutes..."
            )
            pixi_code = _run_subprocess_with_retry(
                [pixi_exe, "install", "-v"],
                target_dir,
                self.log_signal.emit,
                "Dependency install",
                self._is_cancelled,
            )
            if pixi_code is None:
                self._abort_cancelled()
                return
            if pixi_code != 0:
                _cleanup_partial_install(target_dir, self.log_signal.emit)
                self.finished.emit(False, "")
                return

            self.progress_signal.emit(MODEL_DOWNLOAD_PROGRESS_START)

            success_note = ""
            if self.install_mode == "lite":
                self.log_signal.emit(
                    "Lite install: skipping AI models (no semantic search or face detection)."
                )
                self.progress_label_signal.emit("Finishing setup...")
                self.progress_signal.emit(100)
            else:
                self.log_signal.emit(
                    "Downloading AI models (~600 MB). This may take several minutes..."
                )
                self.progress_label_signal.emit("Downloading... 0%")
                self._model_download_log_pct = -1

                def _on_model_download_progress(pct: int, message: str) -> None:
                    self.progress_signal.emit(_map_model_download_progress(pct))
                    label = (message or "").strip() or f"Downloading... {pct}%"
                    self.progress_label_signal.emit(label)
                    last_logged = getattr(self, "_model_download_log_pct", -1)
                    if pct >= 100 or pct <= 0 or pct >= last_logged + 5:
                        self._model_download_log_pct = pct
                        self.log_signal.emit(label)

                models_code = _run_subprocess_with_retry(
                    [pixi_exe, "run", "python", "-u", "scripts/download_mobileclip_onnx.py"],
                    target_dir,
                    self.log_signal.emit,
                    "AI model download",
                    self._is_cancelled,
                    progress_hook=_on_model_download_progress,
                )
                if models_code is None:
                    self._abort_cancelled()
                    return

                if models_code != 0:
                    self.log_signal.emit(
                        "Warning: AI search models were not downloaded. "
                        "Photo browsing will work; open Search in the gallery to download them later."
                    )
                    success_note = (
                        "AI search models were not downloaded (network, proxy, or disk issue). "
                        "Browsing works normally — open Search in the gallery to download them later."
                    )

            if self._is_cancelled():
                self._abort_cancelled()
                return

            launcher_vbs_path = os.path.join(target_dir, "launcher.vbs")
            if not os.path.isfile(target_exe):
                self.log_signal.emit("Creating launcher fallback...")
                launcher_script = f'''Set oWS = WScript.CreateObject("WScript.Shell")
oWS.CurrentDirectory = "{target_dir}"
oWS.Run "{pixi_exe} run start-windowless", 0, False
'''
                with open(launcher_vbs_path, "w", encoding="utf-8") as f:
                    f.write(launcher_script)
                launch_target = launcher_vbs_path
            else:
                launch_target = target_exe

            self.log_signal.emit("Creating shortcuts...")
            try:
                desktop = os.path.join(os.environ["USERPROFILE"], "Desktop")
                programs = os.path.join(
                    os.environ["APPDATA"],
                    "Microsoft",
                    "Windows",
                    "Start Menu",
                    "Programs",
                )

                vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{os.path.join(desktop, 'RAWviewer.lnk')}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{launch_target}"
oLink.WorkingDirectory = "{target_dir}"
oLink.IconLocation = "{target_exe if os.path.isfile(target_exe) else setup_exe}"
oLink.Save

sLinkFile2 = "{os.path.join(programs, 'RAWviewer.lnk')}"
Set oLink2 = oWS.CreateShortcut(sLinkFile2)
oLink2.TargetPath = "{launch_target}"
oLink2.WorkingDirectory = "{target_dir}"
oLink2.IconLocation = "{target_exe if os.path.isfile(target_exe) else setup_exe}"
oLink2.Save
"""
                vbs_path = os.path.join(target_dir, "create_shortcuts.vbs")
                with open(vbs_path, "w", encoding="utf-8") as f:
                    f.write(vbs_script)

                subprocess.run(
                    ["cscript.exe", "//Nologo", vbs_path],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                os.remove(vbs_path)
            except Exception as e:
                self.log_signal.emit(f"Could not create shortcut: {e}")

            self.log_signal.emit("Registering uninstaller...")
            try:
                import ctypes

                key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\RAWviewer"
                icon_path = target_exe if os.path.isfile(target_exe) else setup_exe
                install_date = time.strftime("%Y%m%d")
                uninst_path = os.path.join(target_dir, "uninstall.bat")
                silent_cmd = f'"{uninst_path}"'
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "RAWviewer")
                    winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, icon_path)
                    winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, "2.3.2")
                    winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "QuietUninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, target_dir)
                    winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Mark Yip")
                    winreg.SetValueEx(key, "InstallDate", 0, winreg.REG_SZ, install_date)
                    winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "WindowsInstaller", 0, winreg.REG_DWORD, 0)
                result = ctypes.wintypes.DWORD()
                ctypes.windll.user32.SendMessageTimeoutW(
                    0xFFFF, 0x001A, 0, "Environment", 0x0002, 1000, ctypes.byref(result)
                )
            except Exception as e:
                self.log_signal.emit(f"Failed to register uninstaller: {e}")

            self.log_signal.emit("Registering file associations...")
            _register_file_associations(target_exe, self.log_signal.emit)

            self.progress_signal.emit(100)
            self.finished.emit(True, success_note)

        except Exception as e:
            self.log_signal.emit(f"Error: {e}")
            _cleanup_partial_install(target_dir, self.log_signal.emit)
            self.finished.emit(False, "")

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
        self.setFixedSize(650, 500)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._install_in_progress = False
        self._awaiting_cancel = False
        self.worker = None
        self.thread = None
        self._close_after_failed_install = False

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
            QRadioButton {
                color: #ffffff;
                font-family: 'Segoe UI', Arial;
                font-size: 13px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
        """)

    def init_welcome_page(self):
        layout = QVBoxLayout(self.page_welcome)
        layout.setContentsMargins(40, 20, 40, 20)
        
        title = QLabel("Ready to Install")
        title.setObjectName("title")
        layout.addWidget(title)
        
        desc_text = (
            "RAWviewer helps you review and cull RAW and JPEG photos quickly.\n\n"
            "Choose your installation profile and acceleration backend below. "
            "Full versions require internet access to download dependencies and models."
        )
        desc = QLabel(desc_text)
        desc.setObjectName("desc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        options_label = QLabel("Installation Options:")
        options_label.setStyleSheet("color: #888; font-weight: bold; margin-top: 15px;")
        layout.addWidget(options_label)

        self.radio_cuda = QRadioButton("Full — NVIDIA GPU (CUDA Acceleration, downloads ~600MB models)")
        self.radio_directml = QRadioButton("Full — Default GPU (DirectML Acceleration, downloads ~600MB models)")
        self.radio_lite = QRadioButton("Lite — High Performance (No AI search, no downloading models, saves ~3GB disk space)")

        # Set default to DirectML (most compatible across all devices)
        self.radio_directml.setChecked(True)

        options_layout = QVBoxLayout()
        options_layout.setSpacing(6)
        options_layout.addWidget(self.radio_cuda)
        options_layout.addWidget(self.radio_directml)
        options_layout.addWidget(self.radio_lite)
        layout.addLayout(options_layout)
        
        path_label = QLabel("Installation Directory:")
        path_label.setStyleSheet("color: #888; font-weight: bold; margin-top: 15px;")
        layout.addWidget(path_label)
        
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        app_data = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
        default_path = os.path.join(app_data, "RAWviewer")
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

        self.install_step_label = QLabel("")
        self.install_step_label.setStyleSheet("color: #A0A0A0; font-size: 13px;")
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
        title.setStyleSheet("font-size: 40px; color: #4ade80;")
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
        self.worker = InstallWorker(target_dir, mode)
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
            try:
                self.btn_next.clicked.disconnect()
            except TypeError:
                pass
            self.btn_next.clicked.connect(self.launch_app)
            self.btn_cancel.hide()
        else:
            self.btn_next.setText("RETRY")
            self.btn_next.setEnabled(True)
            try:
                self.btn_next.clicked.disconnect()
            except TypeError:
                pass
            self.btn_next.clicked.connect(self.retry_install)

        if not success and close_after_failure:
            self.close()

    def retry_install(self):
        self.stack.setCurrentIndex(0)
        self.btn_next.setText("INSTALL")
        self.btn_cancel.setText("CANCEL")
        self.btn_cancel.show()
        try:
            self.btn_next.clicked.disconnect()
        except TypeError:
            pass
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
        if os.path.isfile(app_exe):
            subprocess.Popen([app_exe], cwd=install_dir, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            launcher_vbs = os.path.join(install_dir, "launcher.vbs")
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
