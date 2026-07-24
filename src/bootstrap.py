import errno
import hashlib
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

from app_version import APP_VERSION

# Pin the Windows installer pixi binary so a future upstream release cannot
# break installs mid-flight (do not use .../releases/latest/...).
PIXI_VERSION = "0.73.0"
PIXI_DOWNLOAD_URL = (
    f"https://github.com/prefix-dev/pixi/releases/download/v{PIXI_VERSION}/"
    "pixi-x86_64-pc-windows-msvc.zip"
)
# Pinned integrity hash for the asset above (pixi-x86_64-pc-windows-msvc.zip).
# MUST be updated whenever PIXI_VERSION is bumped; compute with `shasum -a 256`.
PIXI_DOWNLOAD_SHA256 = "d9044186bfea9771b8e35b0ed032352b557a82528cd5165db6b8ad137c7a873c"
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SEC = 3
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # ~2 GiB for pixi env + models (Plus)
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


def _directory_size_bytes(root: str) -> int:
    """Sum on-disk file sizes under ``root`` (skips unreadable entries)."""
    total = 0
    if not root or not os.path.isdir(root):
        return 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def _estimated_size_kb(root: str) -> int:
    """Windows Apps list ``EstimatedSize`` is a DWORD of kilobytes."""
    return max(1, int(_directory_size_bytes(root) // 1024))


def _prune_torch_link_libs(target_dir: str, log) -> None:
    """Remove PyTorch ``*.lib`` files that are only needed for compiling extensions.

    The cu124 wheel ships ~0.7–0.8 GB of static import libraries under
    ``torch/lib`` (e.g. ``dnnl.lib``). RAWviewer never links against them at
    runtime; deleting them shrinks CUDA Full installs without changing demosaic.
    """
    site = os.path.join(
        target_dir, ".pixi", "envs", "default", "Lib", "site-packages", "torch", "lib"
    )
    if not os.path.isdir(site):
        return
    removed = 0
    freed = 0
    try:
        for name in os.listdir(site):
            if not name.lower().endswith(".lib"):
                continue
            path = os.path.join(site, name)
            try:
                freed += os.path.getsize(path)
                os.remove(path)
                removed += 1
            except OSError as exc:
                log(f"Could not remove {name}: {exc}")
    except OSError as exc:
        log(f"Torch lib prune skipped: {exc}")
        return
    if removed:
        log(
            f"Pruned {removed} torch link libraries ({_human_bytes(freed)}) "
            "— not required at runtime."
        )


def _patch_python_exe_branding(target_dir: str, log) -> None:
    """Rebrand the pixi-installed python.exe/pythonw.exe so Task Manager shows
    "RAWviewer" instead of "Python".

    RAWviewer.exe (the launcher stub) spawns ``pixi run pythonw src/main.py``
    and exits immediately -- the long-running process Windows actually tracks
    is that unmodified interpreter binary. Task Manager's "Name" column reads
    the exe's embedded FileDescription resource, not the filename or the
    script it's running, so renaming/copying the file changes nothing; only
    patching that resource does. Uses rcedit (bundled under
    scripts/tools/rcedit-x64.exe, MIT-licensed, https://github.com/electron/rcedit)
    on the two interpreter binaries pixi just installed. Best-effort: a
    missing rcedit or a patch failure only affects this cosmetic label, never
    blocks the install.
    """
    rcedit = os.path.join(BUNDLE_DIR, "scripts", "tools", "rcedit-x64.exe")
    if not os.path.isfile(rcedit):
        return
    icon_path = os.path.join(BUNDLE_DIR, "icons", "appicon.ico")
    env_dir = os.path.join(target_dir, ".pixi", "envs", "default")
    patched = 0
    for exe_name in ("python.exe", "pythonw.exe"):
        exe_path = os.path.join(env_dir, exe_name)
        if not os.path.isfile(exe_path):
            continue
        cmd = [
            rcedit, exe_path,
            "--set-version-string", "FileDescription", "RAWviewer",
            "--set-version-string", "ProductName", "RAWviewer",
            "--set-version-string", "InternalName", "RAWviewer",
            "--set-version-string", "OriginalFilename", exe_name,
        ]
        if os.path.isfile(icon_path):
            cmd.extend(["--set-icon", icon_path])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                patched += 1
            else:
                log(
                    f"Could not rebrand {exe_name} for Task Manager: "
                    f"{result.stderr.decode(errors='replace').strip() or result.returncode}"
                )
        except Exception as exc:
            log(f"Could not rebrand {exe_name} for Task Manager: {exc}")
    if patched:
        log(f"Rebranded {patched} interpreter executable(s) for Task Manager display.")


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


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file_with_retry(url: str, dest_path: str, log) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "RAWviewer-Setup/1.0"})
    tmp_path = dest_path + ".part"
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        disk_err = _check_disk_space(dest_path, _min_free_bytes_for_install())
        if disk_err:
            log(disk_err)
            return False
        try:
            log(f"Downloading (attempt {attempt}/{DOWNLOAD_RETRIES})...")
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
            os.replace(tmp_path, dest_path)
            return True
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
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

def _installer_icon_path() -> str | None:
    """Resolve branded icon from PyInstaller bundle or repo icons/."""
    for rel in (
        os.path.join("icons", "appicon.ico"),
        os.path.join("icons", "appicon.png"),
    ):
        for base in (BUNDLE_DIR, EXE_DIR):
            path = os.path.join(base, rel)
            if os.path.isfile(path):
                return path
    return None


def _installed_app_icon_path(install_dir: str) -> str | None:
    ico = os.path.join(install_dir, "icons", "appicon.ico")
    if os.path.isfile(ico):
        return ico
    return _installer_icon_path()

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QProgressBar, QPlainTextEdit,
    QStackedWidget, QFileDialog, QFrame, QDialog, QRadioButton, QCheckBox
)
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread


def _apply_installer_process_icon() -> None:
    """Taskbar / Alt+Tab icon for the setup wizard (Windows needs explicit AppUserModelID)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "com.markyip.rawviewer.setup"
            )
        except Exception:
            pass
    path = _installer_icon_path()
    if not path:
        return
    icon = QIcon(path)
    if icon.isNull():
        return
    app = QApplication.instance()
    if app is not None:
        app.setWindowIcon(icon)

def _user_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".rawviewer_cache")


def _clear_existing_user_cache(log) -> None:
    """Wipe photo cache + session so upgrades unlock perf-v2 defaults.

    Does not delete photos, XMP sidecars, or the install directory.
    """
    removed_any = False

    def _rm_tree(path: str, label: str) -> None:
        nonlocal removed_any
        if not os.path.exists(path):
            return
        log(f"Removing {label}: {path}")
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            if not os.path.exists(path):
                removed_any = True
            else:
                log(f"  Warning: could not fully remove {path}")
        except OSError as exc:
            log(f"  Warning: could not remove {path}: {exc}")

    log("Clearing existing cache (upgrade / performance reset)...")
    _rm_tree(_user_cache_dir(), "image/EXIF/semantic/thumbnail cache")
    local = os.environ.get("LOCALAPPDATA") or ""
    roaming = os.environ.get("APPDATA") or ""
    if local:
        _rm_tree(os.path.join(local, "RAWviewer", "logs"), "runtime logs")
        _rm_tree(os.path.join(local, "RAWviewer", "cache"), "install cache")
        _rm_tree(os.path.join(local, "RAWviewer", "CrashDumps"), "crash dumps")
    if roaming:
        _rm_tree(os.path.join(roaming, "RAWviewer", "logs"), "roaming logs")
    reg_path = r"Software\RAWviewer"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path):
            pass
        _delete_reg_tree(winreg.HKEY_CURRENT_USER, reg_path)
        removed_any = True
        log("Removed HKCU\\Software\\RAWviewer session settings")
    except FileNotFoundError:
        pass
    except OSError as exc:
        log(f"  Warning: could not clear registry settings: {exc}")
    if not removed_any:
        log("No existing cache found — nothing to clear.")
    else:
        log("Cache cleared. First launch after install may rebuild thumbnails/index.")


def _kill_running_app_processes(target_dir: str, log) -> None:
    """Terminate running RAWviewer processes before touching the install dir.

    A live pixi.exe / pythonw.exe from a previous launch holds open handles
    inside the install tree, which made upgrades fail partway (WinError 183
    on the _launcher copytree after rmtree silently failed, then access
    denied on .pixi DLLs during cleanup). Kill by NAME only for RAWviewer.exe
    (unambiguous); pixi/python/pythonw are killed strictly by executable
    PATH under the install dir so unrelated Python apps are never touched.
    The Setup process itself has a different image name and lives outside
    target_dir, so it is never a casualty.
    """
    log("Stopping running RAWviewer instances...")
    flags = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "RAWviewer.exe", "/T"],
            capture_output=True,
            creationflags=flags,
            timeout=15,
        )
    except Exception:
        pass
    # PowerShell path-scoped kill: single quotes in the dir are doubled for
    # PS single-quoted string escaping.
    ps_dir = os.path.normpath(target_dir).rstrip("\\").replace("'", "''")
    ps_cmd = (
        "Get-Process pixi,python,pythonw -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.Path -like '{ps_dir}\\*' }} | "
        "Stop-Process -Force -ErrorAction SilentlyContinue"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            creationflags=flags,
            timeout=30,
        )
    except Exception as exc:
        log(f"  Warning: could not scan for running processes: {exc}")
    # Give the OS a moment to release file handles from the killed processes.
    time.sleep(1.0)


def _delete_reg_tree(root, path: str) -> None:
    """Recursively delete a registry key (winreg.DeleteKey needs empty keys)."""
    try:
        with winreg.OpenKey(root, path, 0, winreg.KEY_ALL_ACCESS) as key:
            while True:
                try:
                    sub = winreg.EnumKey(key, 0)
                except OSError:
                    break
                _delete_reg_tree(root, f"{path}\\{sub}")
        winreg.DeleteKey(root, path)
    except FileNotFoundError:
        pass


class InstallWorker(QObject):
    finished = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    progress_label_signal = pyqtSignal(str)

    def __init__(self, target_dir, install_mode, clear_cache: bool = False):
        super().__init__()
        self.target_dir = os.path.normpath(target_dir)
        self.install_mode = install_mode
        self.clear_cache = bool(clear_cache)
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
            # Kill anything still running out of the install dir FIRST --
            # open handles from a live pixi/pythonw break the file copies
            # below (WinError 183 / access denied on upgrade installs).
            _kill_running_app_processes(target_dir, self.log_signal.emit)
            if self._is_cancelled():
                self._abort_cancelled()
                return

            if self.clear_cache:
                _clear_existing_user_cache(self.log_signal.emit)
                self.progress_signal.emit(1)
                if self._is_cancelled():
                    self._abort_cancelled()
                    return

            # Full (DirectML / CUDA+CuPy) needs headroom for pixi + ~700 MB models;
            # CUDA no longer downloads a multi-GB torch wheel.
            min_bytes = MIN_FREE_BYTES_LITE if self.install_mode == "lite" else MIN_FREE_BYTES
            effective_mode = self.install_mode

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
            manifest_name = f"pixi-{effective_mode}.toml"
            manifest_source = os.path.join(BUNDLE_DIR, manifest_name)
            if not os.path.isfile(manifest_source):
                manifest_source = os.path.join(
                    BUNDLE_DIR, f"pixi-{self.install_mode}.toml"
                )
            if not os.path.isfile(manifest_source):
                manifest_source = os.path.join(BUNDLE_DIR, "pixi.toml")
            shutil.copy2(manifest_source, os.path.join(target_dir, "pixi.toml"))

            # Always clear a previous Pixi env on reinstall: leftover torch+cu124
            # (~2.4 GB+) or a mismatched CuPy/DirectML stack must not linger.
            pixi_env_root = os.path.join(target_dir, ".pixi")
            if os.path.isdir(pixi_env_root):
                self.log_signal.emit(
                    f"{effective_mode} install: clearing previous Pixi environment..."
                )
                shutil.rmtree(pixi_env_root, ignore_errors=True)

            # Drop stale BYO config from older installers (external torch binding).
            for stale in ("torch_provider.json",):
                stale_path = os.path.join(target_dir, stale)
                if os.path.isfile(stale_path):
                    try:
                        os.remove(stale_path)
                        self.log_signal.emit(f"Removed obsolete {stale}")
                    except OSError:
                        pass

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

            # Easy-to-find cache wipe next to RAWviewer.exe / uninstall.bat
            clear_src = os.path.join(BUNDLE_DIR, "clear_cache.bat")
            if not os.path.isfile(clear_src):
                clear_src = os.path.join(
                    BUNDLE_DIR, "scripts", "Launch", "windows", "clear_cache_user.bat"
                )
            if os.path.isfile(clear_src):
                shutil.copy2(clear_src, os.path.join(target_dir, "clear_cache.bat"))

            # Debug-log launcher for freeze / gallery troubleshooting
            debug_log_src = os.path.join(BUNDLE_DIR, "run_with_debug_log.bat")
            if not os.path.isfile(debug_log_src):
                debug_log_src = os.path.join(
                    BUNDLE_DIR, "scripts", "Launch", "windows", "run_with_debug_log.bat"
                )
            if os.path.isfile(debug_log_src):
                shutil.copy2(
                    debug_log_src, os.path.join(target_dir, "run_with_debug_log.bat")
                )

            target_exe = os.path.join(target_dir, "RAWviewer.exe")
            launcher_stub_src = os.path.join(BUNDLE_DIR, "RAWviewer.exe")
            launcher_runtime_src = os.path.join(BUNDLE_DIR, "_launcher")
            if os.path.isfile(launcher_stub_src):
                shutil.copy2(launcher_stub_src, target_exe)
                # onedir stub runtime (avoids PyInstaller _MEI* cleanup MessageBox)
                if os.path.isdir(launcher_runtime_src):
                    launcher_runtime_dst = os.path.join(target_dir, "_launcher")
                    if os.path.isdir(launcher_runtime_dst):
                        shutil.rmtree(launcher_runtime_dst, ignore_errors=True)
                    # dirs_exist_ok: if the rmtree above silently failed on a
                    # still-held handle, overwrite in place instead of dying
                    # with WinError 183 ("file already exists").
                    shutil.copytree(
                        launcher_runtime_src, launcher_runtime_dst, dirs_exist_ok=True
                    )
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
                self.log_signal.emit(
                    f"Downloading Pixi environment manager (v{PIXI_VERSION})..."
                )
                zip_path = os.path.join(pixi_dir, "pixi.zip")
                if not _download_file_with_retry(PIXI_DOWNLOAD_URL, zip_path, self.log_signal.emit):
                    _cleanup_partial_install(target_dir, self.log_signal.emit)
                    self.finished.emit(False, "")
                    return
                if _sha256_of_file(zip_path).lower() != PIXI_DOWNLOAD_SHA256.lower():
                    self.log_signal.emit(
                        "Pixi download failed integrity verification (SHA-256 mismatch); aborting."
                    )
                    _cleanup_partial_install(target_dir, self.log_signal.emit)
                    self.finished.emit(False, "")
                    return
                if self._is_cancelled():
                    self._abort_cancelled()
                    return
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        root = os.path.realpath(pixi_dir)
                        for info in zf.infolist():
                            name = info.filename.replace("\\", "/")
                            if name.startswith("/") or ".." in name.split("/"):
                                raise zipfile.BadZipFile(
                                    f"refusing unsafe zip member: {info.filename!r}"
                                )
                            dest = os.path.realpath(os.path.join(pixi_dir, name))
                            if dest != root and not dest.startswith(root + os.sep):
                                raise zipfile.BadZipFile(
                                    f"refusing zip path escape: {info.filename!r}"
                                )
                        zf.extractall(pixi_dir)
                    os.remove(zip_path)
                except (OSError, zipfile.BadZipFile) as exc:
                    self.log_signal.emit(f"Could not unpack Pixi: {_describe_download_error(exc)}")
                    _cleanup_partial_install(target_dir, self.log_signal.emit)
                    self.finished.emit(False, "")
                    return

            self.progress_signal.emit(5)

            if effective_mode == "cuda":
                self.log_signal.emit(
                    "Downloading Python dependencies (CuPy CUDA demosaic, no PyTorch)..."
                )
            else:
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

            # Safety: if a leftover torch wheel somehow remains, drop link-time *.lib.
            _prune_torch_link_libs(target_dir, self.log_signal.emit)

            _patch_python_exe_branding(target_dir, self.log_signal.emit)

            self.progress_signal.emit(MODEL_DOWNLOAD_PROGRESS_START)

            success_note = ""
            if self.install_mode == "lite":
                self.log_signal.emit(
                    "Standard install: skipping AI models (no semantic search or face detection)."
                )
                self.progress_label_signal.emit("Finishing setup...")
                self.progress_signal.emit(100)
            else:
                self.log_signal.emit(
                    "Downloading AI models (~700 MB). This may take several minutes..."
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
                    [pixi_exe, "run", "python", "-u", "scripts/models/download_mobileclip_onnx.py"],
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
                installed_icon = _installed_app_icon_path(target_dir)
                icon_for_shortcuts = (
                    installed_icon
                    if installed_icon
                    else (target_exe if os.path.isfile(target_exe) else setup_exe)
                )

                vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{os.path.join(desktop, 'RAWviewer.lnk')}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{launch_target}"
oLink.WorkingDirectory = "{target_dir}"
oLink.IconLocation = "{icon_for_shortcuts}"
oLink.Save

sLinkFile2 = "{os.path.join(programs, 'RAWviewer.lnk')}"
Set oLink2 = oWS.CreateShortcut(sLinkFile2)
oLink2.TargetPath = "{launch_target}"
oLink2.WorkingDirectory = "{target_dir}"
oLink2.IconLocation = "{icon_for_shortcuts}"
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
                icon_path = _installed_app_icon_path(target_dir) or (
                    target_exe if os.path.isfile(target_exe) else setup_exe
                )
                install_date = time.strftime("%Y%m%d")
                uninst_path = os.path.join(target_dir, "uninstall.bat")
                silent_cmd = f'"{uninst_path}"'
                # Settings → Apps shows blank size without EstimatedSize (KB).
                est_kb = _estimated_size_kb(target_dir)
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "RAWviewer")
                    winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, icon_path)
                    winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, APP_VERSION)
                    winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "QuietUninstallString", 0, winreg.REG_SZ, silent_cmd)
                    winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, target_dir)
                    winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Mark Yip")
                    winreg.SetValueEx(key, "InstallDate", 0, winreg.REG_SZ, install_date)
                    winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, est_kb)
                    winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
                    winreg.SetValueEx(key, "WindowsInstaller", 0, winreg.REG_DWORD, 0)
                self.log_signal.emit(
                    f"Registered uninstaller (EstimatedSize={_human_bytes(est_kb * 1024)})."
                )
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

# Darkroom palette — keep in sync with src/theme.py (installer exe is built
# standalone, so the module is not imported here).
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
            "Standard — Browse + Basic Adjustment; CPU RAW processing"
        )
        self.radio_directml = QRadioButton(
            "Plus — DirectML (AI search + denoise; CPU RAW processing)"
        )
        self.radio_cuda = QRadioButton(
            "Plus — NVIDIA CUDA (AI search + denoise; GPU RAW processing)"
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
        if path: self.path_edit.setText(os.path.normpath(path))

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
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(400, self.close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_pos'):
            self.move(event.globalPosition().toPoint() - self.drag_pos)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    _apply_installer_process_icon()
    window = InstallerGUI()
    window.show()
    sys.exit(app.exec())
