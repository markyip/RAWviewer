"""Module-level helper functions for the bootstrap installer."""
from __future__ import annotations

import errno
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import winreg

from .constants import (
    BUNDLE_DIR,
    DOWNLOAD_RETRIES,
    EXE_DIR,
    FILE_ASSOC_PROGID,
    INSTALLER_PROGRESS_PREFIX,
    MIN_FREE_BYTES,
    MIN_FREE_BYTES_LITE,
    MODEL_DOWNLOAD_PROGRESS_END,
    MODEL_DOWNLOAD_PROGRESS_START,
    RETRY_DELAY_SEC,
)


def _is_lite_installer() -> bool:
    try:
        from core.build_profile import PROFILE

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
        if exc.errno in (errno.ENOSPC, 28):
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
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
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


def _file_association_extensions() -> list[str]:
    try:
        src_dir = os.path.join(BUNDLE_DIR, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from raw_adjustments.file_extensions import get_supported_extensions

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

import contextlib

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication


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
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/IM", "RAWviewer.exe", "/T"],
            capture_output=True,
            creationflags=flags,
            timeout=15,
        )
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


