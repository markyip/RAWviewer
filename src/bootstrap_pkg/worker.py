"""InstallWorker: background installation logic."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import winreg
import zipfile

from PyQt6.QtCore import QObject, pyqtSignal

from core.app_version import APP_VERSION

from .constants import (
    BUNDLE_DIR,
    MIN_FREE_BYTES,
    MIN_FREE_BYTES_LITE,
    MODEL_DOWNLOAD_PROGRESS_START,
    PIXI_DOWNLOAD_URL,
    PIXI_VERSION,
)
from .helpers import (
    _check_disk_space,
    _cleanup_partial_install,
    _clear_existing_user_cache,
    _describe_download_error,
    _download_file_with_retry,
    _estimated_size_kb,
    _human_bytes,
    _installed_app_icon_path,
    _kill_running_app_processes,
    _map_model_download_progress,
    _patch_python_exe_branding,
    _prune_torch_link_libs,
    _register_file_associations,
    _run_subprocess_with_retry,
)


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

            # Full (DirectML / CUDA+CuPy) needs headroom for pixi + ~600 MB models;
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


