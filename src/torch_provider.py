"""Discover / validate / bind an external or bundled PyTorch CUDA 12.x provider.

Installer preference (CUDA Full):
  1. Probe the machine for an existing torch+cu12x (conda, py launcher, PATH).
  2. Only if none is usable, download torch into the RAWviewer Pixi env.

Runtime:
  - ``apply_at_startup()`` prepends an external site-packages to ``sys.path``.
  - If that provider disappears, ``ensure_torch_for_gpu()`` notifies the user
    and installs a bundled cu12x wheel into the local Pixi environment.
"""

from __future__ import annotations

import json
import logging
import os
import site
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_CONFIG_NAME = "torch_provider.json"
_PROBE_TIMEOUT_S = 45
_MIN_TORCH_MAJOR = 2
_MAX_TORCH_MAJOR = 2  # stay on torch 2.x with the app's kornia pin


@dataclass(frozen=True)
class TorchProvider:
    mode: str  # "external" | "bundled" | "none"
    site_packages: str = ""
    python_exe: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    validated_at: str = ""

    @property
    def is_external(self) -> bool:
        return self.mode == "external" and bool(self.site_packages)

    @property
    def is_bundled(self) -> bool:
        return self.mode == "bundled"


def config_path(install_dir: Optional[str] = None) -> str:
    root = install_dir or _default_install_dir()
    return os.path.join(root, _CONFIG_NAME)


def _default_install_dir() -> str:
    # Prefer the running app tree (pixi / frozen adjacent).
    here = os.path.dirname(os.path.abspath(__file__))
    # src/ -> repo or install root
    parent = os.path.dirname(here)
    if os.path.isdir(os.path.join(parent, ".pixi")) or os.path.isfile(
        os.path.join(parent, "pixi.toml")
    ):
        return parent
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(local, "RAWviewer")


def load_provider(install_dir: Optional[str] = None) -> TorchProvider:
    path = config_path(install_dir)
    if not os.path.isfile(path):
        return TorchProvider(mode="none")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return TorchProvider(
            mode=str(data.get("mode") or "none"),
            site_packages=str(data.get("site_packages") or ""),
            python_exe=str(data.get("python_exe") or ""),
            torch_version=str(data.get("torch_version") or ""),
            cuda_version=str(data.get("cuda_version") or ""),
            validated_at=str(data.get("validated_at") or ""),
        )
    except Exception as exc:
        logger.warning("[TORCH] Failed to read %s: %s", path, exc)
        return TorchProvider(mode="none")


def save_provider(provider: TorchProvider, install_dir: Optional[str] = None) -> str:
    path = config_path(install_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = asdict(provider)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def apply_provider_to_sys_path(provider: Optional[TorchProvider] = None) -> bool:
    """Prepend external site-packages so ``import torch`` resolves there."""
    prov = provider if provider is not None else load_provider()
    if not prov.is_external:
        return False
    sp = os.path.normpath(prov.site_packages)
    if not os.path.isdir(sp):
        logger.warning("[TORCH] Configured site-packages missing: %s", sp)
        return False
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)
    # Also expose as env for child processes / process pool.
    os.environ["RAWVIEWER_TORCH_SITE_PACKAGES"] = sp
    try:
        site.addsitedir(sp)
    except Exception:
        pass
    logger.info(
        "[TORCH] Using external provider %s (CUDA %s) from %s",
        prov.torch_version or "?",
        prov.cuda_version or "?",
        sp,
    )
    return True


def apply_at_startup() -> TorchProvider:
    """Call very early from main.py (before torch_bootstrap)."""
    env_sp = (os.environ.get("RAWVIEWER_TORCH_SITE_PACKAGES") or "").strip()
    if env_sp and os.path.isdir(env_sp):
        if env_sp not in sys.path:
            sys.path.insert(0, env_sp)
        try:
            site.addsitedir(env_sp)
        except Exception:
            pass
    prov = load_provider()
    apply_provider_to_sys_path(prov)
    return prov


_PROBE_SCRIPT = r"""
import json, os, sys
out = {"ok": False, "error": "unknown"}
try:
    import torch
    ver = str(getattr(torch, "__version__", "") or "")
    cuda_rt = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_rt = str(cuda_rt) if cuda_rt else ""
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if not cuda_rt.startswith("12"):
        raise RuntimeError(f"CUDA runtime {cuda_rt!r} is not 12.x")
    major = int(ver.split(".", 1)[0].split("+", 1)[0])
    if major < 2 or major > 2:
        raise RuntimeError(f"torch {ver!r} major not supported (need 2.x)")
    # Prefer cu12 wheels; accept plain version when cuda runtime is 12.x.
    tag_ok = ("cu12" in ver.lower()) or ("+cu12" in ver.lower()) or bool(cuda_rt)
    if not tag_ok:
        raise RuntimeError(f"torch build {ver!r} does not look like cu12")
    site_packages = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    out = {
        "ok": True,
        "torch_version": ver,
        "cuda_version": cuda_rt,
        "site_packages": site_packages,
        "python_exe": sys.executable,
    }
except Exception as e:
    out = {"ok": False, "error": str(e)}
print(json.dumps(out))
"""


def _run_probe(python_exe: str) -> Optional[dict]:
    if not python_exe or not os.path.isfile(python_exe):
        return None
    try:
        proc = subprocess.run(
            [python_exe, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
    except Exception as exc:
        logger.debug("[TORCH] probe failed for %s: %s", python_exe, exc)
        return None
    if proc.returncode != 0:
        return None
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        return None
    try:
        data = json.loads(line[-1])
    except json.JSONDecodeError:
        return None
    if not data.get("ok"):
        return None
    return data


def _candidate_pythons() -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            return
        if os.path.isfile(path):
            seen.add(norm)
            found.append(path)

    # Explicit override first.
    env_py = (os.environ.get("RAWVIEWER_TORCH_PYTHON") or "").strip()
    _add(env_py)

    # py launcher (Windows)
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["py", "-0p"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in (proc.stdout or "").splitlines():
                # formats: "-V:3.12 * C:\...\python.exe" or bare path
                part = line.strip().split()[-1] if line.strip() else ""
                if part.lower().endswith("python.exe"):
                    _add(part)
        except Exception:
            pass

    for name in ("python", "python3"):
        try:
            proc = subprocess.run(
                ["where" if sys.platform == "win32" else "which", name],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0,
            )
            for line in (proc.stdout or "").splitlines():
                _add(line.strip())
        except Exception:
            pass

    local = os.environ.get("LOCALAPPDATA") or ""
    user = os.environ.get("USERPROFILE") or ""
    program_files = os.environ.get("ProgramFiles") or r"C:\Program Files"
    guesses = [
        os.path.join(user, "miniconda3", "python.exe"),
        os.path.join(user, "anaconda3", "python.exe"),
        os.path.join(user, "mambaforge", "python.exe"),
        os.path.join(user, "miniforge3", "python.exe"),
        r"C:\ProgramData\miniconda3\python.exe",
        r"C:\ProgramData\anaconda3\python.exe",
        os.path.join(program_files, "Python312", "python.exe"),
        os.path.join(program_files, "Python311", "python.exe"),
        os.path.join(local, "Programs", "Python", "Python312", "python.exe"),
        os.path.join(local, "Programs", "Python", "Python311", "python.exe"),
    ]
    # Conda envs
    for base in (
        os.path.join(user, "miniconda3", "envs"),
        os.path.join(user, "anaconda3", "envs"),
        os.path.join(user, "mambaforge", "envs"),
        r"C:\ProgramData\miniconda3\envs",
    ):
        if os.path.isdir(base):
            try:
                for name in os.listdir(base):
                    guesses.append(os.path.join(base, name, "python.exe"))
            except OSError:
                pass

    for g in guesses:
        _add(g)

    # Never treat the RAWviewer pixi env as "external" during discovery —
    # that would circularly prefer a half-installed local torch.
    install = _default_install_dir()
    pixi_py = os.path.normcase(
        os.path.join(install, ".pixi", "envs", "default", "python.exe")
    )
    # Skip Windows Store aliases (they are stubs that open the Store).
    filtered = []
    for p in found:
        if os.path.normcase(os.path.abspath(p)) == pixi_py:
            continue
        if "windowsapps" in os.path.normcase(p):
            continue
        filtered.append(p)
    return filtered


def discover_external_torch(
    *,
    log: Optional[Callable[[str], None]] = None,
    pythons: Optional[Iterable[str]] = None,
) -> Optional[TorchProvider]:
    """Return the first usable torch+cu12x provider on this machine."""

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            logger.info("%s", msg)

    _log("Checking for an existing PyTorch CUDA 12.x install (before any download)...")
    candidates = list(pythons) if pythons is not None else _candidate_pythons()
    if not candidates:
        _log("No candidate Python interpreters found.")
        return None

    for py in candidates:
        _log(f"  Probing: {py}")
        data = _run_probe(py)
        if not data:
            continue
        sp = str(data.get("site_packages") or "")
        if not sp or not os.path.isdir(sp):
            continue
        provider = TorchProvider(
            mode="external",
            site_packages=os.path.normpath(sp),
            python_exe=os.path.normpath(str(data.get("python_exe") or py)),
            torch_version=str(data.get("torch_version") or ""),
            cuda_version=str(data.get("cuda_version") or ""),
            validated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        _log(
            f"  Found usable PyTorch {provider.torch_version} "
            f"(CUDA {provider.cuda_version}) — will skip torch download."
        )
        return provider

    _log("No suitable external PyTorch CUDA 12.x found.")
    return None


def validate_provider(provider: TorchProvider) -> tuple[bool, str]:
    """Re-validate a saved provider (external via its python, or import here)."""
    if provider.is_external:
        if provider.python_exe and os.path.isfile(provider.python_exe):
            data = _run_probe(provider.python_exe)
            if data and os.path.normcase(
                os.path.normpath(str(data.get("site_packages") or ""))
            ) == os.path.normcase(os.path.normpath(provider.site_packages)):
                return True, ""
            # site-packages may still work even if python path moved
        if not os.path.isdir(provider.site_packages):
            return False, f"site-packages missing: {provider.site_packages}"
        # Probe with current interpreter after path inject
        apply_provider_to_sys_path(provider)
        try:
            import importlib

            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                return False, "torch.cuda.is_available() is False"
            cuda_rt = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
            if not cuda_rt.startswith("12"):
                return False, f"CUDA {cuda_rt!r} is not 12.x"
            return True, ""
        except Exception as exc:
            return False, str(exc)

    if provider.is_bundled:
        try:
            import importlib

            torch = importlib.import_module("torch")
            if not torch.cuda.is_available():
                return False, "bundled torch has no CUDA"
            return True, ""
        except Exception as exc:
            return False, str(exc)

    return False, "no provider configured"


def _notify_user(title: str, message: str) -> None:
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is not None:
            QMessageBox.warning(None, title, message)
            return
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x00000030)
            return
        except Exception:
            pass
    logger.warning("%s: %s", title, message)


def install_bundled_torch_into_pixi(
    install_dir: Optional[str] = None,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> TorchProvider:
    """Download cu124 torch (+ kornia) into the local Pixi env (legacy BYO path)."""

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            logger.info("%s", msg)

    root = install_dir or _default_install_dir()
    py = os.path.join(root, ".pixi", "envs", "default", "python.exe")
    if not os.path.isfile(py):
        # Dev checkout: use current interpreter
        py = sys.executable

    _log("Downloading PyTorch CUDA 12.4 into the RAWviewer environment...")
    cmds = [
        [
            py,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        [
            py,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "kornia>=0.8.3,<0.9",
        ],
    ]
    for cmd in cmds:
        _log("  " + " ".join(cmd[-6:]))
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60 * 60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32"
            else 0,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError(f"pip install failed ({proc.returncode}): {err}")

    # Drop link-time .lib files if present.
    try:
        from bootstrap import _prune_torch_link_libs  # type: ignore

        _prune_torch_link_libs(root, _log)
    except Exception:
        torch_lib = os.path.join(
            root, ".pixi", "envs", "default", "Lib", "site-packages", "torch", "lib"
        )
        if os.path.isdir(torch_lib):
            for name in os.listdir(torch_lib):
                if name.lower().endswith(".lib"):
                    try:
                        os.remove(os.path.join(torch_lib, name))
                    except OSError:
                        pass

    data = _run_probe(py) or {}
    provider = TorchProvider(
        mode="bundled",
        site_packages=str(data.get("site_packages") or ""),
        python_exe=os.path.normpath(py),
        torch_version=str(data.get("torch_version") or ""),
        cuda_version=str(data.get("cuda_version") or ""),
        validated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    save_provider(provider, root)
    _log(
        f"Bundled PyTorch ready: {provider.torch_version or 'installed'} "
        f"(CUDA {provider.cuda_version or '?'})."
    )
    return provider


def install_byo_sidecar_packages(
    install_dir: str,
    external: TorchProvider,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Install kornia into Pixi without pulling a second torch."""

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            logger.info("%s", msg)

    py = os.path.join(install_dir, ".pixi", "envs", "default", "python.exe")
    if not os.path.isfile(py):
        raise FileNotFoundError(f"Pixi python missing: {py}")

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        external.site_packages
        + os.pathsep
        + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    env["RAWVIEWER_TORCH_SITE_PACKAGES"] = external.site_packages

    _log("Installing kornia (--no-deps) against external torch...")
    cmd_no_deps = [
        py,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-deps",
        "kornia>=0.8.3,<0.9",
        "kornia_rs",
    ]
    cmd_small = [
        py,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "packaging",
    ]
    for cmd in (cmd_no_deps, cmd_small):
        proc = subprocess.run(
            cmd,
            cwd=install_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30 * 60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32"
            else 0,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise RuntimeError(f"sidecar pip failed: {err}")
    _log("Sidecar packages installed.")


def ensure_torch_for_gpu(
    *,
    install_dir: Optional[str] = None,
    notify: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> bool:
    """Ensure a working CUDA torch is importable; repair by bundling if needed.

    Returns True when ``import torch`` + CUDA looks usable after this call.
    """

    def _log(msg: str) -> None:
        if log:
            log(msg)
        else:
            logger.info("%s", msg)

    root = install_dir or _default_install_dir()
    apply_at_startup()
    prov = load_provider(root)

    ok, err = validate_provider(prov) if prov.mode != "none" else (False, "none")
    if ok:
        return True

    # Try a fresh discovery before downloading.
    if prov.is_external or prov.mode == "none":
        found = discover_external_torch(log=_log)
        if found:
            save_provider(found, root)
            apply_provider_to_sys_path(found)
            ok2, err2 = validate_provider(found)
            if ok2:
                return True
            err = err2 or err

    if notify:
        reason = err or "PyTorch CUDA is unavailable"
        _notify_user(
            "RAWviewer — GPU libraries",
            "The external PyTorch CUDA install is missing or broken.\n\n"
            f"Details: {reason}\n\n"
            "RAWviewer will download PyTorch into its own environment "
            "(~2–4 GB). This may take several minutes.",
        )

    try:
        bundled = install_bundled_torch_into_pixi(root, log=_log)
        apply_provider_to_sys_path(TorchProvider(mode="none"))  # clear external path bias
        # Bundled lives on default sys.path inside pixi — remove stale external.
        sp = os.environ.pop("RAWVIEWER_TORCH_SITE_PACKAGES", None)
        if sp and sp in sys.path:
            try:
                sys.path.remove(sp)
            except ValueError:
                pass
        ok3, err3 = validate_provider(bundled)
        if not ok3:
            _log(f"Bundled torch validation failed: {err3}")
            return False
        return True
    except Exception as exc:
        _log(f"Failed to install bundled torch: {exc}")
        if notify:
            _notify_user(
                "RAWviewer — GPU libraries",
                "Could not download PyTorch into the RAWviewer environment.\n\n"
                f"{exc}\n\n"
                "GPU demosaic will stay disabled; CPU Fast RAW still works.",
            )
        return False
