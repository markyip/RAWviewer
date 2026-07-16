"""Cheap import / compile smoke — catches missing imports before packaging.

Would have caught the Lite ``name 'threading' is not defined`` crash in
``rawviewer_app/processing.py`` (module imported fine once ``threading``
was present; ``compileall`` alone does not execute imports).
"""
from __future__ import annotations

import compileall
import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

# Modules that start the GUI, spawn workers, or have heavy side effects at import.
_SKIP_MODULES = {
    "main",  # constructs QApplication / splash
    "bootstrap",
    "nav_autotest",
}
# Platform-specific modules that fail to import off-target (expected).
if sys.platform != "win32":
    _SKIP_MODULES.add("windows_share")
_SKIP_MODULES = frozenset(_SKIP_MODULES)

# Packages whose submodules we walk.
_PACKAGES = ("rawviewer_app", "rawviewer_ui")


def _module_name_from_path(py_path: Path, root: Path) -> str:
    rel = py_path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def _iter_src_modules() -> list[str]:
    names: list[str] = []
    for py in sorted(SRC.glob("*.py")):
        name = py.stem
        if name.startswith("_") or name in _SKIP_MODULES:
            continue
        names.append(name)
    for pkg in _PACKAGES:
        pkg_dir = SRC / pkg
        if not pkg_dir.is_dir():
            continue
        for py in sorted(pkg_dir.rglob("*.py")):
            if py.name.startswith("_") and py.name != "__init__.py":
                continue
            mod = _module_name_from_path(py, SRC)
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            if mod in _SKIP_MODULES:
                continue
            names.append(mod)
    return names


def test_compileall_src() -> None:
    ok = compileall.compile_dir(str(SRC), quiet=1, legacy=True)
    assert ok, "compileall reported failures under src/"


def test_import_src_modules() -> None:
    """Import every lightweight src module; fail on NameError/ImportError."""
    # Headless Qt for widgets that touch QApplication at import time.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        _ = app
    except Exception as exc:
        raise AssertionError(f"PyQt6 offscreen QApplication failed: {exc}") from exc

    failures: list[str] = []
    for name in _iter_src_modules():
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "import failures:\n  " + "\n  ".join(failures)


def main() -> int:
    test_compileall_src()
    print("PASS compileall src/")
    mods = _iter_src_modules()
    test_import_src_modules()
    print(f"PASS import {len(mods)} modules")
    print("PASS t_module_import_smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
