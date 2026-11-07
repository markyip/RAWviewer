#!/usr/bin/env python3
"""Structural tests for idle timer system - verifies code structure via AST parsing."""

import os
import sys
import ast
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _mixin_dir(root: str) -> str:
    return os.path.join(root, "src", "rawviewer_app", "window", "mixins")


def _load_window_methods(root: str):
    """Return ({method name: (source, node)}, [class-level assignment targets])."""
    methods: dict[str, tuple[str, ast.AST]] = {}
    signals: list[str] = []
    mixins = _mixin_dir(root)
    for name in sorted(os.listdir(mixins)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(mixins, name), encoding="utf-8") as fh:
            text = fh.read()
        tree = ast.parse(text)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods[item.name] = (text, item)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            signals.append(target.id)
    return methods, signals


def main() -> int:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    methods, signals = _load_window_methods(ROOT)

    # Find RAWImageViewer class (assembled from mixins)
    from rawviewer_app.window.raw_image_viewer import RAWImageViewer

    check("RAWImageViewer class found", RAWImageViewer is not None)

    # Check for _full_res_idle_timer in __init__
    has_idle_timer = False
    has_idle_path = False
    init_entry = methods.get("__init__")
    if init_entry:
        _text, init_node = init_entry
        for n in ast.walk(init_node):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Attribute):
                        if target.attr == "_full_res_idle_timer":
                            has_idle_timer = True
                        elif target.attr == "_full_res_idle_path":
                            has_idle_path = True

    check("_full_res_idle_timer attribute exists in __init__", has_idle_timer)
    check("_full_res_idle_path attribute exists in __init__", has_idle_path)
    check("_restart_full_res_idle_timer method exists",
          "_restart_full_res_idle_timer" in methods)
    check("_on_full_res_idle_timeout method exists",
          "_on_full_res_idle_timeout" in methods)

    # Verify old watchdog methods are removed
    old_methods = [
        "_arm_first_render_timer",
        "_schedule_ttfr_watchdog",
        "_on_first_render_ack",
        "_on_first_render_timeout",
        "_recover_stuck_first_render",
        "_arm_full_res_upgrade_timer",
        "_on_full_res_upgrade_timeout",
        "_note_displayed_resolution",
        "_displayed_buffer_covers_sensor",
        "_start_full_res_upgrade_safety_monitor",
        "_full_res_upgrade_safety_tick",
    ]
    for method_name in old_methods:
        check(f"Old method {method_name} removed", method_name not in methods)

    # Check signals - _single_view_first_render should be removed
    old_signals = ["_single_view_first_render", "_ttfr_timeout_timer"]
    for signal_name in old_signals:
        check(f"Old signal {signal_name} removed", signal_name not in signals)

    # Test 2: load_raw_image branch coverage
    print("\n=== load_raw_image Branch Coverage Tests ===")
    load_entry = methods.get("load_raw_image")
    check("load_raw_image method exists", load_entry is not None)

    if load_entry:
        text, node = load_entry
        method_src = ast.get_source_segment(text, node) or ""
        checks = [
            ("Full image cache check (via image_cache)",
             "self.image_cache.get_full_image" in method_src),
            ("Preview/thumbnail cache check",
             "get_preview_memory_only" in method_src),
            ("Pixmap cache check (non-RAW)", "get_pixmap" in method_src),
            ("image_manager.load_image call",
             "self.image_manager.load_image" in method_src),
            ("_is_raw_file check",
             "_is_raw_file" in method_src or "is_raw_file" in method_src),
        ]
        for name, condition in checks:
            check(name, condition)

    # Test 3: Verify FullImageCache class is defined
    print("\n=== FullImageCache Class Tests ===")
    cache_module = importlib.import_module("cache.memory")
    full_cache_class = getattr(cache_module, "FullImageCache", None)

    check("FullImageCache class defined", full_cache_class is not None)

    if full_cache_class:
        required_methods = ["__init__", "get", "put", "remove", "clear", "get_stats",
                            "get_full_image", "put_full_image", "peek", "__contains__"]
        for method_name in required_methods:
            check(f"FullImageCache.{method_name} exists",
                  hasattr(full_cache_class, method_name))

    # Summary
    print(f"\n{'='*60}")
    print(f"Total checks run: ~35")
    print(f"Failures: {len(FAILURES)}")

    if FAILURES:
        print("\nFAILED CHECKS:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    else:
        print("\nALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
