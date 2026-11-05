#!/usr/bin/env python3
"""Structural tests for idle timer system - verifies code structure via AST parsing."""

import os
import sys
import ast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MAIN_PY = os.path.join(ROOT, "src", "main.py")

    with open(MAIN_PY, "r", encoding="utf-8") as fh:
        src = fh.read()

    tree = ast.parse(src)

    # Find RAWImageViewer class
    klass = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RAWImageViewer":
            klass = node
            break

    check("RAWImageViewer class found", klass is not None)

    if klass:
        # Check for _full_res_idle_timer in __init__
        has_idle_timer = False
        has_idle_path = False
        has_restart_method = False
        has_timeout_method = False

        for node in klass.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "_restart_full_res_idle_timer":
                    has_restart_method = True
                elif node.name == "_on_full_res_idle_timeout":
                    has_timeout_method = True
                elif node.name == "__init__":
                    # Check assignments in __init__
                    for n in ast.walk(node):
                        if isinstance(n, ast.Assign):
                            for target in n.targets:
                                if isinstance(target, ast.Attribute):
                                    if target.attr == "_full_res_idle_timer":
                                        has_idle_timer = True
                                    elif target.attr == "_full_res_idle_path":
                                        has_idle_path = True

        check("_full_res_idle_timer attribute exists in __init__", has_idle_timer)
        check("_full_res_idle_path attribute exists in __init__", has_idle_path)
        check("_restart_full_res_idle_timer method exists", has_restart_method)
        check("_on_full_res_idle_timeout method exists", has_timeout_method)

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
            found = False
            for node in klass.body:
                if isinstance(node, ast.FunctionDef) and node.name == method_name:
                    found = True
                    break
            check(f"Old method {method_name} removed", not found)

        # Check signals - _single_view_first_render should be removed
        old_signals = [
            "_single_view_first_render",
            "_ttfr_timeout_timer",
        ]
        for signal_name in old_signals:
            found = False
            for node in klass.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute) and target.attr == signal_name:
                            found = True
                            break
            check(f"Old signal {signal_name} removed", not found)

    # Test 2: load_raw_image branch coverage
    print("\n=== load_raw_image Branch Coverage Tests ===")

    # Find load_raw_image method
    load_method = None
    for node in klass.body:
        if isinstance(node, ast.FunctionDef) and node.name == "load_raw_image":
            load_method = node
            break

    check("load_raw_image method exists", load_method is not None)

    if load_method:
        method_src = ast.get_source_segment(src, load_method) or ""

        # Check for key linear cache loading flow
        checks = [
            ("Full image cache check (via image_cache)", "self.image_cache.get_full_image" in method_src),
            ("Preview/thumbnail cache check", "get_preview_memory_only" in method_src),
            ("Pixmap cache check (non-RAW)", "get_pixmap" in method_src),
            ("image_manager.load_image call", "self.image_manager.load_image" in method_src),
            ("_is_raw_file check", "_is_raw_file" in method_src or "is_raw_file" in method_src),
        ]

        for name, condition in checks:
            check(name, condition)

    # Test 3: Verify FullImageCache class is defined
    print("\n=== FullImageCache Class Tests ===")
    
    full_cache_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FullImageCache":
            full_cache_class = node
            break
    
    check("FullImageCache class defined", full_cache_class is not None)
    
    if full_cache_class:
        # Check for key methods
        methods_found = set()
        for node in full_cache_class.body:
            if isinstance(node, ast.FunctionDef):
                methods_found.add(node.name)
        
        required_methods = ["__init__", "get", "put", "remove", "clear", "get_stats", 
                           "get_full_image", "put_full_image", "peek", "__contains__"]
        
        for method_name in required_methods:
            check(f"FullImageCache.{method_name} exists", method_name in methods_found)

    # Summary
    print(f"\n{'='*60}")
    total_checks = 35  # approximate
    print(f"Total checks run: ~{total_checks}")
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