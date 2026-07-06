#!/usr/bin/env python3
"""Adjust-panel slider specs: value<->slider mappings are inverses; ranges sane."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from raw_adjustments import DEFAULT_ADJUSTMENTS, SLIDER_SPECS

    for spec in SLIDER_SPECS:
        check(f"{spec.key} range sane", spec.minimum < spec.maximum)
        check(
            f"{spec.key} default within range",
            spec.minimum <= spec.value_to_slider(spec.default_value) <= spec.maximum,
        )
        for slider_pos in (spec.minimum, spec.value_to_slider(spec.default_value), spec.maximum):
            v = spec.slider_to_value(slider_pos)
            rt = spec.value_to_slider(v)
            check(
                f"{spec.key} inverse @ {slider_pos}",
                rt == slider_pos,
                f"slider {slider_pos} -> value {v} -> slider {rt}",
            )
        check(f"{spec.key} in defaults", spec.key in DEFAULT_ADJUSTMENTS)

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
