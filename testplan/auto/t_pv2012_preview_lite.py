#!/usr/bin/env python3
"""PV2012 preview_lite skips expensive guided-filter stages (live-drag)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    import inspect

    from raw_edit_pipeline import PreviewStageCache, render_adjust_preview_uint8
    from raw_pv2012 import apply_pv2012_tone_rgb

    sig = inspect.signature(apply_pv2012_tone_rgb)
    check("apply_pv2012_tone_rgb has preview_lite", "preview_lite" in sig.parameters)

    src = inspect.getsource(apply_pv2012_tone_rgb)
    check("preview_lite skips y0 guided filter branch", "if preview_lite:" in src)

    # Assert behaviour, not source text. This check used to grep
    # apply_pv2012_tone_rgb for "and not preview_lite"; hoisting that
    # condition into a named skip_chroma_damp variable broke the test while
    # the behaviour was completely unchanged. Count the calls instead.
    import raw_pv2012

    _probe = np.random.default_rng(1).random((120, 160, 3), dtype=np.float32) * 0.08
    _probe_adj = {
        "Contrast2012": 40.0,
        "Shadows2012": 80.0,
        "Blacks2012": 20.0,
        "Highlights2012": -10.0,
        "Whites2012": 0.0,
    }
    _real_damp = raw_pv2012._damp_chroma_after_lift
    _calls = {"n": 0}

    def _counting_damp(*a, **kw):
        _calls["n"] += 1
        return _real_damp(*a, **kw)

    raw_pv2012._damp_chroma_after_lift = _counting_damp
    try:
        counted = {}
        for label, kwargs in (
            ("full", {}),
            ("lite", {"preview_lite": True}),
            ("settle", {"settle_fast": True}),
        ):
            _calls["n"] = 0
            raw_pv2012.apply_pv2012_tone_rgb(_probe, _probe_adj, **kwargs)
            counted[label] = _calls["n"]
    finally:
        raw_pv2012._damp_chroma_after_lift = _real_damp

    check(
        "full quality runs the shadow chroma damp",
        counted["full"] > 0,
        f"{counted['full']} call(s)",
    )
    check(
        "preview_lite skips shadow chroma damp",
        counted["lite"] == 0,
        f"{counted['lite']} call(s)",
    )
    check(
        "settle_fast skips shadow chroma damp too",
        counted["settle"] == 0,
        f"{counted['settle']} call(s)",
    )

    rng = np.random.default_rng(1)
    # Small linear float RGB with deep shadows so lift path is exercised.
    img = rng.random((120, 160, 3), dtype=np.float32) * 0.08
    adj = {
        "Contrast2012": 40.0,
        "Shadows2012": 80.0,
        "Blacks2012": 20.0,
        "Highlights2012": -10.0,
        "Whites2012": 0.0,
    }
    full = apply_pv2012_tone_rgb(img, adj, preview_lite=False)
    lite = apply_pv2012_tone_rgb(img, adj, preview_lite=True)
    check("full and lite return same shape", full.shape == lite.shape == img.shape)
    check("full and lite are finite", np.isfinite(full).all() and np.isfinite(lite).all())
    # Lite intentionally omits chroma damp / y0 smooth — outputs must differ
    # when Shadows lift is strong, otherwise the flag is a no-op.
    check(
        "lite differs from full under strong Shadows",
        not np.allclose(full, lite, rtol=1e-4, atol=1e-4),
    )

    # Staged render accepts preview_lite and returns uint8
    base = (img * 65535.0).astype(np.float32)
    cache = PreviewStageCache()
    out = render_adjust_preview_uint8(
        base, adj, cache, preview_lite=True
    )
    check(
        "render_adjust_preview_uint8 lite is uint8",
        out is not None and out.dtype == np.uint8,
    )
    main_py = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "main.py"
    )
    with open(main_py, "r", encoding="utf-8") as f:
        main_src = f.read()
    check(
        "_ADJUST_FAST_PREVIEW_MAX_EDGE is 640",
        "_ADJUST_FAST_PREVIEW_MAX_EDGE = 640" in main_src,
    )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
