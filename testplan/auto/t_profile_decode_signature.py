#!/usr/bin/env python3
"""Camera profile decode signature -- headless.

A ColorChecker profile is a correction measured on top of whatever the
decoder produced the day the chart was shot. The decoder is not fixed:
fast_raw_decode repairs LibRaw's black level, and on bodies LibRaw misparses
it repairs the as-shot white balance against the embedded JPEG. A LibRaw
upgrade can stop a repair firing because the defect it worked around is
gone -- at which point the saved profile is compensating for a decode that no
longer exists and pushes colour the wrong way by roughly the size of the old
repair.

That is not hypothetical for the EOS R6 Mark III: the WB repair no longer
fires for it. So profiles record what they were calibrated against, and a
mismatch is surfaced rather than silently applied.

The profile is still APPLIED on mismatch. It is the user's measurement and
may well be closer than nothing; what is unacceptable is the change being
invisible.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import color_calibration as cc  # noqa: E402


def _profile(**signature):
    prof = {"temperature_shift": 120.0, "tint_shift": -3.0, "make": "Canon", "model": "X"}
    if signature:
        prof["decode_signature"] = dict(signature)
    return prof


def test_signature_records_the_libraw_build():
    sig = cc.current_decode_signature()
    assert "libraw" in sig and "wb_corrected" in sig
    assert sig["libraw"], "no LibRaw version captured -- the check cannot fire"
    print(f"  OK   signature records LibRaw {sig['libraw']}")


def test_unknown_wb_state_is_none_not_false():
    """False would read as a measurement; None says we did not look."""
    sig = cc.current_decode_signature(None)
    assert sig["wb_corrected"] is None, (
        f"unknown WB state reported as {sig['wb_corrected']!r}"
    )
    print("  OK   an unmeasured WB state is None, not a false negative")


def test_libraw_change_is_reported():
    prof = _profile(libraw="0.0.1", wb_corrected=False)
    reason = cc.profile_decode_mismatch(prof)
    assert reason, "a LibRaw change was not reported"
    assert "0.0.1" in reason, reason
    assert "chart" in reason.lower(), "the message does not say what to do"
    print("  OK   a LibRaw change is reported, with the fix")


def test_matching_signature_is_silent():
    prof = _profile(**cc.current_decode_signature())
    assert cc.profile_decode_mismatch(prof) is None, "a matching decode warned anyway"
    print("  OK   a matching decode says nothing")


def test_wb_guardrail_state_change_is_reported_both_ways():
    now = cc.current_decode_signature()

    was_on = _profile(libraw=now["libraw"], wb_corrected=True)
    off_now = cc.profile_decode_mismatch(was_on, None)
    # With no file the current state is unknown, so nothing can be claimed.
    assert off_now is None, f"claimed a change from an unknown state: {off_now}"

    # Both directions matter: a guardrail that starts firing invalidates a
    # profile just as thoroughly as one that stops.
    for saved, current, expect in ((True, False, "no longer"), (False, True, "now being")):
        prof = _profile(libraw=now["libraw"], wb_corrected=saved)
        reason = cc._compare_signatures(prof["decode_signature"],
                                        {"libraw": now["libraw"], "wb_corrected": current})
        assert reason and expect in reason, f"{saved}->{current}: {reason!r}"
    print("  OK   a WB guardrail state change is reported in both directions")


def test_legacy_profiles_without_a_signature_do_not_warn():
    """Every profile saved before this existed lacks one and is not known bad."""
    assert cc.profile_decode_mismatch(_profile()) is None
    assert cc.profile_decode_mismatch({}) is None
    assert cc.profile_decode_mismatch(None) is None
    print("  OK   a profile with no signature is quiet, not assumed stale")


def test_saving_stamps_the_signature(tmp=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = os.path.join(tmpdir, "camera_profiles.json")
        original = cc.get_camera_profile_path
        cc.get_camera_profile_path = lambda: store
        try:
            assert cc.save_camera_profile("Canon", "EOS R6 Mark III",
                                          {"temperature_shift": 10.0}, iso=100)
            saved = json.load(open(store, encoding="utf-8"))
            entry = next(iter(saved.values()))
            assert "decode_signature" in entry, "profile saved without a signature"
            assert entry["decode_signature"].get("libraw"), "signature has no LibRaw build"

            # And it round-trips through the reader the app actually uses.
            prof = cc.get_camera_profile("Canon", "EOS R6 Mark III", iso=100)
            assert prof and "decode_signature" in prof
            assert cc.profile_decode_mismatch(prof) is None, (
                "a profile just saved on this build reports itself stale"
            )
        finally:
            cc.get_camera_profile_path = original
    print("  OK   saving stamps the signature; it round-trips clean")


def test_banner_shows_the_reason():
    """The log is not enough -- the banner is where the user can see it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = os.path.join(tmpdir, "camera_profiles.json")
        original = cc.get_camera_profile_path
        cc.get_camera_profile_path = lambda: store
        try:
            cc.save_camera_profile("Canon", "EOS R6 Mark III",
                                   {"temperature_shift": 10.0}, iso=100)
            # Age the stored signature.
            data = json.load(open(store, encoding="utf-8"))
            key = next(iter(data))
            data[key]["decode_signature"]["libraw"] = "0.0.1"
            json.dump(data, open(store, "w", encoding="utf-8"))

            label = cc.describe_camera_profile("Canon", "EOS R6 Mark III", iso=100)
            assert label and "0.0.1" in label, f"banner hides the mismatch: {label!r}"
            assert "EOS R6 Mark III" in label, "banner lost the camera name"
        finally:
            cc.get_camera_profile_path = original
    print("  OK   the banner names the camera and the mismatch")


def test_a_stale_profile_is_still_applied():
    """Dropping the user's measurement silently would be the worse failure."""
    src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "raw_adjustments.py"
    )
    text = open(src, encoding="utf-8").read()
    assert "_warn_if_profile_decode_stale" in text, "no staleness check in the load path"
    start = text.index("def _warn_if_profile_decode_stale")
    end = text.index("\ndef ", start + 1)
    body = text[start:end]
    assert "return" in body and "raise" not in body, (
        "the staleness check should warn, not abort the load"
    )
    assert "_PROFILE_STALE_WARNED" in body, "warns per file instead of per profile"
    print("  OK   a stale profile still applies, and warns once per profile")


def main() -> int:
    test_signature_records_the_libraw_build()
    test_unknown_wb_state_is_none_not_false()
    test_libraw_change_is_reported()
    test_matching_signature_is_silent()
    test_wb_guardrail_state_change_is_reported_both_ways()
    test_legacy_profiles_without_a_signature_do_not_warn()
    test_saving_stamps_the_signature()
    test_banner_shows_the_reason()
    test_a_stale_profile_is_still_applied()
    print("\nPASS t_profile_decode_signature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
