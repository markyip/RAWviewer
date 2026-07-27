"""Local InstructPix2Pix provider: tokenizer, scheduler, wiring.

Runs without the 2.2 GB model set present -- the pieces that can be checked
cheaply are checked properly, and the parts that need weights are skipped
rather than faked into a green tick. The tokenizer is the exception worth
being strict about: it has a known-correct answer, and a wrong one produces
a plausible-looking image that quietly ignores the instruction.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_scheduler():
    from onnx_ip2p_scheduler import EulerAncestralScheduler

    s = EulerAncestralScheduler()
    s.set_timesteps(10)
    check("timesteps count", len(s.timesteps) == 10, f"{len(s.timesteps)}")
    check("sigmas has the trailing zero", len(s.sigmas) == 11 and s.sigmas[-1] == 0.0)
    check("timesteps descend", bool(np.all(np.diff(s.timesteps) < 0)))
    check("sigmas descend", bool(np.all(np.diff(s.sigmas) <= 0)))
    check("init sigma is the largest", s.init_noise_sigma == float(s.sigmas[0]))

    # scale_model_input must bring a sigma-scaled latent back to ~unit variance.
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((1, 4, 8, 8)).astype(np.float32) * s.init_noise_sigma
    scaled = s.scale_model_input(latent, 0)
    check(
        "scale_model_input normalises variance",
        0.5 < float(scaled.std()) < 2.0,
        f"std={float(scaled.std()):.3f}",
    )

    # Zero epsilon means "this latent holds no noise", so the step must NOT
    # drift -- derivative is zero and only ancestral noise is added. Anything
    # else would mean the drift term has the wrong sign or scale.
    quiet = s.step(np.zeros_like(latent), 0, latent, generator=np.random.default_rng(1))
    drift = float(np.abs(quiet - latent).mean())
    sigma_up_scale = float(s.sigmas[1])
    check(
        "zero epsilon produces no drift, only ancestral noise",
        drift < sigma_up_scale,
        f"drift={drift:.2f} < sigma_1={sigma_up_scale:.2f}",
    )

    # The real contract: a latent that IS pure noise has true epsilon
    # sample/sigma, and predicting it must drive the latent toward zero.
    # This is the direction that makes sampling converge at all.
    latents = rng.standard_normal((1, 4, 8, 8)).astype(np.float32) * s.init_noise_sigma
    start = float(np.abs(latents).mean())
    for i in range(len(s.timesteps)):
        sigma = float(s.sigmas[i])
        eps = latents / sigma if sigma > 0 else np.zeros_like(latents)
        latents = s.step(eps, i, latents, generator=rng)
    end = float(np.abs(latents).mean())
    check("a full sweep stays finite", bool(np.all(np.isfinite(latents))))
    check(
        "predicting all-noise converges toward a clean latent",
        end < start * 0.1,
        f"mean|x| {start:.2f} -> {end:.3f}",
    )


def test_tokenizer():
    import raw_generative_local as loc
    from clip_bpe_tokenizer import load_tokenizer

    vocab, merges = loc.file_path("vocab"), loc.file_path("merges")
    if not (os.path.isfile(vocab) and os.path.isfile(merges)):
        print("SKIP  tokenizer files not downloaded on this machine")
        return

    tok = load_tokenizer(vocab, merges)
    check("tokenizer loads", tok is not None)
    if tok is None:
        return
    check("CLIP vocabulary size", len(tok.encoder) == 49408, f"{len(tok.encoder)}")
    check("start/end ids", (tok.sot, tok.eot) == (49406, 49407))

    # Reference ids for this phrase are stable across every CLIP release.
    got = [i for i in tok.tokenize("a photo of a cat") if i not in (tok.sot, tok.eot)]
    check(
        "known phrase tokenises to the reference ids",
        got == [320, 1125, 539, 320, 2368],
        f"{got}",
    )
    check(
        "always padded to the context length",
        all(len(tok.tokenize(t)) == 77 for t in ("", "hi", "word " * 200)),
    )
    over = tok.tokenize("word " * 200)
    check("over-long text still ends with the end token", over[-1] == tok.eot)
    check("empty text is just start+end padding", tok.tokenize("")[0] == tok.sot)


def test_fit_to_model():
    import raw_generative_local as loc

    big = np.zeros((2000, 3000, 3), dtype=np.float32)
    small, orig = loc._fit_to_model(big, 512)
    check("original size is reported back", orig == (2000, 3000))
    check("long edge is capped", max(small.shape[:2]) <= 512, f"{small.shape}")
    check(
        "both sides are latent-aligned",
        small.shape[0] % 8 == 0 and small.shape[1] % 8 == 0,
        f"{small.shape}",
    )
    # A tiny image must not be upscaled into fake detail.
    tiny = np.zeros((64, 48, 3), dtype=np.float32)
    out, _ = loc._fit_to_model(tiny, 512)
    check("small images are not upscaled", out.shape[:2] == (64, 48), f"{out.shape}")


def test_provider_wiring():
    import raw_generative_local as loc
    from raw_generative_edit import make_provider

    p = make_provider({"provider": "local", "local_steps": 7})
    check("make_provider builds the local provider", type(p).__name__ == "LocalInstructPix2PixProvider")
    check("settings reach the provider", p.steps == 7, f"steps={p.steps}")
    check("local editing needs no consent", p.requires_consent is False)
    check("provider name", p.name == "local-ip2p")
    check(
        "is_configured tracks whether the model is present",
        p.is_configured() == loc.is_downloaded(),
    )
    check(
        "describe() names the download when absent",
        loc.is_downloaded() or "not downloaded" in p.describe().lower(),
        p.describe(),
    )

    # An empty instruction must be refused before any 2 GB session is built.
    from raw_generative_edit import GenerativeEditError, GenerativeRequest

    if loc.is_downloaded():
        try:
            p.edit(GenerativeRequest(image=np.zeros((64, 64, 3), np.uint8),
                                     instruction="   ", source_path="/tmp/x"))
            check("empty instruction is refused", False)
        except GenerativeEditError:
            check("empty instruction is refused", True)


def test_settings_roundtrip():
    from PyQt6.QtCore import QCoreApplication, QSettings

    QCoreApplication.setOrganizationName("RAWviewerTest")
    QCoreApplication.setApplicationName("RAWviewerTest")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    import generative_settings as gs

    scope = QSettings("RAWviewerTest", "GenLocalTest")
    scope.clear()
    real = gs._settings
    gs._settings = lambda: scope
    try:
        gs.save_local_settings(steps=14, text_guidance=6.0, image_guidance=2.0)
        loaded = gs.load_settings()
        check("provider switched to local", loaded["provider"] == "local")
        check("steps persisted", loaded["local_steps"] == 14)
        check("text guidance persisted", abs(loaded["local_text_guidance"] - 6.0) < 1e-6)
        check("image guidance persisted", abs(loaded["local_image_guidance"] - 2.0) < 1e-6)
    finally:
        gs._settings = real
        scope.clear()


def test_weights_are_pinned():
    """The conversion repo is a 2-download third-party upload.

    Pinning the bytes is what makes that acceptable: a re-upload or a
    substituted graph must fail loudly rather than run.
    """
    import raw_generative_local as loc

    for key, entry in loc._FILES.items():
        check(f"{key} carries a pinned hash", len(entry) == 3 and len(entry[2]) == 64)

    if not loc.is_downloaded():
        print("SKIP  weights not present; cannot verify contents")
        return
    bad = loc.verify_downloaded()
    check("every downloaded file matches its pin", bad == [], f"mismatched: {bad}")


def test_model_dir_is_not_committed():
    """2.2 GB must never enter the repository."""
    import subprocess

    repo = os.path.join(os.path.dirname(__file__), "..", "..")
    probe = os.path.join("models", "ip2p", "unet_model.onnx")
    result = subprocess.run(
        ["git", "check-ignore", probe],
        cwd=repo, capture_output=True, text=True,
    )
    check("models/ip2p is gitignored", result.returncode == 0, result.stdout.strip())


def main() -> int:
    test_scheduler()
    test_tokenizer()
    test_fit_to_model()
    test_provider_wiring()
    test_settings_roundtrip()
    test_weights_are_pinned()
    test_model_dir_is_not_committed()
    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
