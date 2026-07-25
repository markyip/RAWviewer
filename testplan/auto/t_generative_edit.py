#!/usr/bin/env python3
"""Generative editing provider layer (raw_generative_edit).

Runs with no model, no GPU and no network: the HTTP provider is
exercised against a local one-shot server on 127.0.0.1, and everything
else against the StubProvider. That is the point of the provider
abstraction -- the whole round-trip is testable before any real endpoint
or API key exists.

Checks:
  1. Pixel baking: float/uint8/grayscale in, uint8 RGB out, highlight
     overshoot clipped (the latitude loss is real and happens once, here).
  2. PNG base64 round-trip is lossless.
  3. Provenance records model/instruction/seed, and CHAINS when a
     generated file is generated from again -- the lineage back to the
     RAW must stay legible.
  4. Cancellation: a cancelled request raises rather than returning a
     half-finished image.
  5. Endpoint validation refuses plain http to a non-loopback host unless
     explicitly allowed -- photos must not leak over the wire by default.
  6. HTTP provider: real request/response cycle, error surfacing, and
     that a server-side error message reaches the user.
  7. make_provider never raises on empty/garbage settings.
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402

import raw_generative_edit as gen  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def _img(h=8, w=8):
    rng = np.random.default_rng(7)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_bake():
    f = np.zeros((4, 4, 3), dtype=np.float32)
    f[0, 0] = 0.5
    out = gen._to_uint8_rgb(f)
    check("float -> uint8", out.dtype == np.uint8 and out[0, 0, 0] in (127, 128))

    # Scene-linear highlight overshoot must clip here, not silently wrap.
    hot = np.full((2, 2, 3), 4.0, dtype=np.float32)
    check("highlight overshoot clipped", gen._to_uint8_rgb(hot).max() == 255)

    gray = np.zeros((4, 4), dtype=np.uint8)
    check("grayscale promoted", gen._to_uint8_rgb(gray).shape == (4, 4, 3))

    try:
        gen._to_uint8_rgb(np.zeros((4,), dtype=np.uint8))
        check("non-image rejected", False, "no raise")
    except gen.GenerativeEditError:
        check("non-image rejected", True)


def test_png_roundtrip():
    src = _img(16, 12)
    decoded = gen._decode_png_b64(gen._encode_png_b64(src))
    check("PNG round-trip lossless", np.array_equal(src, decoded))
    check("round-trip preserves shape", decoded.shape == src.shape)

    try:
        gen._decode_png_b64("not base64 @@@")
        check("garbage rejected", False, "no raise")
    except gen.GenerativeEditError:
        check("garbage rejected", True)


def test_provenance_chain():
    req = gen.GenerativeRequest(_img(), "remove the bin", seed=42, source_path="/x/IMG_1.CR3")
    p1 = gen.build_provenance(req, "stub", "stub-v1")
    check("records instruction", p1["instruction"] == "remove the bin")
    check("records seed", p1["seed"] == 42)
    check("records model", p1["model"] == "stub-v1")
    check("flagged as generated", p1["generated"] is True)
    check("first edit has empty chain", p1["chain"] == [])

    # Chaining: edit the generated file again.
    req2 = gen.GenerativeRequest(_img(), "now make it dusk", seed=1)
    p2 = gen.build_provenance(req2, "stub", "stub-v1", parent_provenance=p1)
    check("chain records the parent", len(p2["chain"]) == 1, f"got {len(p2['chain'])}")
    check("parent instruction preserved", p2["chain"][0]["instruction"] == "remove the bin")
    check("chain entries are flat", "chain" not in p2["chain"][0])

    p3 = gen.build_provenance(req2, "stub", "stub-v1", parent_provenance=p2)
    check("chain grows, not nests", len(p3["chain"]) == 2, f"got {len(p3['chain'])}")


def test_stub_and_cancel():
    provider = gen.StubProvider()
    req = gen.GenerativeRequest(_img(), "do a thing")
    result = provider.edit(req)
    check("stub returns an image", result.image.shape == req.image.shape)
    check("stub result is uint8", result.image.dtype == np.uint8)
    check("stub actually changed pixels", not np.array_equal(result.image, req.image))
    check("stub carries provenance", result.provenance["model"] == "stub-v1")
    check("stub needs no consent", provider.requires_consent is False)

    # Cancelled before start.
    token = gen.CancelToken()
    token.cancel()
    try:
        gen.StubProvider(delay_s=0.5).edit(req, cancel=token)
        check("cancel raises", False, "returned a result")
    except gen.CancelledError:
        check("cancel raises", True)

    # Cancelled mid-flight from another thread.
    token2 = gen.CancelToken()
    threading.Timer(0.05, token2.cancel).start()
    try:
        gen.StubProvider(delay_s=3.0).edit(req, cancel=token2)
        check("mid-flight cancel raises", False, "returned a result")
    except gen.CancelledError:
        check("mid-flight cancel raises", True)

    # Provider errors surface as GenerativeEditError.
    try:
        gen.StubProvider(fail_with="model exploded").edit(req)
        check("provider error surfaces", False, "no raise")
    except gen.GenerativeEditError as exc:
        check("provider error surfaces", "model exploded" in str(exc))


def test_endpoint_validation():
    empty = gen.HttpEndpointProvider("")
    check("empty endpoint is unconfigured", empty.is_configured() is False)
    try:
        empty.edit(gen.GenerativeRequest(_img(), "x"))
        check("empty endpoint raises", False, "no raise")
    except gen.GenerativeEditError:
        check("empty endpoint raises", True)

    # Plain http to a remote host would leak the photo on the wire.
    insecure = gen.HttpEndpointProvider("http://example.com/edit")
    try:
        insecure.edit(gen.GenerativeRequest(_img(), "x"))
        check("plain http to remote refused", False, "no raise")
    except gen.GenerativeEditError as exc:
        check("plain http to remote refused", "plain http" in str(exc).lower(), str(exc))

    # Loopback is the privacy-preserving case and must stay allowed.
    local = gen.HttpEndpointProvider("http://127.0.0.1:9/edit")
    try:
        local._validate_endpoint()
        check("loopback http allowed", True)
    except gen.GenerativeEditError as exc:
        check("loopback http allowed", False, str(exc))

    # Explicit opt-in for a trusted LAN box.
    opted = gen.HttpEndpointProvider("http://192.168.1.5/edit", allow_insecure=True)
    try:
        opted._validate_endpoint()
        check("insecure opt-in honoured", True)
    except gen.GenerativeEditError as exc:
        check("insecure opt-in honoured", False, str(exc))

    bad = gen.HttpEndpointProvider("ftp://example.com/edit")
    try:
        bad._validate_endpoint()
        check("non-http scheme refused", False, "no raise")
    except gen.GenerativeEditError:
        check("non-http scheme refused", True)


def _serve_once(handler):
    """Minimal one-request HTTP server on a free loopback port."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            status, payload = handler(body, self.headers)
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=server.handle_request, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/edit"


def test_http_roundtrip():
    seen = {}

    def handler(body, headers):
        seen["instruction"] = body.get("instruction")
        seen["seed"] = body.get("seed")
        seen["auth"] = headers.get("Authorization")
        img = gen._decode_png_b64(body["image"])
        seen["shape"] = img.shape
        # Echo a modified image back.
        return 200, {"image": gen._encode_png_b64(255 - img), "model": "fake-model-v2"}

    server, url = _serve_once(handler)
    try:
        provider = gen.HttpEndpointProvider(url, api_key="sekrit")
        src = _img(10, 14)
        result = provider.edit(
            gen.GenerativeRequest(src, "make it rain", seed=99, source_path="/a/b.CR3")
        )
        check("http returns an image", result.image.shape == src.shape, f"got {result.image.shape}")
        check("http result inverted as expected", np.array_equal(result.image, 255 - src))
        check("instruction transmitted", seen["instruction"] == "make it rain")
        check("seed transmitted", seen["seed"] == 99)
        check("api key sent as bearer", seen["auth"] == "Bearer sekrit")
        check("image transmitted intact", seen["shape"] == src.shape)
        check("model name from response", result.provenance["model"] == "fake-model-v2")
        check("source recorded", result.provenance["source"] == "/a/b.CR3")
    finally:
        server.server_close()


def test_http_error_surfacing():
    def handler(body, headers):
        return 400, {"error": "prompt rejected by safety filter"}

    server, url = _serve_once(handler)
    try:
        provider = gen.HttpEndpointProvider(url)
        try:
            provider.edit(gen.GenerativeRequest(_img(), "something"))
            check("server error surfaces", False, "no raise")
        except gen.GenerativeEditError as exc:
            check("server error surfaces", "safety filter" in str(exc), str(exc))
    finally:
        server.server_close()

    def no_image(body, headers):
        return 200, {"model": "x"}

    server2, url2 = _serve_once(no_image)
    try:
        try:
            gen.HttpEndpointProvider(url2).edit(gen.GenerativeRequest(_img(), "y"))
            check("missing image surfaces", False, "no raise")
        except gen.GenerativeEditError as exc:
            check("missing image surfaces", "no image" in str(exc).lower(), str(exc))
    finally:
        server2.server_close()


def test_blank_instruction():
    try:
        gen.HttpEndpointProvider("https://example.com/e").edit(
            gen.GenerativeRequest(_img(), "   ")
        )
        check("blank instruction rejected", False, "no raise")
    except gen.GenerativeEditError as exc:
        check("blank instruction rejected", "describe" in str(exc).lower(), str(exc))


def test_make_provider():
    p = gen.make_provider({})
    check("default provider is http", isinstance(p, gen.HttpEndpointProvider))
    check("default is unconfigured (no default endpoint)", p.is_configured() is False)
    check("remote requires consent", p.requires_consent is True)

    p2 = gen.make_provider({"provider": "stub"})
    check("stub selectable", isinstance(p2, gen.StubProvider))

    # Garbage settings must not raise -- the feature just stays off.
    try:
        p3 = gen.make_provider({"provider": None, "timeout_s": None, "endpoint": None})
        check("garbage settings tolerated", p3.is_configured() is False)
    except Exception as exc:  # noqa: BLE001
        check("garbage settings tolerated", False, f"raised {exc!r}")


def main():
    print("Generative editing provider layer")
    test_bake()
    test_png_roundtrip()
    test_provenance_chain()
    test_stub_and_cancel()
    test_endpoint_validation()
    test_http_roundtrip()
    test_http_error_surfacing()
    test_blank_instruction()
    test_make_provider()

    print("")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
        return 1
    print("All generative edit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
