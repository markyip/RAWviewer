"""Local-server provider: the OpenAI /v1/images/edits client.

Exercised against a mock server implementing the same contract mlx-serve
does, so the wire format is verified without needing a 10 GB model or the
server installed.

The property worth guarding hardest is the consent rule: loopback needs no
consent because the photograph never leaves the machine, but the same
provider pointed at a remote host absolutely does. Getting that backwards
would upload a client's photographs silently.
"""

import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

FAILURES = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


class _Handler(BaseHTTPRequestHandler):
    """Minimal stand-in for mlx-serve's image-edit endpoint."""

    mode = "ok"
    seen = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length)
        _Handler.seen = {
            "content_type": self.headers.get("content-type", ""),
            "auth": self.headers.get("authorization", ""),
            "body": body,
        }

        if _Handler.mode == "http_error":
            payload = json.dumps({"error": {"message": "model not loaded"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if _Handler.mode == "url_response":
            out = {"data": [{"url": "https://example.com/x.png"}]}
        elif _Handler.mode == "empty":
            out = {"data": []}
        else:
            import cv2

            img = np.zeros((8, 12, 3), np.uint8)
            img[:, :, 0] = 200  # distinctive red so RGB/BGR order is testable
            ok, buf = cv2.imencode(".png", img[:, :, ::-1])  # cv2 wants BGR
            out = {"data": [{"b64_json": base64.b64encode(buf.tobytes()).decode()}]}

        payload = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    from raw_generative_edit import GenerativeEditError, GenerativeRequest, make_provider
    from raw_generative_local_server import (
        DEFAULT_ENDPOINT,
        LocalServerProvider,
        is_loopback,
    )

    # --- consent rule: the thing that must never be backwards ---
    check("loopback needs no consent", LocalServerProvider("http://localhost:11234/v1/images/edits").requires_consent is False)
    check("127.0.0.1 needs no consent", LocalServerProvider("http://127.0.0.1:9/v1/images/edits").requires_consent is False)
    check("a remote host DOES need consent", LocalServerProvider("https://api.example.com/v1/images/edits").requires_consent is True)
    check("is_loopback rejects lookalike hosts", not is_loopback("http://localhost.evil.com/x"))
    check("default endpoint is mlx-serve's", DEFAULT_ENDPOINT.startswith("http://localhost:11234"))

    # Plain http to a remote host must be refused outright.
    try:
        LocalServerProvider("http://example.com/v1/images/edits").edit(
            GenerativeRequest(image=np.zeros((4, 4, 3), np.uint8), instruction="x", source_path="/tmp/a")
        )
        check("plain http to a remote host is refused", False)
    except GenerativeEditError:
        check("plain http to a remote host is refused", True)

    # --- wire format against a mock server ---
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/v1/images/edits"

    try:
        p = LocalServerProvider(url, model_name="mage-flow-edit-turbo-8bit", api_key="k")
        img = np.zeros((8, 12, 3), np.uint8)
        img[:, :, 1] = 150
        req = GenerativeRequest(image=img, instruction="make it winter", source_path="/tmp/a.CR3", seed=7)

        steps = []
        result = p.edit(req, progress=steps.append)
        check("returns an image", result.image is not None and result.image.shape == (8, 12, 3), str(getattr(result.image, "shape", None)))
        check("RGB order preserved", int(result.image[0, 0, 0]) == 200, f"px={result.image[0,0].tolist()}")
        check("provenance records the provider", result.provenance.get("provider") == "local-server")
        check("progress was reported", len(steps) >= 2, f"{steps}")

        sent = _Handler.seen
        check("sent as multipart/form-data", sent["content_type"].startswith("multipart/form-data"))
        check("api key sent as bearer", sent["auth"] == "Bearer k")
        body = sent["body"]
        check("prompt in the body", b"make it winter" in body)
        check("model in the body", b"mage-flow-edit-turbo-8bit" in body)
        check("asks for b64_json", b"b64_json" in body)
        check("seed forwarded", b'name="seed"' in body and b"7" in body)
        check("image sent as PNG part", b"Content-Type: image/png" in body and b"\x89PNG" in body)

        # --- error paths ---
        _Handler.mode = "http_error"
        try:
            p.edit(req)
            check("server error message is surfaced", False)
        except GenerativeEditError as exc:
            check("server error message is surfaced", "model not loaded" in str(exc), str(exc))

        _Handler.mode = "url_response"
        try:
            p.edit(req)
            check("a URL response is refused, not fetched", False)
        except GenerativeEditError as exc:
            check("a URL response is refused, not fetched", "b64_json" in str(exc), str(exc))

        _Handler.mode = "empty"
        try:
            p.edit(req)
            check("empty data is an error", False)
        except GenerativeEditError:
            check("empty data is an error", True)
        _Handler.mode = "ok"

        # Empty instruction must never reach the network.
        try:
            p.edit(GenerativeRequest(image=img, instruction="  ", source_path="/tmp/a"))
            check("empty instruction is refused", False)
        except GenerativeEditError:
            check("empty instruction is refused", True)
    finally:
        server.shutdown()

    # A server that is not running must say so usefully.
    dead = LocalServerProvider("http://127.0.0.1:1/v1/images/edits")
    try:
        dead.edit(GenerativeRequest(image=np.zeros((4, 4, 3), np.uint8), instruction="x", source_path="/tmp/a"))
        check("unreachable server hints at the cause", False)
    except GenerativeEditError as exc:
        check("unreachable server hints at the cause", "running" in str(exc).lower(), str(exc))

    # --- wiring ---
    for kind in ("local_server", "mlx"):
        prov = make_provider({"provider": kind})
        check(f"make_provider('{kind}') builds it", type(prov).__name__ == "LocalServerProvider")
    prov = make_provider({"provider": "local_server", "server_endpoint": "http://localhost:9999/v1/images/edits"})
    check("endpoint setting is honoured", prov.endpoint.endswith(":9999/v1/images/edits"))

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
