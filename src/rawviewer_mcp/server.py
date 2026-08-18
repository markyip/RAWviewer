"""MCP over stdio. Prefers the official SDK; otherwise a minimal JSON-RPC server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

from . import __version__
from .paths import ENV_ROOT, PathSandboxError, resolve_root
from .service import McpToolError, RawviewerMcpService

# Stdio MCP must not write logs to stdout.
def _log(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_photos",
        "description": (
            "List photo files under the rooted folder (RAW + JPEG/TIFF/etc. "
            "the app already treats as photos). Optional glob and limit."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": "Optional glob relative to the folder root, e.g. '**/*.CR3'",
                },
                "limit": {"type": "integer", "description": "Max number of paths to return"},
            },
        },
    },
    {
        "name": "get_photo",
        "description": (
            "EXIF summary, rating, color label, keywords, and current adjustment "
            "dict (numeric sliders + notable string keys). Does not modify files."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Photo path relative to the folder root"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_preview",
        "description": (
            "JPEG of the current edit (or as-shot if defaults). Long edge ~800–1200. "
            "Call after list/get and again after set_adjustments or auto_adjust."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "long_edge": {
                    "type": "integer",
                    "description": "Preview long edge in pixels (default 1000)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "set_adjustments",
        "description": (
            "Partial patch of slider keys, clamped to SLIDER_SPECS ranges, written "
            "to the XMP sidecar. Never writes RAW pixels. No brush/heal masks in v1."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "adjustments": {
                    "type": "object",
                    "description": "Slider key → numeric value (partial patch)",
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["path", "adjustments"],
        },
    },
    {
        "name": "auto_adjust",
        "description": (
            "Same Auto as the app: white balance + Light (PV2012) + straighten. "
            "Does not bake P-key Reconstruct. Persists to XMP."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "reset_adjustments",
        "description": (
            "Write default sliders to XMP. Keeps rating, color label, and keywords."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "apply_preset",
        "description": "Apply a named XMP preset from the managed preset library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string", "description": "Preset basename, e.g. look.xmp"},
            },
            "required": ["path", "name"],
        },
    },
    {
        "name": "set_catalog",
        "description": "Set rating (0–5), Lightroom color label, and/or keywords on the sidecar.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "rating": {"type": "integer", "minimum": 0, "maximum": 5},
                "label": {
                    "type": "string",
                    "description": "Red, Yellow, Green, Blue, Purple, or empty to clear",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "export_jpeg",
        "description": (
            "Bake a JPEG of the current edit to a path inside the folder root. "
            "Does not overwrite the original. GPS stripped by default (matches the app)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "dest": {
                    "type": "string",
                    "description": "Destination JPEG path relative to the folder root",
                },
                "strip_gps": {"type": "boolean", "default": True},
            },
            "required": ["path", "dest"],
        },
    },
    {
        "name": "save_derivative",
        "description": (
            "Save a generated JPEG/WebP as a NEW file under the folder root "
            "(e.g. {name}_gen.jpg). Never overwrites the RAW/source. Size cap 25 MB. "
            "Pass dest, or source to default to {name}_gen.jpg next to the photo."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dest": {
                    "type": "string",
                    "description": "Destination path relative to the folder root (.jpg/.webp)",
                },
                "image_base64": {
                    "type": "string",
                    "description": "JPEG or WebP bytes, base64-encoded",
                },
                "source": {
                    "type": "string",
                    "description": "Original photo; used to refuse overwrite and default dest",
                },
            },
            "required": ["image_base64"],
        },
    },
]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if k != "jpeg_bytes"}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, bytes):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def dispatch_tool(svc: RawviewerMcpService, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = arguments or {}
    if name == "list_photos":
        return svc.list_photos(glob=args.get("glob"), limit=args.get("limit"))
    if name == "get_photo":
        return svc.get_photo(args.get("path", ""))
    if name == "get_preview":
        return svc.get_preview(args.get("path", ""), long_edge=int(args.get("long_edge") or 1000))
    if name == "set_adjustments":
        return svc.set_adjustments(args.get("path", ""), args.get("adjustments") or {})
    if name == "auto_adjust":
        return svc.auto_adjust(args.get("path", ""))
    if name == "reset_adjustments":
        return svc.reset_adjustments(args.get("path", ""))
    if name == "apply_preset":
        return svc.apply_preset(args.get("path", ""), args.get("name", ""))
    if name == "set_catalog":
        return svc.set_catalog(
            args.get("path", ""),
            rating=args.get("rating"),
            label=args.get("label"),
            keywords=args.get("keywords"),
        )
    if name == "export_jpeg":
        strip = args.get("strip_gps")
        if strip is None:
            strip = True
        return svc.export_jpeg(args.get("path", ""), args.get("dest", ""), strip_gps=bool(strip))
    if name == "save_derivative":
        return svc.save_derivative(
            str(args.get("dest") or ""),
            str(args.get("image_base64") or ""),
            source=str(args.get("source") or ""),
        )
    raise McpToolError(f"Unknown tool: {name}")


def _preview_content(result: dict[str, Any]) -> list[dict[str, Any]]:
    jpeg = result.get("jpeg_bytes") or b""
    meta = _json_safe(result)
    import base64

    return [
        {"type": "text", "text": json.dumps(meta, ensure_ascii=False)},
        {
            "type": "image",
            "data": base64.b64encode(jpeg).decode("ascii"),
            "mimeType": "image/jpeg",
        },
    ]


def tool_call_result(svc: RawviewerMcpService, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        result = dispatch_tool(svc, name, arguments)
    except (PathSandboxError, McpToolError, ValueError, FileNotFoundError) as exc:
        return {
            "content": [{"type": "text", "text": str(exc)}],
            "isError": True,
        }
    except Exception as exc:
        _log(traceback.format_exc())
        return {
            "content": [{"type": "text", "text": f"Internal error: {exc}"}],
            "isError": True,
        }
    if name == "get_preview":
        return {"content": _preview_content(result), "isError": False}
    return {
        "content": [{"type": "text", "text": json.dumps(_json_safe(result), ensure_ascii=False)}],
        "isError": False,
    }


def _try_run_sdk(svc: RawviewerMcpService) -> bool:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        return False

    mcp = FastMCP("rawviewer")

    @mcp.tool(name="list_photos", description=TOOL_SCHEMAS[0]["description"])
    def list_photos(glob: str | None = None, limit: int | None = None) -> dict:
        return svc.list_photos(glob=glob, limit=limit)

    @mcp.tool(name="get_photo", description=TOOL_SCHEMAS[1]["description"])
    def get_photo(path: str) -> dict:
        return svc.get_photo(path)

    @mcp.tool(name="get_preview", description=TOOL_SCHEMAS[2]["description"])
    def get_preview(path: str, long_edge: int = 1000):
        result = svc.get_preview(path, long_edge=long_edge)
        jpeg = result.get("jpeg_bytes") or b""
        meta = _json_safe(result)
        try:
            from mcp.server.fastmcp import Image

            return [meta, Image(data=jpeg, format="jpeg")]
        except Exception:
            return _preview_content(result)

    @mcp.tool(name="set_adjustments", description=TOOL_SCHEMAS[3]["description"])
    def set_adjustments(path: str, adjustments: dict) -> dict:
        return svc.set_adjustments(path, adjustments)

    @mcp.tool(name="auto_adjust", description=TOOL_SCHEMAS[4]["description"])
    def auto_adjust(path: str) -> dict:
        return svc.auto_adjust(path)

    @mcp.tool(name="reset_adjustments", description=TOOL_SCHEMAS[5]["description"])
    def reset_adjustments(path: str) -> dict:
        return svc.reset_adjustments(path)

    @mcp.tool(name="apply_preset", description=TOOL_SCHEMAS[6]["description"])
    def apply_preset(path: str, name: str) -> dict:
        return svc.apply_preset(path, name)

    @mcp.tool(name="set_catalog", description=TOOL_SCHEMAS[7]["description"])
    def set_catalog(
        path: str,
        rating: int | None = None,
        label: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict:
        return svc.set_catalog(path, rating=rating, label=label, keywords=keywords)

    @mcp.tool(name="export_jpeg", description=TOOL_SCHEMAS[8]["description"])
    def export_jpeg(path: str, dest: str, strip_gps: bool = True) -> dict:
        return svc.export_jpeg(path, dest, strip_gps=strip_gps)

    @mcp.tool(name="save_derivative", description=TOOL_SCHEMAS[9]["description"])
    def save_derivative(
        image_base64: str, dest: str = "", source: str = ""
    ) -> dict:
        return svc.save_derivative(dest, image_base64, source=source)

    _log("rawviewer MCP: using official Python MCP SDK (FastMCP stdio)")
    mcp.run(transport="stdio")
    return True


def _read_stdio_message(stdin) -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            text = line.decode("latin-1")
        if ":" not in text:
            continue
        key, _, val = text.partition(":")
        headers[key.strip().lower()] = val.strip()
    n = int(headers.get("content-length") or "0")
    if n <= 0:
        return None
    body = stdin.read(n)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_stdio_message(stdout, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
    stdout.write(header + raw)
    stdout.flush()


def _run_jsonrpc(svc: RawviewerMcpService) -> None:
    _log("rawviewer MCP: JSON-RPC stdio (mcp package not installed)")
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            msg = _read_stdio_message(stdin)
        except Exception:
            _log(traceback.format_exc())
            continue
        if msg is None:
            break
        method = msg.get("method")
        msg_id = msg.get("id")
        if method is None or method.startswith("notifications/"):
            continue
        result: Any = None
        error: dict[str, Any] | None = None
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "rawviewer", "version": __version__},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOL_SCHEMAS}
            elif method == "tools/call":
                params = msg.get("params") or {}
                result = tool_call_result(
                    svc,
                    str(params.get("name") or ""),
                    params.get("arguments") or {},
                )
            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}
        except Exception as exc:
            _log(traceback.format_exc())
            error = {"code": -32603, "message": str(exc)}
        if msg_id is None:
            continue
        if error is not None:
            _write_stdio_message(stdout, {"jsonrpc": "2.0", "id": msg_id, "error": error})
        else:
            _write_stdio_message(stdout, {"jsonrpc": "2.0", "id": msg_id, "result": result})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAWviewer headless MCP server (stdio). Edits XMP sidecars only."
    )
    p.add_argument(
        "--root",
        default=os.environ.get(ENV_ROOT, ""),
        help=f"Folder root for all photo paths (or env {ENV_ROOT})",
    )
    p.add_argument(
        "--jsonrpc",
        action="store_true",
        help="Force the built-in JSON-RPC stdio server (skip FastMCP even if installed)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = resolve_root(args.root or None)
    except PathSandboxError as exc:
        _log(str(exc))
        return 2
    os.environ[ENV_ROOT] = root
    os.environ.setdefault("RAWVIEWER_ENABLE_EDITING", "1")
    svc = RawviewerMcpService(root)
    _log(f"rawviewer MCP root={root}")
    if not args.jsonrpc:
        try:
            if _try_run_sdk(svc):
                return 0
        except Exception:
            _log("FastMCP failed; falling back to JSON-RPC")
            _log(traceback.format_exc())
    _run_jsonrpc(svc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
