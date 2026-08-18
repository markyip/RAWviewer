# Headless MCP server

Non-destructive edits for an AI agent: **XMP sidecars + a small JPEG preview loop**. Does not start Qt/GUI. Original pixels are never overwritten.

Typical loop: `list_photos` / `get_photo` → `get_preview` → `set_adjustments` or `auto_adjust` → `get_preview` → `export_jpeg`.

## Launch

From the repo root (folder root is required):

```bash
cd /path/to/RAWviewer
PYTHONPATH=src pixi run python3 -m rawviewer_mcp --root /path/to/photos
```

Equivalent env:

```bash
export RAWVIEWER_MCP_ROOT=/path/to/photos
PYTHONPATH=src pixi run python3 -m rawviewer_mcp
```

`--jsonrpc` forces the built-in JSON-RPC stdio server. If the optional [`mcp`](https://pypi.org/project/mcp/) package is installed, the official FastMCP stdio transport is used instead.

## Cursor

Project `.cursor/mcp.json` (or Cursor **Settings → MCP**):

```json
{
  "mcpServers": {
    "rawviewer": {
      "command": "pixi",
      "args": ["run", "python3", "-m", "rawviewer_mcp", "--root", "/path/to/photos"],
      "cwd": "/path/to/RAWviewer",
      "env": {
        "PYTHONPATH": "src",
        "RAWVIEWER_MCP_ROOT": "/path/to/photos"
      }
    }
  }
}
```

All tool paths are sandboxed to that folder root (traversal is rejected).

## Tools (v1)

| Tool | Notes |
|------|--------|
| `list_photos` | RAW + JPEG/TIFF/etc. Optional `glob`, `limit`. |
| `get_photo` | EXIF summary, rating, label, keywords, adjustments. |
| `get_preview` | JPEG of the **current edit** (long edge ~1000). Image content + path/width/height. |
| `set_adjustments` | Partial slider patch → XMP. Values clamped to `SLIDER_SPECS`. |
| `auto_adjust` | App Auto: WB + Light + straighten. Does **not** bake P-key Reconstruct. |
| `reset_adjustments` | Defaults; keeps rating/label/keywords. |
| `apply_preset` | Named preset from `raw_xmp_presets` library. |
| `set_catalog` | `rating` / color `label` / `keywords`. |
| `export_jpeg` | Bake JPEG under the root. `strip_gps` default `true`. No watermark. |

No generative inpainting, no mask alpha dump, no brush/heal tools in v1.

## Known gaps

- The GUI does **not** live-reload sidecars written by MCP. Re-open the photo or restart Adjust to see agent edits.
- Preview decode uses `UnifiedImageProcessor.decode_raw_edit_base` (half-res). That module imports Qt but this server never starts `QApplication`.
- `export_jpeg` is a bake to a **new** file under the root; it does not replace the original.
