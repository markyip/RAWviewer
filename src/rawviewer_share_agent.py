"""Share the current photo(s) to an AI agent: preview JPEG + ticket, never the RAW.

Editing instructions are **per photo** (or a group override for this share),
not an app-wide style preset. Persistence is a small JSON sidecar next to the
file (``DSC01234.ARW.rawviewer-ai.json``). Catalog XMP already owns rating /
label / ``dc:subject`` keywords and does **not** write ``dc:description``;
using that field would collide with camera/IPTC captions, so this sidecar is
the source of truth.

QSettings ``ai_share/style_brief`` is only an optional *template* the user
can insert — it is not applied to every share.

Qt-free so tests can round-trip tickets without a GUI.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence

SETTINGS_TEMPLATE_KEY = "ai_share/style_brief"
DESCRIPTION_MAX = 4096
SIDECAR_SUFFIX = ".rawviewer-ai.json"

MODE_MCP_EDIT = "mcp_edit"
MODE_GENERATE = "generate"
SHARE_MODES = (MODE_MCP_EDIT, MODE_GENERATE)

INSTRUCTIONS_MCP_EDIT = (
    "Mode: mcp_edit. Use RAWviewer MCP only: get_photo, get_preview, "
    "set_adjustments, auto_adjust, export_jpeg. Non-destructive XMP. "
    "Follow the description for this photo or group (how to grade). "
    "Do not generate pixels, delete files, or overwrite the RAW."
)
INSTRUCTIONS_GENERATE = (
    "Mode: generate. Use the attached/preview JPEG plus the description as "
    "the brief. You (the model) generate a NEW derivative image — RAWviewer "
    "is not running a generative model. Save as a new JPEG next to the source "
    "(e.g. {name}_gen.jpg) via MCP save_derivative, or write JPEG/WebP under "
    "the folder root. Do not overwrite the RAW."
)
DEFAULT_INSTRUCTIONS = INSTRUCTIONS_MCP_EDIT

TICKET_FILENAME = "ticket.json"
PROMPT_FILENAME = "prompt.md"

# Back-compat aliases (template clamp / old call sites).
STYLE_BRIEF_MAX = DESCRIPTION_MAX
NOTE_MAX = DESCRIPTION_MAX
SETTINGS_STYLE_BRIEF_KEY = SETTINGS_TEMPLATE_KEY


class SharePathError(ValueError):
    """Source photo is missing or outside the folder root."""


def _norm_real(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def is_within_root(root: str, path: str) -> bool:
    root_n = _norm_real(root)
    path_n = _norm_real(path)
    if path_n == root_n:
        return True
    return path_n.startswith(root_n + os.sep)


def folder_root_for_photo(source_path: str, current_folder: str | None = None) -> str:
    """Folder MCP should sandbox to: open folder if the photo is inside it."""
    source = os.path.abspath(source_path)
    if current_folder:
        folder = os.path.abspath(current_folder)
        if os.path.isdir(folder) and is_within_root(folder, source):
            return _norm_real(folder)
    parent = os.path.dirname(source)
    if not parent or not os.path.isdir(parent):
        raise SharePathError(f"Photo folder is missing: {source_path}")
    return _norm_real(parent)


def folder_root_for_photos(
    source_paths: Sequence[str], current_folder: str | None = None
) -> str:
    paths = [os.path.abspath(p) for p in source_paths if p]
    if not paths:
        raise SharePathError("No photos to share")
    if current_folder:
        folder = os.path.abspath(current_folder)
        if os.path.isdir(folder) and all(is_within_root(folder, p) for p in paths):
            return _norm_real(folder)
    return folder_root_for_photo(paths[0], current_folder)


def assert_source_in_folder(source_path: str, folder_root: str) -> str:
    if not source_path or not str(source_path).strip():
        raise SharePathError("Source path is required")
    if not os.path.isfile(source_path):
        raise SharePathError(f"Source photo does not exist: {source_path}")
    if not folder_root or not os.path.isdir(folder_root):
        raise SharePathError(f"Folder root is not a directory: {folder_root}")
    resolved = _norm_real(source_path)
    if not is_within_root(folder_root, resolved):
        raise SharePathError("Source photo is outside the folder root")
    return resolved


def clamp_description(text: str | None) -> str:
    return str(text or "").strip()[:DESCRIPTION_MAX]


def normalize_share_mode(mode: str | None) -> str:
    text = str(mode or MODE_MCP_EDIT).strip().lower()
    if text in ("generate", "gen", "image_gen", "image-generation"):
        return MODE_GENERATE
    return MODE_MCP_EDIT


def instructions_for_mode(mode: str | None) -> str:
    if normalize_share_mode(mode) == MODE_GENERATE:
        return INSTRUCTIONS_GENERATE
    return INSTRUCTIONS_MCP_EDIT


def suggested_derivative_path(source_path: str) -> str:
    raw = os.path.abspath(source_path)
    if os.path.lexists(raw):
        raw = _norm_real(raw)
    stem, _ext = os.path.splitext(raw)
    return stem + "_gen.jpg"


def generate_hook_status() -> tuple[bool, str]:
    """Generate is the agent's job; we only need a Cursor/agent handoff."""
    if find_cursor_agent() or find_cursor_bin():
        return True, ""
    if sys.platform == "darwin" and os.path.isdir("/Applications/Cursor.app"):
        return True, ""
    return False, "No Cursor / agent hook on this Mac"


def clamp_style_brief(text: str | None) -> str:
    return clamp_description(text)


def clamp_note(text: str | None) -> str:
    return clamp_description(text)


def description_template_from_store(store: Any) -> str:
    """Optional insert-only template (QSettings or dict). Not auto-applied."""
    raw = ""
    if store is None:
        raw = ""
    elif hasattr(store, "value"):
        raw = store.value(SETTINGS_TEMPLATE_KEY, "") or ""
    elif isinstance(store, Mapping):
        raw = store.get(SETTINGS_TEMPLATE_KEY, "") or ""
    return clamp_description(str(raw))


def save_description_template_to_store(store: Any, text: str | None) -> str:
    value = clamp_description(text)
    if store is None:
        return value
    if hasattr(store, "setValue"):
        store.setValue(SETTINGS_TEMPLATE_KEY, value)
        sync = getattr(store, "sync", None)
        if callable(sync):
            sync()
    elif isinstance(store, MutableMapping):
        store[SETTINGS_TEMPLATE_KEY] = value
    return value


def style_brief_from_store(store: Any) -> str:
    return description_template_from_store(store)


def save_style_brief_to_store(store: Any, text: str | None) -> str:
    return save_description_template_to_store(store, text)


def ai_sidecar_path(photo_path: str) -> str:
    """``photo.ext.rawviewer-ai.json`` next to the file (not the XMP sidecar)."""
    return os.path.abspath(photo_path) + SIDECAR_SUFFIX


def load_photo_description(photo_path: str) -> str:
    sidecar = ai_sidecar_path(photo_path)
    if not os.path.isfile(sidecar):
        return ""
    try:
        with open(sidecar, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return ""
    if isinstance(data, dict):
        return clamp_description(data.get("description"))
    return ""


def save_photo_description(photo_path: str, text: str | None) -> str:
    """Write or remove the per-photo AI description sidecar. Does not touch XMP."""
    value = clamp_description(text)
    sidecar = ai_sidecar_path(photo_path)
    if not value:
        try:
            if os.path.isfile(sidecar):
                os.remove(sidecar)
        except OSError:
            pass
        return ""
    payload = {"description": value}
    dest_dir = os.path.dirname(sidecar)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    tmp = sidecar + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, sidecar)
    return value


def load_photo_descriptions(photo_paths: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in photo_paths:
        if not path:
            continue
        key = os.path.abspath(path)
        out[key] = load_photo_description(key)
    return out


def suggest_share_description(photo_paths: Sequence[str]) -> str:
    """Prefill: the shared sidecar text if every member has the same non-empty one."""
    texts = [load_photo_description(p) for p in photo_paths if p]
    nonempty = [t for t in texts if t]
    if not texts or not nonempty:
        return ""
    first = nonempty[0]
    if all(t == first for t in texts):
        return first
    return ""


def member_description_lines(photo_paths: Sequence[str], *, limit: int = 6) -> list[str]:
    lines: list[str] = []
    for path in photo_paths:
        if not path:
            continue
        text = load_photo_description(path)
        if not text:
            continue
        name = os.path.basename(path)
        one = " ".join(text.split())
        if len(one) > 100:
            one = one[:97] + "…"
        lines.append(f"{name}: {one}")
        if len(lines) >= limit:
            break
    return lines


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if os.path.isdir(os.path.join(parent, "src")):
        return parent
    return here


def mcp_module_available() -> bool:
    try:
        import rawviewer_mcp  # noqa: F401

        return True
    except Exception:
        return os.path.isdir(os.path.join(repo_root(), "src", "rawviewer_mcp"))


def mcp_launch_command(folder_root: str) -> str:
    """Documented stdio launch; Cursor should spawn this, not a detached orphan."""
    root = repo_root()
    folder = os.path.abspath(folder_root)
    q_root = shlex.quote(root)
    q_folder = shlex.quote(folder)
    pixi = shutil.which("pixi")
    if pixi and os.path.isfile(os.path.join(root, "pixi.toml")):
        return (
            f"cd {q_root} && PYTHONPATH=src RAWVIEWER_MCP_ROOT={q_folder} "
            f"pixi run python3 -m rawviewer_mcp --root {q_folder}"
        )
    py = shlex.quote(sys.executable or "python3")
    return (
        f"cd {q_root} && PYTHONPATH=src RAWVIEWER_MCP_ROOT={q_folder} "
        f"{py} -m rawviewer_mcp --root {q_folder}"
    )


def cache_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "rawviewer_ai_share")
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_old_cache(max_age_sec: float = 48 * 3600) -> None:
    root = cache_dir()
    now = time.time()
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        try:
            if now - os.path.getmtime(path) > max_age_sec:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        except OSError:
            pass


def new_share_dir(stem: str = "photo") -> str:
    _cleanup_old_cache()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (stem or "photo"))[:40]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(cache_dir(), f"{safe}_{stamp}_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


def strip_jpeg_gps(path: str) -> None:
    """Re-encode JPEG without EXIF so GPS is not attached to the agent preview."""
    if not path or not os.path.isfile(path):
        return
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            im.save(path, format="JPEG", quality=90, optimize=True)
    except Exception:
        pass


def bake_preview_jpeg(src_path: str, dest_path: str) -> bool:
    """Write an attachable JPEG. Never copies RAW bytes to dest."""
    if not src_path or not os.path.isfile(src_path):
        return False
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    try:
        from share_payload import needs_jpeg_proxy, write_jpeg_proxy

        if needs_jpeg_proxy(src_path):
            ok = bool(write_jpeg_proxy(src_path, dest_path))
        else:
            ok = _reencode_image_jpeg(src_path, dest_path)
        if not ok or not os.path.isfile(dest_path) or os.path.getsize(dest_path) <= 0:
            return False
        strip_jpeg_gps(dest_path)
        return is_jpeg_file(dest_path)
    except Exception:
        return False


def _reencode_image_jpeg(src_path: str, dest_path: str) -> bool:
    try:
        from PIL import Image, ImageOps

        with Image.open(src_path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            elif im.mode == "L":
                im = im.convert("RGB")
            im.save(dest_path, format="JPEG", quality=90, optimize=True)
        return os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False


def is_jpeg_file(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\xff\xd8"
    except OSError:
        return False


def _relpath_under(source: str, folder: str) -> str:
    try:
        rel = os.path.relpath(source, folder)
    except ValueError:
        rel = os.path.basename(source)
    return rel.replace("\\", "/")


def build_ticket(
    *,
    source_path: str,
    preview_path: str,
    folder_root: str | None = None,
    current_folder: str | None = None,
    description: str | None = "",
    photo_description: str | None = None,
    note: str | None = None,
    instructions: str | None = None,
    mode: str | None = MODE_MCP_EDIT,
) -> dict[str, Any]:
    """One-photo share ticket. ``description`` is what the agent should follow."""
    source = os.path.abspath(source_path)
    folder = folder_root or folder_root_for_photo(source, current_folder)
    source = assert_source_in_folder(source, folder)
    preview = os.path.abspath(preview_path)
    if not is_jpeg_file(preview):
        raise SharePathError("Preview JPEG is required; RAW bytes are not shared")
    desc = clamp_description(description)
    if not desc:
        desc = clamp_description(note)
    stored = (
        clamp_description(photo_description)
        if photo_description is not None
        else load_photo_description(source)
    )
    share_mode = normalize_share_mode(mode)
    instr = (instructions or instructions_for_mode(share_mode)).strip()
    return {
        "source_path": source,
        "preview_path": preview,
        "folder_root": os.path.abspath(folder),
        "description": desc,
        "photo_description": stored,
        "mode": share_mode,
        "instructions": instr,
        "derivative_path": suggested_derivative_path(source),
        "source_relpath": _relpath_under(source, folder),
        "mcp_command": mcp_launch_command(folder),
        "mcp_available": mcp_module_available(),
    }


def build_share_tickets(
    items: Sequence[tuple[str, str]],
    *,
    description: str,
    folder_root: str | None = None,
    current_folder: str | None = None,
    instructions: str | None = None,
    mode: str | None = MODE_MCP_EDIT,
) -> list[dict[str, Any]]:
    """Tickets for a share batch. Every ticket gets the same share ``description``."""
    if not items:
        raise SharePathError("No photos to share")
    sources = [src for src, _prev in items]
    folder = folder_root or folder_root_for_photos(sources, current_folder)
    desc = clamp_description(description)
    share_mode = normalize_share_mode(mode)
    instr = instructions or instructions_for_mode(share_mode)
    tickets: list[dict[str, Any]] = []
    for source, preview in items:
        tickets.append(
            build_ticket(
                source_path=source,
                preview_path=preview,
                folder_root=folder,
                description=desc,
                instructions=instr,
                mode=share_mode,
            )
        )
    return tickets


def build_batch_ticket(tickets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not tickets:
        raise SharePathError("No photos to share")
    first = dict(tickets[0])
    desc = clamp_description(first.get("description"))
    folder = str(first.get("folder_root") or "")
    share_mode = normalize_share_mode(first.get("mode"))
    return {
        "description": desc,
        "folder_root": folder,
        "mode": share_mode,
        "instructions": first.get("instructions") or instructions_for_mode(share_mode),
        "mcp_command": first.get("mcp_command") or mcp_launch_command(folder),
        "mcp_available": bool(first.get("mcp_available")),
        "photos": [dict(t) for t in tickets],
    }


def render_prompt_markdown(ticket: Mapping[str, Any]) -> str:
    desc = str(ticket.get("description") or "").strip() or "(none — ask the user)"
    photos = ticket.get("photos")
    if not isinstance(photos, list) or not photos:
        photos = [ticket]
    share_mode = normalize_share_mode(ticket.get("mode"))
    mcp = str(
        ticket.get("mcp_command")
        or mcp_launch_command(str(ticket.get("folder_root") or ""))
    )
    if share_mode == MODE_GENERATE:
        loop = (
            "Use the preview JPEG(s) below as image input. Follow the "
            "description. Write a new JPEG/WebP with MCP `save_derivative` "
            "(suggested `{name}_gen.jpg` next to each source). Do not "
            "overwrite the RAW. RAWviewer is not running a generative model."
        )
        title = "# RAWviewer — generate a new image\n"
    else:
        loop = (
            "Typical loop: `get_photo` → `get_preview` → `set_adjustments` or "
            "`auto_adjust` → `get_preview`. Optional `export_jpeg` (new file, "
            "do not replace the original). No generative pixels; sliders via XMP only.\n"
            "The GUI does not live-reload sidecars. The user re-opens the photo "
            "to see agent edits."
        )
        title = "# RAWviewer — edit this photo (MCP)\n"
    lines = [
        title,
        f"mode: `{share_mode}`\n",
        f"{ticket.get('instructions') or instructions_for_mode(share_mode)}\n",
        f"{loop}\n",
        "## Description (this share)\n",
        f"{desc}\n",
        "## Paths\n",
    ]
    for i, photo in enumerate(photos, start=1):
        stored = str(photo.get("photo_description") or "").strip()
        extra = f"\n  - saved on photo: {stored}" if stored and stored != desc else ""
        deriv = photo.get("derivative_path") or suggested_derivative_path(
            str(photo.get("source_path") or "")
        )
        lines.append(
            f"{i}. `{photo.get('source_path')}`\n"
            f"  - preview JPEG: `{photo.get('preview_path')}`\n"
            f"  - MCP path: `{photo.get('source_relpath') or photo.get('source_path')}`\n"
            f"  - derivative (generate): `{deriv}`"
            f"{extra}\n"
        )
    lines.extend(
        [
            f"- folder root (`RAWVIEWER_MCP_ROOT`): `{ticket.get('folder_root')}`\n",
            "## MCP launch\n",
            f"```\n{mcp}\n```\n",
        ]
    )
    return "\n".join(lines)


def write_ticket_sidecars(ticket: Mapping[str, Any], dest_dir: str) -> dict[str, str]:
    os.makedirs(dest_dir, exist_ok=True)
    ticket_path = os.path.join(dest_dir, TICKET_FILENAME)
    prompt_path = os.path.join(dest_dir, PROMPT_FILENAME)
    with open(ticket_path, "w", encoding="utf-8") as fh:
        json.dump(dict(ticket), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    photos = ticket.get("photos") if isinstance(ticket.get("photos"), list) else None
    extra: list[str] = []
    if photos and len(photos) > 1:
        for i, photo in enumerate(photos, start=1):
            per = os.path.join(dest_dir, f"ticket_{i:02d}.json")
            with open(per, "w", encoding="utf-8") as fh:
                json.dump(dict(photo), fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            extra.append(per)
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(render_prompt_markdown(ticket))
    out = {"ticket_path": ticket_path, "prompt_path": prompt_path}
    if extra:
        out["per_photo_tickets"] = extra
    return out


def persist_share_descriptions(
    source_paths: Sequence[str], description: str, *, save_on_photos: bool
) -> None:
    if not save_on_photos:
        return
    text = clamp_description(description)
    for path in source_paths:
        if path and os.path.isfile(path):
            save_photo_description(path, text)


def find_cursor_bin() -> Optional[str]:
    found = shutil.which("cursor")
    if found:
        return found
    extras = [
        "/usr/local/bin/cursor",
        os.path.expanduser("~/.local/bin/cursor"),
        "/Applications/Cursor.app/Contents/Resources/app/bin/cursor",
    ]
    for path in extras:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def find_cursor_agent() -> Optional[str]:
    found = shutil.which("cursor-agent")
    if found:
        return found
    extras = [
        os.path.expanduser("~/.local/bin/cursor-agent"),
        "/usr/local/bin/cursor-agent",
    ]
    for path in extras:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _agent_prompt_arg(ticket: Mapping[str, Any], prompt_path: str) -> str:
    desc = clamp_description(ticket.get("description"))
    share_mode = normalize_share_mode(ticket.get("mode"))
    parts = [
        instructions_for_mode(share_mode),
        f"Read {prompt_path} and the preview JPEG(s).",
        f"Folder root: {ticket.get('folder_root')}.",
        f"mode: {share_mode}.",
    ]
    if desc:
        parts.append(f"Description: {desc}")
    photos = ticket.get("photos")
    if isinstance(photos, list) and photos:
        parts.append(f"{len(photos)} photos in this share.")
        parts.append(f"First source: {photos[0].get('source_path')}.")
    elif ticket.get("source_path"):
        parts.append(f"Source photo: {ticket.get('source_path')}.")
    return " ".join(parts)


def _spawn(cmd: list[str], *, cwd: str | None, env: dict[str, str]) -> bool:
    try:
        kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "start_new_session": True,
        }
        if sys.platform == "win32":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
            kwargs["creationflags"] = flags
            kwargs.pop("start_new_session", None)
        subprocess.Popen(cmd, **kwargs)
        return True
    except Exception:
        return False


def launch_agent(
    ticket: Mapping[str, Any],
    *,
    prompt_path: str,
    preview_path: str | Sequence[str] = "",
) -> dict[str, Any]:
    """Open Cursor / cursor-agent with preview + prompt. Clipboard is last resort."""
    folder = str(ticket.get("folder_root") or "")
    env = os.environ.copy()
    if folder:
        env["RAWVIEWER_MCP_ROOT"] = folder
    env.setdefault("RAWVIEWER_ENABLE_EDITING", "1")
    prompt_arg = _agent_prompt_arg(ticket, prompt_path)
    if isinstance(preview_path, str):
        previews = [preview_path] if preview_path else []
    else:
        previews = [p for p in preview_path if p]
    if not previews:
        photos = ticket.get("photos")
        if isinstance(photos, list):
            previews = [str(p.get("preview_path") or "") for p in photos if p.get("preview_path")]
        elif ticket.get("preview_path"):
            previews = [str(ticket.get("preview_path"))]
    open_files = [prompt_path] + previews[:8]

    agent = find_cursor_agent()
    if agent and _spawn([agent, "-b", prompt_arg], cwd=folder or None, env=env):
        cursor = find_cursor_bin()
        if cursor:
            _spawn(
                [cursor, "--new-window", *open_files],
                cwd=folder or None,
                env=env,
            )
        return {"ok": True, "method": "cursor-agent", "prompt": prompt_arg}

    cursor = find_cursor_bin()
    if cursor and _spawn(
        [cursor, "--new-window", *open_files],
        cwd=folder or None,
        env=env,
    ):
        return {"ok": True, "method": "cursor", "prompt": prompt_arg}

    if sys.platform == "darwin" and os.path.isdir("/Applications/Cursor.app"):
        if _spawn(
            ["open", "-n", "-a", "Cursor", *open_files],
            cwd=folder or None,
            env=env,
        ):
            return {"ok": True, "method": "open-cursor", "prompt": prompt_arg}

    return {"ok": False, "method": "clipboard", "prompt": render_prompt_markdown(ticket)}
