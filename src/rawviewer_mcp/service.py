"""Headless tool implementations (no Qt event loop)."""

from __future__ import annotations

import os
from typing import Any, Callable

from raw_adjustments import (
    DEFAULT_ADJUSTMENTS,
    load_adjustments_for_file,
    load_catalog_for_file,
    write_keywords_for_file,
    write_label_for_file,
    write_rating_for_file,
    write_xmp_adjustments_for_file,
)
from xmp_catalog import COLOR_LABELS, normalize_color_label

from .adjustments import clamp_adjustment_value, clamp_patch, public_adjustments
from .paths import (
    PathSandboxError,
    ensure_parent_under_root,
    list_photo_files,
    resolve_under_root,
)
from .preview import LinearBaseCache, render_preview_jpeg


class McpToolError(ValueError):
    pass


_EXPORT_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".webp"})


def _assert_safe_image_dest(dest_path: str, *, source_photo: str = "") -> None:
    """Refuse RAW/source overwrite and existing photos (export / derivative)."""
    ext = os.path.splitext(dest_path)[1].lower()
    if ext not in _EXPORT_IMAGE_EXTS:
        raise McpToolError("dest must be a .jpg/.jpeg/.webp path")
    dest_abs = os.path.abspath(dest_path)
    if source_photo:
        try:
            if os.path.normcase(os.path.realpath(dest_abs)) == os.path.normcase(
                os.path.realpath(source_photo)
            ):
                raise McpToolError("must not overwrite the source photo")
        except OSError:
            pass
    if os.path.lexists(dest_path):
        from raw_file_extensions import get_supported_extensions

        photo_exts = {e.lower() for e in get_supported_extensions()}
        if ext in photo_exts:
            raise McpToolError("refusing to overwrite an existing photo")
        raise McpToolError("refusing to overwrite an existing file")


def _exif_summary(path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from metadata_backend import process_file_from_path

        tags = process_file_from_path(path, details=False) or {}
    except Exception:
        tags = {}

    def _get(*names: str) -> str:
        for name in names:
            tag = tags.get(name)
            if tag is None:
                continue
            text = getattr(tag, "printable", None)
            if text is None:
                text = str(tag)
            text = str(text).strip()
            if text:
                return text
        return ""

    out["make"] = _get("Image Make", "EXIF Make")
    out["model"] = _get("Image Model", "EXIF Model")
    out["lens"] = _get("EXIF LensModel", "Image LensModel", "EXIF LensModel")
    out["iso"] = _get("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
    out["aperture"] = _get("EXIF FNumber")
    out["shutter"] = _get("EXIF ExposureTime")
    out["focal_length"] = _get("EXIF FocalLength")
    out["datetime"] = _get("EXIF DateTimeOriginal", "Image DateTime")
    out["orientation"] = _get("Image Orientation", "EXIF Orientation")
    # Drop empty
    return {k: v for k, v in out.items() if v}


def _require_file(root: str, path: str) -> str:
    resolved = resolve_under_root(root, path, must_exist=True)
    if not os.path.isfile(resolved):
        raise PathSandboxError(f"Not a file: {path}")
    return resolved


class RawviewerMcpService:
    def __init__(
        self,
        root: str,
        *,
        decode_fn: Callable[..., Any] | None = None,
    ) -> None:
        os.environ.setdefault("RAWVIEWER_ENABLE_EDITING", "1")
        self.root = root
        self.decode_fn = decode_fn
        self.cache = LinearBaseCache()

    def _photo(self, path: str) -> str:
        return _require_file(self.root, path)

    def _adj(self, path: str) -> dict:
        return load_adjustments_for_file(path)

    def _catalog_dict(self, path: str) -> dict[str, Any]:
        """Rating, label, and keywords from a single XMP sidecar parse."""
        rating, label, keywords = load_catalog_for_file(path)
        return {
            "rating": int(rating or 0),
            "label": label or "",
            "keywords": list(keywords or []),
        }

    def _lens_flag(self, adj: dict) -> bool:
        try:
            return float(adj.get("LensCorrectionEnabled", 0.0) or 0.0) > 0.5
        except (TypeError, ValueError):
            return False

    def _base(self, path: str, *, full: bool = False, adj: dict | None = None):
        if adj is None:
            adj = self._adj(path)
        return self.cache.get(
            path,
            use_full_resolution=full,
            apply_lens_correction=self._lens_flag(adj),
            decode_fn=self.decode_fn,
        )

    def list_photos(
        self,
        glob: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        files = list_photo_files(self.root, pattern=glob, limit=limit)
        rel = [os.path.relpath(p, self.root) for p in files]
        return {"root": self.root, "count": len(rel), "photos": rel}

    def get_photo(self, path: str) -> dict[str, Any]:
        photo = self._photo(path)
        adj = self._adj(photo)
        out: dict[str, Any] = {
            "path": os.path.relpath(photo, self.root),
            "abs_path": photo,
            "exif": _exif_summary(photo),
            "adjustments": public_adjustments(adj),
        }
        out.update(self._catalog_dict(photo))
        return out

    def get_preview(
        self,
        path: str,
        *,
        long_edge: int = 1000,
    ) -> dict[str, Any]:
        photo = self._photo(path)
        adj = self._adj(photo)
        base = self._base(photo, full=False, adj=adj)
        jpeg, w, h = render_preview_jpeg(base, adj, long_edge=int(long_edge or 1000))
        return {
            "path": os.path.relpath(photo, self.root),
            "width": w,
            "height": h,
            "mime_type": "image/jpeg",
            "jpeg_bytes": jpeg,
        }

    def set_adjustments(self, path: str, adjustments: dict[str, Any]) -> dict[str, Any]:
        photo = self._photo(path)
        patch = clamp_patch(adjustments or {})
        merged = dict(self._adj(photo))
        merged.update(patch)
        write_xmp_adjustments_for_file(photo, merged)
        return {
            "path": os.path.relpath(photo, self.root),
            "adjustments": public_adjustments(self._adj(photo)),
        }

    def auto_adjust(self, path: str) -> dict[str, Any]:
        """WB + Light + straighten; does not bake P-key Reconstruct."""
        from raw_adjustments import solve_white_balance_from_sample, wb_reference_temperature
        from raw_auto_adjust import (
            estimate_auto_tone,
            estimate_neutral_rgb,
            estimate_straighten_angle,
        )
        from raw_edit_pipeline import _linear_float_from_buffer, linear_to_display_uint8

        photo = self._photo(path)
        adj = dict(self._adj(photo))
        base = self._base(photo, full=False, adj=adj)
        lin = _linear_float_from_buffer(base)
        updates: dict[str, float] = {}
        skipped: list[str] = []
        parts: list[str] = []

        est = estimate_neutral_rgb(lin)
        if est is None:
            skipped.append("WB")
        else:
            r, g, b = est
            ref_temp = wb_reference_temperature(adj)
            temperature, tint = solve_white_balance_from_sample(r, g, b, ref_temp)
            updates["Temperature"] = clamp_patch({"Temperature": temperature})["Temperature"]
            updates["Tint"] = clamp_patch({"Tint": tint})["Tint"]
            parts.append(f"WB {int(round(temperature))}K")

        tone = estimate_auto_tone(lin)
        if tone is None:
            skipped.append("tone")
        else:
            updates.update(clamp_patch({k: float(v) for k, v in tone.items()}))
            parts.append(f"tone {float(tone.get('Exposure2012', 0.0)):+.2f} EV")

        rgb = None
        try:
            adj0 = dict(adj)
            adj0.update(updates)
            adj0["CropAngle"] = 0.0
            adj0["PerspectiveVertical"] = 0.0
            adj0["PerspectiveHorizontal"] = 0.0
            rgb = linear_to_display_uint8(base, adj0)
        except Exception:
            rgb = None
        if rgb is None:
            skipped.append("straighten")
        else:
            angle = estimate_straighten_angle(rgb)
            if angle is None:
                skipped.append("straighten")
            else:
                import numpy as np

                new_angle = float(np.clip(angle, -45.0, 45.0))
                updates["CropAngle"] = clamp_patch({"CropAngle": new_angle})["CropAngle"]
                parts.append(f"straighten {new_angle:+.1f}°")

        if updates:
            adj.update(updates)
            write_xmp_adjustments_for_file(photo, adj)
        return {
            "path": os.path.relpath(photo, self.root),
            "applied": parts,
            "skipped": skipped,
            "adjustments": public_adjustments(self._adj(photo)),
        }

    def reset_adjustments(self, path: str) -> dict[str, Any]:
        photo = self._photo(path)
        current = self._adj(photo)
        from raw_adjustments import AS_SHOT_TEMP_KEY

        reset = dict(DEFAULT_ADJUSTMENTS)
        if AS_SHOT_TEMP_KEY in current:
            reset[AS_SHOT_TEMP_KEY] = current[AS_SHOT_TEMP_KEY]
        write_xmp_adjustments_for_file(photo, reset)
        out: dict[str, Any] = {
            "path": os.path.relpath(photo, self.root),
            "adjustments": public_adjustments(self._adj(photo)),
        }
        out.update(self._catalog_dict(photo))
        return out

    def apply_preset(self, path: str, name: str) -> dict[str, Any]:
        from raw_xmp_presets import load_preset_adjustments

        photo = self._photo(path)
        if not name or not str(name).strip():
            raise McpToolError("Preset name is required")
        try:
            preset = load_preset_adjustments(str(name).strip())
        except FileNotFoundError as exc:
            raise McpToolError(f"Preset not found: {name}") from exc
        try:
            from raw_dodge_burn import MASK_KEY

            preset.pop(MASK_KEY, None)
        except Exception:
            pass
        try:
            from raw_spot_heal import MASK_KEY as HEAL_MASK_KEY

            preset.pop(HEAL_MASK_KEY, None)
        except Exception:
            pass
        from raw_adjustments import AS_SHOT_TEMP_KEY

        current = self._adj(photo)
        as_shot = current.get(AS_SHOT_TEMP_KEY)
        preset.pop(AS_SHOT_TEMP_KEY, None)
        merged = dict(DEFAULT_ADJUSTMENTS)
        merged.update(preset)
        if as_shot is not None:
            merged[AS_SHOT_TEMP_KEY] = as_shot
        for key in list(DEFAULT_ADJUSTMENTS):
            if key not in merged:
                continue
            try:
                merged[key] = clamp_adjustment_value(key, merged[key])
            except ValueError:
                merged[key] = DEFAULT_ADJUSTMENTS[key]
        write_xmp_adjustments_for_file(photo, merged)
        return {
            "path": os.path.relpath(photo, self.root),
            "preset": os.path.basename(str(name)),
            "adjustments": public_adjustments(self._adj(photo)),
        }

    def set_catalog(
        self,
        path: str,
        *,
        rating: int | None = None,
        label: str | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        photo = self._photo(path)
        if rating is not None:
            r = int(rating)
            if r < 0 or r > 5:
                raise McpToolError("rating must be 0–5")
            write_rating_for_file(photo, r)
        if label is not None:
            text = str(label).strip()
            if text:
                norm = normalize_color_label(text)
                if norm not in COLOR_LABELS:
                    raise McpToolError(
                        "label must be one of: " + ", ".join(COLOR_LABELS) + " (or empty)"
                    )
                write_label_for_file(photo, norm)
            else:
                write_label_for_file(photo, "")
        if keywords is not None:
            if isinstance(keywords, str):
                words = [w.strip() for w in keywords.split(",") if w.strip()]
            else:
                words = [str(w).strip() for w in keywords if str(w).strip()]
            write_keywords_for_file(photo, words)
        out: dict[str, Any] = {"path": os.path.relpath(photo, self.root)}
        out.update(self._catalog_dict(photo))
        return out

    def export_jpeg(
        self,
        path: str,
        dest: str,
        *,
        strip_gps: bool = True,
    ) -> dict[str, Any]:
        from raw_edit_pipeline import export_adjusted_image

        photo = self._photo(path)
        dest_path = ensure_parent_under_root(self.root, dest)
        _assert_safe_image_dest(dest_path, source_photo=photo)
        parent = os.path.dirname(dest_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
            if not os.path.isdir(parent) or not (
                os.path.normcase(os.path.realpath(parent)).startswith(
                    os.path.normcase(os.path.realpath(self.root)) + os.sep
                )
            ):
                raise PathSandboxError("Could not create export directory under root")
        adj = self._adj(photo)
        base = self._base(photo, full=True, adj=adj)
        extra: dict[str, Any] = {
            "output_path": dest_path,
            "rgb_linear": base,
            "adj": adj,
            "strip_gps": bool(strip_gps),
            "watermark_text": "",
            "source_path": photo,
            "copy_metadata": True,
        }
        export_adjusted_image("jpeg", **extra)
        return {
            "path": os.path.relpath(photo, self.root),
            "dest": os.path.relpath(dest_path, self.root),
            "strip_gps": bool(strip_gps),
        }

    def save_derivative(
        self,
        dest: str,
        image_base64: str,
        *,
        source: str = "",
    ) -> dict[str, Any]:
        """Write a new JPEG/WebP under the root. Never overwrites the source/RAW."""
        import base64

        dest_text = str(dest or "").strip()
        source_photo = ""
        if source:
            source_photo = self._photo(source)
        if not dest_text:
            if not source_photo:
                raise McpToolError(
                    "dest is required (or pass source to use {name}_gen.jpg)"
                )
            dest_text = os.path.splitext(source_photo)[0] + "_gen.jpg"
        dest_path = ensure_parent_under_root(self.root, dest_text)
        _assert_safe_image_dest(dest_path, source_photo=source_photo)
        ext = os.path.splitext(dest_path)[1].lower()
        raw = str(image_base64 or "").strip()
        if raw.startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw)
        except Exception as exc:
            raise McpToolError(f"image_base64 is not valid base64: {exc}") from exc
        if not data:
            raise McpToolError("image data is empty")
        max_bytes = 25 * 1024 * 1024
        if len(data) > max_bytes:
            raise McpToolError("image exceeds size cap (25 MB)")
        if ext in {".jpg", ".jpeg"} and data[:2] != b"\xff\xd8":
            raise McpToolError("payload is not a JPEG")
        if ext == ".webp" and not (data[:4] == b"RIFF" and b"WEBP" in data[:16]):
            raise McpToolError("payload is not a WebP")
        parent = os.path.dirname(dest_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
            from .paths import is_within_root

            if not is_within_root(self.root, parent):
                raise PathSandboxError("Could not create derivative directory under root")
        with open(dest_path, "wb") as fh:
            fh.write(data)
        return {
            "dest": os.path.relpath(dest_path, self.root),
            "bytes": len(data),
        }
