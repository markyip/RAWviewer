"""Engine mixin part 2 of 7 for SemanticImageIndex."""
from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from io import BytesIO
from typing import Any

import numpy as np
from PIL.Image import Image as PilImage

from metadata import backend as metadata_backend
from search.backends import MobileCLIPONNXBackend, copy_bgra_into_cvpixelbuffer, resolve_mobileclip_backend
from search.indexer import (
    SEMANTIC_BATCH_TUNE_CACHE_VERSION,
    _apply_opencv_dnn_acceleration,
    _coerce_float,
    _coerce_int,
    _load_index_source_image,
    _ort_get_available_providers,
    _yunet_detect_faces_bgr,
    resolve_onnx_execution_providers,
)
from search.queries import CustomReverseGeocoder
from search.types import MetadataRow


class EnginePart2:

    def _gps_to_decimal(self, gps_vals: Any, ref: str) -> float | None:
        try:
            # If gps_vals is an IfdTag/IfdTagLite, it has a .values attribute
            vals = getattr(gps_vals, "values", gps_vals)

            # If it's a string from some backends, try to parse it
            if isinstance(vals, str):
                # Common formats: "25/1 10/1 0/1" or "25, 10, 0"
                import re
                parts = re.split(r"[\s,]+", vals.strip())
                if len(parts) >= 3:
                    parsed = []
                    for p in parts:
                        if "/" in p:
                            num, den = p.split("/", 1)
                            parsed.append(float(num) / (float(den) if float(den) != 0 else 1.0))
                        else:
                            parsed.append(float(p))
                    vals = parsed

            if not vals or not isinstance(vals, (list, tuple)) or len(vals) < 3:
                return None

            d = self._ratio_to_float(vals[0])
            m = self._ratio_to_float(vals[1])
            s = self._ratio_to_float(vals[2])

            if d is None or m is None or s is None:
                return None

            dec = float(d) + float(m) / 60.0 + float(s) / 3600.0

            ref_str = str(ref or "").strip().upper()
            if ref_str in ("S", "W"):
                dec = -dec
            return dec
        except Exception:
            return None

    def _extract_exif_brief(
        self,
        file_path: str,
        include_face: bool = False,
        file_stat: os.stat_result | None = None,
        *,
        persist_exif: bool = True,
    ) -> dict[str, object]:
        result = {
            "capture_time": "",
            "camera_model": "",
            "lens_model": "",
            "iso": 0,
            "width": 0,
            "height": 0,
            "gps_lat": None,
            "gps_lon": None,
            "gps_raw": "",
            "city": "",
            "landmark": "",
            "admin1": "",
            "country": "",
            "country_code": "",
            "face_count": None,
            "orientation": 1,
        }
        try:
            im = None
            if include_face:
                with contextlib.suppress(Exception):
                    im = _load_index_source_image(file_path, max_size=1280, qt_decode=False)

            t_exif = time.time()
            # Stop after GPS block when possible; RAW uses exifread (header-only) in auto mode.
            tags = metadata_backend.process_file_from_path(
                file_path,
                details=False,
                stop_tag="GPS GPSLongitudeRef",
            )
            dur_exif = time.time() - t_exif
            dur_dims = 0.0
            dur_geo = 0.0
            dur_cache = 0.0
            result["capture_time"] = self._tag_text(
                tags,
                "EXIF DateTimeOriginal",
                "EXIF DateTimeDigitized",
                "Image DateTime",
                "EXIF DateTime",
            )
            make = self._tag_text(tags, "Image Make")
            model = self._tag_text(tags, "Image Model", "EXIF BodySerialNumber")
            result["camera_model"] = self._clean_join(make, model)
            result["lens_model"] = self._tag_text(
                tags,
                "EXIF LensModel",
                "EXIF LensMake",
                "MakerNote LensType",
                "MakerNote Lens",
                "Image LensModel",
                "Composite LensID",
            )
            result["iso"] = self._tag_int(
                tags,
                "EXIF ISOSpeedRatings",
                "EXIF PhotographicSensitivity",
                "EXIF RecommendedExposureIndex",
                "EXIF ISO",
                "MakerNote ISO",
            )
            result["width"] = self._tag_int(
                tags,
                "EXIF ExifImageWidth",
                "EXIF ImageWidth",
                "Image ImageWidth",
                "Image Width",
                "EXIF PixelXDimension",
                "MakerNote ImageWidth",
            )
            result["height"] = self._tag_int(
                tags,
                "EXIF ExifImageLength",
                "EXIF ImageLength",
                "Image ImageLength",
                "Image Height",
                "Image Length",
                "EXIF PixelYDimension",
                "MakerNote ImageHeight",
            )
            if not result["width"] or not result["height"]:
                t_dims = time.time()
                w, h = self._pil_dimensions(file_path)
                dur_dims = time.time() - t_dims
                result["width"] = int(result["width"] or w)
                result["height"] = int(result["height"] or h)

            # Robust orientation extraction logic
            orientation = 1
            for key in ("Exif.Image.Orientation", "Exif.Photo.Orientation", "Image Orientation", "EXIF Orientation"):
                val = tags.get(key)
                if val is not None:
                    try:
                        vals = getattr(val, "values", val)
                        o = vals[0] if isinstance(vals, (list, tuple)) and vals else vals
                        if isinstance(o, int) and 1 <= o <= 8:
                            orientation = o
                            break
                    except Exception:
                        pass

                    orientation_str = str(val).strip()
                    orientation_map = {
                        'Horizontal (normal)': 1,
                        'Mirrored horizontal': 2,
                        'Rotated 180': 3,
                        'Mirrored vertical': 4,
                        'Mirrored horizontal then rotated 90 CCW': 5,
                        'Rotated 90 CW': 6,
                        'Mirrored horizontal then rotated 90 CW': 7,
                        'Rotated 90 CCW': 8
                    }
                    if orientation_str in orientation_map:
                        orientation = orientation_map[orientation_str]
                        break

                    try:
                        first = orientation_str.split()[0]
                        o = int(first)
                        if 1 <= o <= 8:
                            orientation = o
                            break
                    except Exception:
                        pass
            result["orientation"] = orientation

            # GPS Extraction
            lat = tags.get("GPS GPSLatitude") or tags.get("EXIF GPSLatitude") or tags.get("GPSLatitude")
            lon = tags.get("GPS GPSLongitude") or tags.get("EXIF GPSLongitude") or tags.get("GPSLongitude")
            lat_ref = self._tag_text(tags, "GPS GPSLatitudeRef", "EXIF GPSLatitudeRef", "GPSLatitudeRef")
            lon_ref = self._tag_text(tags, "GPS GPSLongitudeRef", "EXIF GPSLongitudeRef", "GPSLongitudeRef")

            result["gps_lat"] = self._gps_to_decimal(lat, lat_ref) if lat else None
            result["gps_lon"] = self._gps_to_decimal(lon, lon_ref) if lon else None

            if lat and lon:
                result["gps_raw"] = f"{lat_ref} {lat} | {lon_ref} {lon}"

            # Reverse Geocoding
            if result["gps_lat"] is not None and result["gps_lon"] is not None and (abs(result["gps_lat"]) > 0.001 or abs(result["gps_lon"]) > 0.001):
                # Skip (0,0) as it's often a placeholder for no-fix
                geo = self._ensure_reverse_geocoder()
                if isinstance(geo, CustomReverseGeocoder):
                    try:
                        t_geo = time.time()
                        # reverse_geocoder is not thread-safe; serialize lookups.
                        with self._rg_lock:
                            recs = geo.search(
                                [(
                                    _coerce_float(result["gps_lat"]) or 0.0,
                                    _coerce_float(result["gps_lon"]) or 0.0,
                                )],
                                mode=1,
                            )
                        dur_geo = time.time() - t_geo
                        if recs:
                            rec = recs[0] or {}
                            city_name = str(rec.get("name", "") or "")
                            alternates = str(rec.get("alternates", "") or "")
                            result["city"] = f"{city_name}, {alternates}" if alternates else city_name
                            result["landmark"] = str(rec.get("landmark", "") or "")
                            result["admin1"] = str(rec.get("admin1", "") or "")
                            cc = str(rec.get("cc", "") or "").upper()
                            result["country_code"] = cc
                            result["country"] = self._country_name_from_code(cc)
                    except Exception:
                        pass
            if include_face:
                result["face_count"] = self._detect_face_count(file_path, preloaded_im=im)

            # Automatically backfill and populate the main gallery's PersistentEXIFCache/ImageCache!
            if persist_exif:
                try:
                    orientation = result.get("orientation") or 1

                    from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER
                    from image_cache import get_image_cache

                    t_cache = time.time()
                    cache = get_image_cache()
                    cache_dict = {
                        "orientation": int(orientation),
                        "camera_make": str(tags.get("Image Make") or ""),
                        "camera_model": str(result["camera_model"]),
                        "capture_time": str(result["capture_time"]),
                        "original_width": int(result["width"] or 0),
                        "original_height": int(result["height"] or 0),
                        "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                        "exif_data": {
                            "original_width": int(result["width"] or 0),
                            "original_height": int(result["height"] or 0),
                            "orientation": int(orientation),
                            "capture_time": str(result["capture_time"]),
                            "camera_make": str(tags.get("Image Make") or ""),
                            "camera_model": str(result["camera_model"]),
                            "lens_model": str(result["lens_model"]),
                            "iso": int(result["iso"] or 0),
                            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                        }
                    }
                    # Skip the write when the shared cache already holds an
                    # equivalent, version-fresh row (e.g. populated moments earlier
                    # by gallery's bulk EXIF fetch or a prior sort/prefetch pass).
                    # A redundant INSERT OR REPLACE here holds the EXIF DB's single
                    # writer connection just as pointlessly as the per-item SQLite
                    # reads that turned out to be the real cause of gallery's
                    # multi-second metadata-rebuild stalls (see gallery_view.py's
                    # _metadata_exif_for_path) -- this is the write-side twin of
                    # that same contention, worth avoiding for the same reason.
                    existing = None
                    try:
                        existing = cache.get_exif(file_path)
                    except Exception:
                        existing = None
                    already_fresh = (
                        existing is not None
                        and not existing.get("minimal_preview_exif")
                        and existing.get("orientation") == cache_dict["orientation"]
                        and existing.get("camera_model") == cache_dict["camera_model"]
                        and existing.get("capture_time") == cache_dict["capture_time"]
                        and int(existing.get("original_width") or 0) == cache_dict["original_width"]
                        and int(existing.get("original_height") or 0) == cache_dict["original_height"]
                    )
                    if not already_fresh:
                        if file_stat is not None:
                            cache.put_exif(
                                file_path,
                                cache_dict,
                                file_size=int(file_stat.st_size),
                                file_mtime=float(file_stat.st_mtime),
                            )
                        else:
                            cache.put_exif(file_path, cache_dict)
                    dur_cache = time.time() - t_cache
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"[INDEX] Could not populate ImageCache for {file_path}: {e}")
                    dur_cache = 0.0
            else:
                dur_cache = 0.0
            total_dur = dur_exif + dur_dims + dur_geo + dur_cache
            if total_dur > 2.0:
                import logging
                logging.getLogger(__name__).warning(
                    "[INDEX] Slow metadata phases for %s: total=%.3fs exif=%.3fs dims=%.3fs geo=%.3fs cache_put=%.3fs",
                    os.path.basename(file_path),
                    total_dur,
                    dur_exif,
                    dur_dims,
                    dur_geo,
                    dur_cache,
                )
        except Exception:
            pass
        return result

    @staticmethod
    def _pil_to_cvpixelbuffer_bgra(im: PilImage, Quartz):
        """Build a BGRA CVPixelBuffer from a PIL image (no temp files)."""
        rgb = im.convert("RGB")
        width, height = rgb.size
        if width <= 0 or height <= 0:
            raise RuntimeError("invalid face-scan image size")
        status, pixel_buffer = Quartz.CVPixelBufferCreate(
            None,
            int(width),
            int(height),
            Quartz.kCVPixelFormatType_32BGRA,
            {
                Quartz.kCVPixelBufferCGImageCompatibilityKey: True,
                Quartz.kCVPixelBufferCGBitmapContextCompatibilityKey: True,
            },
            None,
        )
        if status != 0 or pixel_buffer is None:
            raise RuntimeError(f"Failed to create CVPixelBuffer: status {status}")
        rgba = np.asarray(rgb, dtype=np.uint8)
        bgra = np.empty((height, width, 4), dtype=np.uint8)
        bgra[..., 0] = rgba[..., 2]
        bgra[..., 1] = rgba[..., 1]
        bgra[..., 2] = rgba[..., 0]
        bgra[..., 3] = 255
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
        try:
            bytes_per_row = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            if not copy_bgra_into_cvpixelbuffer(
                base, bytes_per_row, bgra, height=height, width=width
            ):
                buf = base.as_buffer(bytes_per_row * height)
                row_bytes = width * 4
                for y in range(height):
                    start = y * bytes_per_row
                    buf[start : start + row_bytes] = bgra[y].tobytes()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
        return pixel_buffer

    @staticmethod
    def _vision_face_count_from_pil(im: PilImage) -> int | None:
        """Apple Vision face count from an in-memory PIL image (no temp JPEG)."""
        try:
            import Foundation
            import objc
            import Quartz
            import Vision

            from search.engine.semantic_index import SemanticImageIndex
        except Exception:
            return None

        try:
            with objc.autorelease_pool():
                request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                # Prefer CVPixelBuffer — avoids disk I/O and JPEG re-encode.
                try:
                    pixel_buffer = SemanticImageIndex._pil_to_cvpixelbuffer_bgra(im, Quartz)
                    handler = Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(
                        pixel_buffer, None
                    )
                except Exception:
                    # In-memory JPEG via NSData if pixel-buffer path fails.
                    buf = BytesIO()
                    im.convert("RGB").save(buf, format="JPEG", quality=90)
                    payload = buf.getvalue()
                    ns_data = Foundation.NSData.dataWithBytes_length_(payload, len(payload))
                    if ns_data is None:
                        return None
                    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                        ns_data, None
                    )
                ok, err = handler.performRequests_error_([request], None)
                if not ok or err is not None:
                    return None
                return len(request.results() or [])
        except Exception:
            return None

    @staticmethod
    def _detect_face_count(file_path: str, preloaded_im: PilImage | None = None) -> int:
        try:
            from search.engine.semantic_index import SemanticImageIndex

            if preloaded_im is not None:
                im = preloaded_im
            else:
                im = _load_index_source_image(
                    file_path, max_size=SemanticImageIndex.face_detection_max_edge(), qt_decode=False
                )

            if sys.platform == "darwin":
                count = SemanticImageIndex._vision_face_count_from_pil(im)
                return int(count or 0)

            # 1. Try OpenCV YuNet (Ultra-Lightweight, extremely fast, highly accurate)
            try:
                import cv2

                img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                return _yunet_detect_faces_bgr(img_bgr)
            except Exception as yn_err:
                import logging

                logging.getLogger(__name__).debug(
                    f"[VISION] OpenCV YuNet failed on {file_path}, trying OpenCV DNN: {yn_err}"
                )

            # 2. Windows fallback using OpenCV DNN Face Detector
            try:
                import threading

                import cv2

                global _FACE_DETECTOR_NET
                global _FACE_DETECTOR_LOCK
                if "_FACE_DETECTOR_NET" not in globals():
                    _FACE_DETECTOR_NET = None
                    _FACE_DETECTOR_LOCK = threading.Lock()

                if _FACE_DETECTOR_NET is None:
                    with _FACE_DETECTOR_LOCK:
                        if _FACE_DETECTOR_NET is None:
                            models_dir = os.path.expanduser("~/.rawviewer_cache/models")
                            os.makedirs(models_dir, exist_ok=True)

                            prototxt_path = os.path.join(models_dir, "deploy.prototxt")
                            caffemodel_path = os.path.join(
                                models_dir, "res10_300x300_ssd_iter_140000.caffemodel"
                            )

                            if not os.path.exists(prototxt_path):
                                import logging

                                logging.getLogger(__name__).info(
                                    "[VISION] Downloading DNN face detector prototxt..."
                                )
                                from core.ssl_certs import urlretrieve

                                urlretrieve(
                                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                                    prototxt_path,
                                )
                            if not os.path.exists(caffemodel_path):
                                import logging

                                logging.getLogger(__name__).info(
                                    "[VISION] Downloading DNN face detector weights..."
                                )
                                from core.ssl_certs import urlretrieve

                                urlretrieve(
                                    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                                    caffemodel_path,
                                )
                            _FACE_DETECTOR_NET = cv2.dnn.readNetFromCaffe(
                                prototxt_path, caffemodel_path
                            )
                            _apply_opencv_dnn_acceleration(_FACE_DETECTOR_NET)

                img_bgr = np.array(im.convert("RGB"))[:, :, ::-1].copy()
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(img_bgr, (600, 600)),
                    1.0,
                    (600, 600),
                    (104.0, 177.0, 123.0),
                )

                with _FACE_DETECTOR_LOCK:
                    _FACE_DETECTOR_NET.setInput(blob)
                    detections = _FACE_DETECTOR_NET.forward()

                face_count = 0
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.85:
                        face_count += 1
                return face_count
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"[VISION] OpenCV DNN face detection fallback error on {file_path}: {e}"
                )
                return 0
        except Exception:
            return 0

    def _lookup_index_rows(self, file_path: str, st: os.stat_result | None = None) -> list[sqlite3.Row]:
        canonical = self._canonical_path(file_path)
        if st is None:
            st = os.stat(canonical)
        signature = self._file_signature_from_stat(canonical, st)
        aliases = self._path_aliases(canonical)
        placeholders = ",".join(["?"] * len(aliases))
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            f"""
            SELECT file_path, file_name, file_signature, file_size, file_mtime, mtime_ns, model_name, face_count
            FROM semantic_index
            WHERE (file_signature = ? AND model_name = ?)
               OR file_path IN ({placeholders})
            """,
            [signature, self.model_name, *aliases],
        ).fetchall()

    def _row_matches_file(self, row: sqlite3.Row, st: os.stat_result) -> bool:
        if str(row["model_name"]) != self.model_name:
            return False
        if int(row["file_size"]) != int(st.st_size):
            return False
        row_mtime_ns = row.get("mtime_ns", None)
        if row_mtime_ns is not None:
            try:
                return int(row_mtime_ns) == self._mtime_ns_from_stat(st)
            except Exception:
                pass
        return self._mtime_matches(_coerce_float(row["file_mtime"]) or 0.0, st)

    def _row_semantic_ready(self, row: MetadataRow) -> bool:
        try:
            return _coerce_int(row["semantic_ready"]) == 1
        except Exception:
            # Backward compatibility for rows loaded before migration.
            return True

    @staticmethod
    def _row_semantic_skipped(row: sqlite3.Row) -> bool:
        """Permanent skip for this file revision (decode failed); do not block index completion."""
        try:
            return _coerce_int(row["semantic_ready"]) == -1
        except Exception:
            return False

    def _needs_reindex(self, file_path: str, st: os.stat_result) -> bool:
        rows = self._lookup_index_rows(file_path, st)
        if not rows:
            return True
        for row in rows:
            if self._row_matches_file(row, st) and (
                self._row_semantic_ready(row) or self._row_semantic_skipped(row)
            ):
                return False
        return True

    def _mark_semantic_skipped(
        self,
        canonical_fp: str,
        st: os.stat_result,
        conn: sqlite3.Connection,
    ) -> None:
        """Mark file as non-indexable for this model revision (won't retry until file changes)."""
        conn.execute(
            """
            UPDATE semantic_index
            SET semantic_ready = -1, dim = 0, embedding = ?, updated_at = ?
            WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
            """,
            (
                b"",
                float(time.time()),
                canonical_fp,
                self.model_name,
                int(st.st_size),
                self._mtime_ns_from_stat(st),
            ),
        )

    def _encode_image(self, file_path: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_image(file_path)
        model = self._ensure_model()
        try:
            im = _load_index_source_image(file_path, max_size=1024, qt_decode=False)
            emb = model.encode(im, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
            ) from exc

    def _encode_images_best_effort(self, file_paths: Sequence[str]) -> list[np.ndarray]:
        """
        Encode multiple images. Falls back to per-image encoding when batch path
        is unavailable or unsupported by the model graph.
        """
        paths = [p for p in (file_paths or []) if p]
        if not paths:
            return []
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            if hasattr(backend, "encode_images"):
                try:
                    return list(backend.encode_images(paths))
                except Exception:
                    pass
        return [self._encode_image(p) for p in paths]

    def _semantic_batch_cache_key(self, accel_detail: str = "") -> str:
        ep = ""
        try:
            if self.model_name.startswith("mobileclip-"):
                backend = self._mobileclip_backend or resolve_mobileclip_backend()
                self._mobileclip_backend = backend
                if isinstance(backend, MobileCLIPONNXBackend):
                    selected = resolve_onnx_execution_providers(
                        _ort_get_available_providers()
                    )
                    ep = ",".join(selected)
                else:
                    ep = backend.__class__.__name__
        except Exception:
            ep = accel_detail or "unknown"
        return (
            f"{sys.platform}|{self.model_name}|{ep}|{accel_detail or ''}"
            f"|{SEMANTIC_BATCH_TUNE_CACHE_VERSION}"
        )
