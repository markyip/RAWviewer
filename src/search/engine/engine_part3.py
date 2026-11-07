"""Engine mixin part 3 of 7 for SemanticImageIndex."""
from __future__ import annotations

import concurrent
import contextlib
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageOps

from core.errors import IndexAborted
from search.backends import MobileCLIPCoreMLBackend, MobileCLIPONNXBackend, resolve_mobileclip_backend
from search.indexer import (
    SEMANTIC_BATCH_TUNE_CACHE_VERSION,
    _cached_index_source_array,
    _coerce_float,
    _get_shared_index_executor,
    _index_thumbnail_needs_warm,
    _load_semantic_batch_cache,
    _save_semantic_batch_cache,
    _thumbnail_jpeg_bytes,
    face_scan_enabled,
    format_index_progress,
    metadata_auto_index_enabled,
    semantic_batch_auto_enabled,
    semantic_batch_candidates,
    semantic_batch_size,
    semantic_batch_size_forced,
    semantic_batch_tie_ratio,
    semantic_batch_tune_sample_count,
    semantic_coreml_batch_candidates,
    semantic_coreml_chunk_forced,
    semantic_coreml_chunk_size,
    semantic_coreml_prep_chunk_size,
    semantic_coreml_tune_early_stop_ratio,
    semantic_coreml_tune_sample_count,
    semantic_coreml_use_native_batch,
    semantic_encode_prep_workers,
    semantic_warm_thumbs_before_index,
)
from search.queries import CustomReverseGeocoder, _semantic_text_variants
from search.types import ProgressCallback


class EnginePart3:




    def _auto_select_semantic_batch_size(
        self,
        pending_for_semantic: Sequence[tuple[str, os.stat_result]],
        accel_detail: str = "",
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
    ) -> int:
        import logging

        logger = logging.getLogger(__name__)
        total = len(pending_for_semantic or [])
        if total <= 1:
            return 1

        is_coreml = False
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            self._mobileclip_backend = backend
            is_coreml = isinstance(backend, MobileCLIPCoreMLBackend)
            if is_coreml:
                coreml_forced = semantic_coreml_chunk_forced()
                if coreml_forced is not None:
                    logger.info(
                        "[INDEX][SPEED] Core ML encode chunk (forced): %d",
                        coreml_forced,
                    )
                    return min(total, coreml_forced)
                if os.environ.get("RAWVIEWER_SEMANTIC_COREML_TUNE", "").strip().lower() not in ("1", "true", "yes", "on"):
                    if semantic_coreml_use_native_batch():
                        chunk = semantic_coreml_chunk_size()
                        logger.info(
                            "[INDEX][SPEED] Core ML encode chunk size: %d (native batch)",
                            chunk,
                        )
                    else:
                        chunk = semantic_coreml_prep_chunk_size()
                        logger.info(
                            "[INDEX][SPEED] Core ML prep chunk: %d (serial predict)",
                            chunk,
                        )
                    return min(total, chunk)
            elif not isinstance(backend, MobileCLIPONNXBackend):
                return 1

        forced = semantic_batch_size_forced()
        if forced is not None:
            return forced

        if not semantic_batch_auto_enabled():
            if is_coreml:
                if semantic_coreml_use_native_batch():
                    chunk = semantic_coreml_chunk_size()
                    logger.info(
                        "[INDEX][SPEED] Core ML encode chunk size: %d (auto-tune off, native batch)",
                        chunk,
                    )
                else:
                    chunk = semantic_coreml_prep_chunk_size()
                    logger.info(
                        "[INDEX][SPEED] Core ML prep chunk: %d (auto-tune off, serial predict)",
                        chunk,
                    )
                return min(total, chunk)
            return semantic_batch_size()

        candidates = (
            semantic_coreml_batch_candidates()
            if is_coreml
            else semantic_batch_candidates()
        )
        tune_label = "Core ML encode chunk" if is_coreml else "Semantic batch"

        cache_key = self._semantic_batch_cache_key(accel_detail)
        cache = _load_semantic_batch_cache()
        cached = cache.get(cache_key)
        if cached not in candidates and cache:
            # Upgrades: reuse v6 auto-tune picks when v7 key is not cached yet.
            legacy_key = cache_key.replace(
                f"|{SEMANTIC_BATCH_TUNE_CACHE_VERSION}",
                "|v6",
            )
            if legacy_key != cache_key:
                cached = cache.get(legacy_key)
        if cached in candidates:
            logger.info(
                "[INDEX][SPEED] %s auto cached: %d",
                tune_label,
                int(cached),
            )
            return int(cached)

        if is_coreml:
            sample_n = min(total, semantic_coreml_tune_sample_count())
        else:
            max_cand = max(candidates)
            sample_n = min(
                total,
                max(semantic_batch_tune_sample_count(), max_cand),
            )
        sample_paths = [cp for cp, _ in list(pending_for_semantic)[:sample_n]]
        best_batch = candidates[0] if candidates else 1
        best_tput = 0.0
        tie_ratio = semantic_batch_tie_ratio()
        early_stop_ratio = semantic_coreml_tune_early_stop_ratio() if is_coreml else 0.0
        logger.info(
            "[INDEX][SPEED] Auto-tuning %s on %d samples; candidates=%s (tie_ratio=%.2f)",
            tune_label.lower(),
            sample_n,
            candidates,
            tie_ratio,
        )
        if progress_callback:
            progress_callback(
                progress_indexed_base,
                progress_album_total or total,
                format_index_progress("Semantic", progress_indexed_base, progress_album_total or total)
                if progress_album_total > 0
                else f"Semantic: tuning 0/{len(candidates)}",
            )

        for ci, cand in enumerate(candidates):
            logger.info(
                "[INDEX][SPEED] Testing %s candidate %d/%d (size=%d)...",
                tune_label.lower(),
                ci + 1,
                len(candidates),
                cand,
            )
            if progress_callback:
                tune_msg = f"Semantic: tuning {ci + 1}/{len(candidates)}"
                if progress_album_total > 0:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        tune_msg,
                    )
                else:
                    progress_callback(ci + 1, len(candidates), tune_msg)
            t0 = time.time()
            ok = True
            try:
                for i in range(0, sample_n, cand):
                    chunk = sample_paths[i : i + cand]
                    self._encode_images_best_effort(chunk)
            except Exception as e:
                ok = False
                logger.info(
                    "[INDEX][SPEED] Batch candidate %d failed during auto-tune: %s",
                    cand,
                    e,
                )
            elapsed = max(1e-9, time.time() - t0)
            tput = (sample_n / elapsed) if ok else 0.0
            logger.info(
                "[INDEX][SPEED] Batch candidate %d -> %.2f img/s (%s, %.1fs)",
                cand,
                tput,
                "ok" if ok else "failed",
                elapsed,
            )
            if ok:
                if tput > best_tput:
                    best_tput = tput
                    best_batch = cand
                elif (
                    best_tput > 0
                    and tput >= best_tput
                    and tput >= best_tput * tie_ratio
                    and cand > best_batch
                ):
                    # True tie only: never prefer a larger batch when it is slower.
                    best_batch = cand
                elif (
                    is_coreml
                    and early_stop_ratio > 0
                    and best_tput > 0
                    and tput < best_tput * early_stop_ratio
                    and cand > best_batch
                ):
                    logger.info(
                        "[INDEX][SPEED] Core ML tune early stop at size=%d "
                        "(%.2f img/s vs best %d at %.2f img/s)",
                        cand,
                        tput,
                        best_batch,
                        best_tput,
                    )
                    break

        cache[cache_key] = int(best_batch)
        _save_semantic_batch_cache(cache)
        logger.info(
            "[INDEX][SPEED] Auto-selected %s size: %d (%.2f img/s on %d samples)",
            tune_label.lower(),
            best_batch,
            best_tput,
            sample_n,
        )
        return int(best_batch)

    def _encode_text(self, text: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-"):
            backend = self._mobileclip_backend or resolve_mobileclip_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP backend unavailable: {err}")
            return backend.encode_text(text)
        model = self._ensure_model()
        return np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def _encode_semantic_query(self, text: str) -> list[np.ndarray]:
        """Encode semantic query with singular/plural and CLIP-style prompt variants."""
        variants = _semantic_text_variants(text)
        if not variants:
            return []
        return [self._encode_text(v) for v in variants]

    @staticmethod
    def _semantic_similarity(query_vecs: Sequence[np.ndarray], image_vec: np.ndarray) -> float:
        if not query_vecs:
            return 0.0
        return max(float(np.dot(qv, image_vec)) for qv in query_vecs)

    @staticmethod
    def _face_scan_worker_count(sample_path: str | None = None) -> int:
        """Parallel image-load workers for face scan. YuNet inference is serialized."""
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WORKERS", "").strip()
        if raw:
            try:
                return max(1, min(16, int(raw)))
            except ValueError:
                pass

        from common_image_loader import is_external_or_network_volume
        if is_external_or_network_volume(sample_path):
            return 2
        try:
            from core.profile import classify_memory_tier, system_total_ram_gb

            tier = classify_memory_tier(system_total_ram_gb())
            if tier in ("low", "medium", "balanced"):
                return 2
        except Exception:
            pass
        cpu = os.cpu_count() or 4
        return min(4, max(2, cpu // 2))

    @staticmethod
    def _face_scan_inflight_chunk() -> int:
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_CHUNK", "64").strip()
        try:
            return max(4, min(512, int(raw)))
        except ValueError:
            return 64

    @staticmethod
    def _face_scan_parallel_enabled() -> bool:
        v = os.environ.get("RAWVIEWER_FACE_SCAN_PARALLEL", "").strip().lower()
        if not v:
            # Nested thread pools + Qt/raw decode from worker threads crash on Windows.
            return sys.platform != "win32"
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _defer_face_scan_during_build() -> bool:
        """
        When true, build_index finishes after metadata + semantic embeddings.
        Face backfill runs in a second background pass so search becomes usable sooner.
        """
        v = os.environ.get("RAWVIEWER_INDEX_DEFER_FACE_SCAN", "1").strip().lower()
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _thumbnail_warm_before_face_scan() -> bool:
        # Default off: warming 6k+ RAWs before face scan blocks indexing for hours.
        v = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_THUMBS", "0").strip().lower()
        return v not in ("0", "false", "no", "off")

    @staticmethod
    def _face_scan_warm_max_files() -> int:
        """
        Cap warm-up work so face indexing cannot appear frozen for giant albums.
        0 disables warm-up entirely; negative means unlimited.
        """
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_MAX_FILES", "256").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        return 256

    @staticmethod
    def _face_scan_warm_max_seconds() -> float:
        """Best-effort warm-up deadline; exceeded budget continues with face scan directly."""
        raw = os.environ.get("RAWVIEWER_FACE_SCAN_WARM_MAX_SECONDS", "25").strip()
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
        return 25.0

    @staticmethod
    def face_detection_max_edge() -> int:
        """Longest edge passed to YuNet/Vision (see _load_index_source_image in _detect_face_count)."""
        raw = os.environ.get("RAWVIEWER_FACE_DETECT_MAX_EDGE", "").strip()
        if raw:
            try:
                return max(320, min(2048, int(raw)))
            except ValueError:
                pass
        return 1280

    def _warm_thumbnail_cache_for_semantic_index(
        self,
        paths: list[str],
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
        stage_label: str = "Semantic",
    ) -> int:
        """Pre-extract thumbnails so MobileCLIP encode hits ImageCache instead of rawpy per file."""
        if not paths or not semantic_warm_thumbs_before_index():
            return 0
        return self._ensure_index_thumbnails_cached(
            paths,
            progress_callback,
            progress_album_total=progress_album_total,
            progress_indexed_base=progress_indexed_base,
            purpose="semantic",
        )

    def _ensure_index_thumbnails_cached(
        self,
        paths: list[str],
        progress_callback: ProgressCallback = None,
        *,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
        purpose: str = "index",
        max_workers: int | None = None,
    ) -> int:
        """
        Fill ImageCache for paths missing preview/grid/thumbnail tiers.
        Shared by semantic encoding and deferred face scan (same warmed pixels).
        """
        import logging

        from image_cache import get_image_cache

        logger = logging.getLogger(__name__)
        if not paths:
            return 0

        cache = get_image_cache()
        pending = [p for p in paths if _index_thumbnail_needs_warm(cache, p)]
        already_cached = len(paths) - len(pending)
        if not pending:
            logger.info(
                "[INDEX] %s thumbnail cache: all %d paths already in ImageCache",
                purpose,
                len(paths),
            )
            return 0
        if already_cached:
            logger.info(
                "[INDEX] %s thumbnail cache: %d/%d already in ImageCache; %d still need extract",
                purpose,
                already_cached,
                len(paths),
                len(pending),
            )

        if max_workers is None:
            sample_path = pending[0] if pending else None
            max_workers = semantic_encode_prep_workers(sample_path)
        workers = max(1, int(max_workers))
        t0 = time.time()
        warmed = 0

        @staticmethod
        def _warm_one(path: str) -> bool:
            try:
                self._wait_if_paused()
                from common_image_loader import (
                    finalize_index_thumbnail_array,
                    index_thumbnail_needs_orient_fix,
                    is_raw_file,
                )
                from enhanced_raw_processor import (
                    ThumbnailExtractor,
                    extract_embedded_jpeg_by_scan,
                )

                arr = _cached_index_source_array(cache, path)
                if arr is not None and not index_thumbnail_needs_orient_fix(
                    path, arr, cache=cache
                ):
                    return True

                if arr is not None and index_thumbnail_needs_orient_fix(
                    path, arr, cache=cache
                ):
                    arr = finalize_index_thumbnail_array(path, arr, cache=cache)
                else:
                    if is_raw_file(path):
                        arr = extract_embedded_jpeg_by_scan(path, 1024)
                        if arr is None:
                            arr = ThumbnailExtractor().extract_thumbnail_from_raw(
                                path, max_size=1024, allow_scan_fallback=False
                            )
                            if arr is not None:
                                arr = np.asarray(arr, dtype=np.uint8)
                        if arr is not None:
                            arr = finalize_index_thumbnail_array(
                                path, arr, cache=cache
                            )
                    else:
                        with Image.open(path) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            im.thumbnail((1024, 1024), Image.Resampling.BICUBIC)
                            arr = np.asarray(im)

                if arr is not None:
                    jpeg = None
                    with contextlib.suppress(Exception):
                        jpeg = _thumbnail_jpeg_bytes(arr)
                    cache.put_thumbnail(path, arr, jpeg if jpeg is not None else b"")
                    return True
            except Exception:
                pass
            return False

        logger.info(
            "[INDEX] Warming thumbnail cache for %s: %d/%d files (%d workers)",
            purpose,
            len(pending),
            len(paths),
            workers,
        )
        total = len(pending)
        warmed_base = progress_indexed_base + (len(paths) - len(pending)) * 0.1

        @staticmethod
        def _thumb_progress_ui_count(done: int) -> int:
            """User-visible thumbnail count (not album-weighted 10% slice)."""
            return min(total, max(0, int(done)))

        @staticmethod
        def _thumb_progress_message(done: int) -> str:
            return format_index_progress(
                "Preparing thumbnails", _thumb_progress_ui_count(done), total
            )

        @staticmethod
        def _emit_thumb_progress(done: int) -> None:
            if not progress_callback:
                return
            ui_done = _thumb_progress_ui_count(done)
            if progress_album_total > 0:
                progress_callback(
                    int(warmed_base + ui_done * 0.1),
                    progress_album_total,
                    _thumb_progress_message(done),
                )
            else:
                progress_callback(ui_done, total, _thumb_progress_message(done))

        @staticmethod
        def _should_emit_thumb_progress(done: int) -> bool:
            if done <= 3 or done >= total:
                return True
            # ~1% steps, at least every 10 files (backend can finish >>10/s).
            step = max(10, total // 100)
            return done % step == 0

        if workers <= 1 or total < 4:
            for i, path in enumerate(pending, start=1):
                self._wait_if_paused()
                if _warm_one(path):
                    warmed += 1
                if _should_emit_thumb_progress(i):
                    _emit_thumb_progress(i)
        else:
            completed = 0
            executor = _get_shared_index_executor()
            futures = {executor.submit(_warm_one, p): p for p in pending}
            try:
                for future in concurrent.futures.as_completed(futures):
                    self._wait_if_paused()
                    completed += 1
                    if future.result():
                        warmed += 1
                    if _should_emit_thumb_progress(completed):
                        _emit_thumb_progress(completed)
            except IndexAborted:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            # Removed executor.shutdown() since it's shared
        logger.info(
            "[INDEX][SPEED] %s thumbnail warm-up: warmed=%d pending=%d already_cached=%d "
            "elapsed=%.3fs throughput=%.2f img/s",
            purpose,
            warmed,
            len(pending),
            already_cached,
            time.time() - t0,
            float(len(pending)) / max(1e-9, time.time() - t0),
        )
        return warmed

    def _face_pending_paths(
        self, conn: sqlite3.Connection, canonical_paths: Sequence[str]
    ) -> list[str]:
        """Paths in this batch that still need face_count in the DB."""
        if not canonical_paths:
            return []
        pending: list[str] = []
        chunk_size = 900
        unique = list(dict.fromkeys(canonical_paths))
        for i in range(0, len(unique), chunk_size):
            chunk = unique[i : i + chunk_size]
            qs = ",".join(["?"] * len(chunk))
            cursor = conn.execute(
                f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs}) AND face_count IS NULL AND model_name = ?",
                [*chunk, self.model_name],
            )
            pending.extend(row[0] for row in cursor.fetchall())
        return pending

    def get_face_pending_count(self, file_paths: Sequence[str]) -> int:
        if not face_scan_enabled():
            return 0
        if not file_paths:
            return 0
        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        canonical = [self._canonical_path(p) for p in indexable_paths if p]
        if not canonical:
            return 0
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                pending = set(self._face_pending_paths(conn, canonical))
                # Rows not in the DB yet also need a face scan pass.
                chunk_size = 900
                indexed_paths: set[str] = set()
                for i in range(0, len(canonical), chunk_size):
                    chunk = canonical[i : i + chunk_size]
                    qs = ",".join(["?"] * len(chunk))
                    cursor = conn.execute(
                        f"SELECT file_path FROM semantic_index WHERE file_path IN ({qs})",
                        chunk,
                    )
                    indexed_paths.update(row[0] for row in cursor.fetchall())
                for cp in canonical:
                    if cp not in indexed_paths:
                        pending.add(cp)
                return len(pending)
            finally:
                conn.close()
        except Exception:
            return 0

    def _enqueue_metadata_backfill(
        self,
        file_paths: Sequence[str],
        *,
        geocode_paths: Sequence[str] | None = None,
    ) -> None:
        """Queue missing metadata / geocode repair for a background worker."""
        if not metadata_auto_index_enabled():
            return
        with self._metadata_backfill_lock:
            for p in file_paths or []:
                if p:
                    self._metadata_backfill_paths.add(p)
            for p in geocode_paths or []:
                if p:
                    self._metadata_backfill_geocode.add(p)
            if self._metadata_backfill_running:
                return
            if not self._metadata_backfill_paths and not self._metadata_backfill_geocode:
                return
            self._metadata_backfill_running = True
            thread = threading.Thread(
                target=self._run_metadata_backfill_worker,
                name="rawviewer-metadata-backfill",
                daemon=True,
            )
            thread.start()

    def _run_metadata_backfill_worker(self) -> None:
        import logging

        logger = logging.getLogger(__name__)
        try:
            while True:
                with self._metadata_backfill_lock:
                    extract_paths = list(self._metadata_backfill_paths)
                    geocode_paths = list(self._metadata_backfill_geocode)
                    self._metadata_backfill_paths.clear()
                    self._metadata_backfill_geocode.clear()
                    if not extract_paths and not geocode_paths:
                        self._metadata_backfill_running = False
                        return

                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=60000")
                exif_batch: list[tuple] = []
                commit_every = 40
                batch_writes = 0
                try:
                    if geocode_paths:
                        geo = self._ensure_reverse_geocoder()
                        if isinstance(geo, CustomReverseGeocoder):
                            for fp in geocode_paths:
                                try:
                                    canonical = self._canonical_path(fp)
                                    row = conn.execute(
                                        "SELECT gps_lat, gps_lon, city FROM semantic_index WHERE file_path = ?",
                                        (canonical,),
                                    ).fetchone()
                                    if not row or row[2]:
                                        continue
                                    lat, lon = row[0], row[1]
                                    if lat is None or lon is None:
                                        continue
                                    with self._rg_lock:
                                        recs = geo.search(
                                            [(_coerce_float(lat) or 0.0, _coerce_float(lon) or 0.0)],
                                            mode=1,
                                        )
                                    if not recs:
                                        continue
                                    rec = recs[0] or {}
                                    city_name = str(rec.get("name", "") or "")
                                    alternates = str(rec.get("alternates", "") or "")
                                    city = f"{city_name}, {alternates}" if alternates else city_name
                                    landmark = str(rec.get("landmark", "") or "")
                                    admin1 = str(rec.get("admin1", "") or "")
                                    cc = str(rec.get("cc", "") or "").upper()
                                    country = self._country_name_from_code(cc)
                                    conn.execute(
                                        "UPDATE semantic_index SET city=?, landmark=?, admin1=?, country=?, country_code=? WHERE file_path=?",
                                        (city, landmark, admin1, country, cc, canonical),
                                    )
                                    batch_writes += 1
                                    if batch_writes >= commit_every:
                                        conn.commit()
                                        batch_writes = 0
                                except Exception:
                                    continue

                    for fp in extract_paths:
                        try:
                            canonical = self._canonical_path(fp)
                            if not os.path.isfile(canonical):
                                continue
                            st = os.stat(canonical)
                            meta = self._extract_exif_brief(
                                canonical, include_face=False, file_stat=st, persist_exif=False
                            )
                            if meta:
                                self._upsert_metadata(canonical, st, meta, conn=conn)
                                batch_writes += 1
                                exif_batch.append((canonical, meta, st))
                            if batch_writes >= commit_every:
                                conn.commit()
                                batch_writes = 0
                                self._flush_exif_backfill_batch(exif_batch)
                                exif_batch.clear()
                        except Exception:
                            continue

                    if batch_writes > 0:
                        conn.commit()
                    self._flush_exif_backfill_batch(exif_batch)
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("[INDEX] Metadata backfill worker failed: %s", exc)
            with self._metadata_backfill_lock:
                self._metadata_backfill_running = False

    @staticmethod
    def _flush_exif_backfill_batch(exif_batch: list[tuple]) -> None:
        if not exif_batch:
            return
        try:
            from enhanced_raw_processor import RAW_EXIF_SENSOR_META_VER
            from image_cache import get_image_cache

            cache = get_image_cache()
            items = []
            for canonical, meta, st in exif_batch:
                items.append(
                    (
                        canonical,
                        {
                            "orientation": int(meta.get("orientation") or 1),
                            "camera_make": "",
                            "camera_model": str(meta.get("camera_model") or ""),
                            "capture_time": str(meta.get("capture_time") or ""),
                            "original_width": int(meta.get("width") or 0),
                            "original_height": int(meta.get("height") or 0),
                            "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                            "exif_data": {
                                "capture_time": str(meta.get("capture_time") or ""),
                                "camera_model": str(meta.get("camera_model") or ""),
                                "lens_model": str(meta.get("lens_model") or ""),
                                "iso": int(meta.get("iso") or 0),
                                "raw_exif_sensor_meta_ver": RAW_EXIF_SENSOR_META_VER,
                            },
                        },
                        int(st.st_size),
                        float(st.st_mtime),
                    )
                )
            put_many = getattr(cache.exif_cache, "put_many", None)
            if callable(put_many):
                put_many(items, commit=True)
        except Exception:
            pass
