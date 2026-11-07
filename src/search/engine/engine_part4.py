"""Engine mixin part 4 of 7 for SemanticImageIndex."""
from __future__ import annotations

import concurrent
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Sequence

from raw_adjustments.file_extensions import RAW_FILE_EXTENSIONS
from search.indexer import (
    _dnn_get_backend_name,
    _get_shared_index_executor,
    _index_source_thumbnail_present,
    _load_face_scan_image,
    _yunet_warmup,
    face_scan_enabled,
    format_index_progress,
    resolve_opencv_dnn_backend_target,
    semantic_encode_prep_workers,
)
from search.types import ProgressCallback


class EnginePart4:




    def backfill_search_metadata(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
    ) -> int:
        """Background metadata extraction + geocode repair (search/index gap fill)."""
        if not file_paths:
            return 0
        self._enqueue_metadata_backfill(list(file_paths))
        return len(file_paths)

    def backfill_face_counts(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
        *,
        album_total: int = 0,
        album_indexed_base: int = 0,
    ) -> int:
        """Deferred face-only pass (after semantic index is ready for search)."""
        import sqlite3

        if not face_scan_enabled():
            return 0
        if not file_paths:
            return 0
        self.release_mobileclip_gpu_memory()
        canonical_map = {self._canonical_path(p): p for p in file_paths if p}
        unique_canonical = list(canonical_map.keys())
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        try:
            face_pending = self._face_pending_paths(conn, unique_canonical)
            if face_pending:
                if progress_callback:
                    progress_callback(
                        album_indexed_base,
                        album_total or len(file_paths),
                        format_index_progress("Face", 0, len(face_pending)),
                    )
                self._run_parallel_face_scan(
                    face_pending,
                    conn,
                    progress_callback,
                    commit_every=40,
                    progress_album_total=album_total or len(file_paths),
                    progress_indexed_base=album_indexed_base,
                )
            return len(face_pending)
        finally:
            conn.close()

    def _run_parallel_face_scan(
        self,
        face_pending: list[str],
        conn: sqlite3.Connection,
        progress_callback: ProgressCallback,
        *,
        conn_lock: threading.Lock | None = None,
        commit_every: int = 40,
        progress_album_total: int = 0,
        progress_indexed_base: int = 0,
    ) -> None:
        import logging

        logger = logging.getLogger(__name__)
        total_face = len(face_pending)
        if total_face <= 0:
            return

        max_edge = self.face_detection_max_edge()
        cache_misses: list[str] = []
        try:
            from image_cache import get_image_cache

            cache = get_image_cache()
            cached_n = 0
            for p in face_pending:
                if _index_source_thumbnail_present(cache, p):
                    cached_n += 1
                else:
                    cache_misses.append(p)
            logger.info(
                "[INDEX] Face scan ImageCache ready: %d/%d (preview/grid/thumbnail or disk)",
                cached_n,
                total_face,
            )
        except Exception:
            cache_misses = list(face_pending)

        if cache_misses:
            logger.info(
                "[INDEX] Face scan filling %d missing thumbnails (semantic pass should have cached most)",
                len(cache_misses),
            )
            sample_path = cache_misses[0] if cache_misses else None
            face_warm_workers = (
                1
                if sys.platform == "win32"
                else min(semantic_encode_prep_workers(sample_path), 4)
            )
            self._ensure_index_thumbnails_cached(
                cache_misses,
                progress_callback,
                progress_album_total=progress_album_total,
                progress_indexed_base=progress_indexed_base,
                purpose="face",
                max_workers=face_warm_workers,
            )
        else:
            logger.info(
                "[INDEX] Face scan reusing semantic thumbnails for all %d files",
                total_face,
            )

        _yunet_warmup()

        sample_path = face_pending[0] if face_pending else None
        workers = self._face_scan_worker_count(sample_path)
        chunk_size = self._face_scan_inflight_chunk()
        t_face_start = time.time()
        batch_writes = 0
        lock = conn_lock or threading.Lock()
        parallel = self._face_scan_parallel_enabled() and total_face >= 4
        try:
            import cv2

            backend_id, target_id = resolve_opencv_dnn_backend_target()
            backend_name = _dnn_get_backend_name(cv2, backend_id)
            logger.info(
                "[INDEX] Face scan config: backend=%s target=%s workers=%d chunk=%d parallel=%s",
                backend_name,
                target_id,
                workers,
                chunk_size,
                parallel,
            )
        except Exception:
            pass

        @staticmethod
        def _scan_one(cp: str) -> tuple:
            try:
                im = _load_face_scan_image(cp, max_size=max_edge)
                if im is None:
                    logger.warning(
                        "[INDEX] Face scan cache miss (no warmed thumbnail): %s",
                        os.path.basename(cp),
                    )
                    return cp, 0
                return cp, int(self._detect_face_count(cp, preloaded_im=im) or 0)
            except Exception as e:
                logger.error(
                    "[INDEX] Face scanning failed for %s: %s",
                    os.path.basename(cp),
                    e,
                )
                return cp, 0

        if not parallel:
            for idx, cp in enumerate(face_pending, start=1):
                if progress_callback and (
                    idx <= 2 or idx >= total_face or idx % 10 == 0
                ):
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", idx, total_face),
                    )
                cp, face_count = _scan_one(cp)
                with lock:
                    self._store_face_count(cp, face_count, conn=conn, commit=False)
                    batch_writes += 1
                    if batch_writes >= commit_every:
                        conn.commit()
                        batch_writes = 0
            if batch_writes > 0:
                with lock:
                    conn.commit()
            dur = time.time() - t_face_start
            logger.info(
                "[INDEX] Face scan (sequential): %d files in %.2fs (%.0f ms/img)",
                total_face,
                dur,
                (dur / total_face) * 1000.0 if total_face else 0,
            )
            return

        logger.info(
            "[INDEX] Parallel face scan: %d files, %d workers",
            total_face,
            workers,
        )
        completed = 0
        executor = _get_shared_index_executor()
        for chunk_start in range(0, total_face, chunk_size):
            chunk = face_pending[chunk_start : chunk_start + chunk_size]
            futures = [executor.submit(_scan_one, cp) for cp in chunk]
            for future in concurrent.futures.as_completed(futures):
                cp, face_count = future.result()
                with lock:
                    self._store_face_count(cp, face_count, conn=conn, commit=False)
                    batch_writes += 1
                    if batch_writes >= commit_every:
                        conn.commit()
                        batch_writes = 0
                completed += 1
                if progress_callback and (
                    completed <= 2
                    or completed >= total_face
                    or completed % 10 == 0
                ):
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", completed, total_face),
                    )

        if batch_writes > 0:
            with lock:
                conn.commit()

        dur = time.time() - t_face_start
        logger.info(
            "[INDEX] Face scan (parallel, %d workers): %d files in %.2fs (%.0f ms/img)",
            workers,
            total_face,
            dur,
            (dur / total_face) * 1000.0 if total_face else 0,
        )

    @staticmethod
    def _raw_companion_key(fp: str) -> str:
        base = os.path.basename(fp)
        parts = base.split(".")
        if len(parts) > 1:
            for idx, part in enumerate(parts):
                if part.startswith("RAW-"):
                    return ".".join(parts[:idx]).lower()
            return ".".join(parts[:-1]).lower()
        return base.lower()

    def _filter_duplicate_raw_companions(
        cls, file_paths: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        """Return (indexable paths, skipped RAW paths with a non-raw companion)."""
        filtered_paths: list[str] = []
        skipped_raw: list[str] = []
        non_raw_keys: set = set()
        for fp in file_paths:
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext and ext not in RAW_FILE_EXTENSIONS:
                non_raw_keys.add((os.path.dirname(fp), cls._raw_companion_key(fp)))
        for fp in file_paths:
            if not fp:
                continue
            ext = os.path.splitext(fp)[1].lower().lstrip(".")
            if ext in RAW_FILE_EXTENSIONS and (os.path.dirname(fp), cls._raw_companion_key(fp)) in non_raw_keys:
                skipped_raw.append(fp)
                continue
            filtered_paths.append(fp)
        return filtered_paths, skipped_raw
