"""Engine mixin part 5 of 7 for SemanticImageIndex."""
from __future__ import annotations

import concurrent
import os
import threading
import time
from collections.abc import Sequence

from core.errors import IndexAborted
from search.backends import MobileCLIPCoreMLBackend, MobileCLIPONNXBackend, resolve_mobileclip_backend
from search.indexer import (
    _coerce_int,
    _coreml_compute_units_setting_for_indexing,
    _coreml_compute_units_setting_for_search,
    _get_shared_index_executor,
    _ort_get_available_providers,
    face_scan_enabled,
    format_index_progress,
    log_inference_acceleration_profile,
    resolve_onnx_execution_providers,
    semantic_embeddings_enabled,
    semantic_gpu_throttle_seconds,
)
from search.types import ProgressCallback


class EnginePart5:




    def build_index(
        self,
        file_paths: Sequence[str],
        progress_callback: ProgressCallback = None,
        *,
        album_total: int = 0,
        album_indexed_base: int = 0,
        run_face_scan: bool | None = None,
        run_semantic_embeddings: bool | None = None,
    ) -> dict[str, int]:
        import logging
        import sqlite3
        import sys
        logger = logging.getLogger(__name__)
        t_start = time.time()
        if run_semantic_embeddings is None:
            run_semantic_embeddings = semantic_embeddings_enabled()
        logger.info(f"[INDEX] Starting indexing of {len(file_paths)} file paths.")
        if run_semantic_embeddings:
            log_inference_acceleration_profile()
        if sys.platform != "darwin":
            logger.info("[VISION] Using OpenCV offline face scanner on Windows.")

        filtered_paths, skipped_raw_paths = self._filter_duplicate_raw_companions(file_paths)
        skipped_companions = len(skipped_raw_paths)
        for fp in skipped_raw_paths:
            logger.info(
                "[INDEX] Skipping RAW companion file to avoid duplicate results: %s",
                os.path.basename(fp),
            )
        logger.info(
            "[INDEX] Filtered out %d RAW companion files. Actual files to evaluate: %d",
            skipped_companions,
            len(filtered_paths),
        )

        total = len(file_paths)
        indexed = 0
        skipped = skipped_companions
        failed = 0
        pending_for_semantic: list[tuple[str, os.stat_result]] = []
        progress_album_total = album_total if album_total > 0 else total
        progress_indexed_base = max(0, album_indexed_base)

        # Create a thread-local, dedicated connection for this background worker thread!
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=60000")

        if skipped_raw_paths:
            for fp in skipped_raw_paths:
                try:
                    cp = self._canonical_path(fp)
                    st = os.stat(cp)
                    self._mark_semantic_skipped(cp, st, conn)
                except OSError:
                    pass
            conn.commit()

        total_face = 0
        run_face_inline = False
        try:
            # 1.1 Pre-fetch existing metadata in bulk to avoid thousands of small SQL queries
            existing_meta = {} # {canonical_path: (mtime, size, semantic_ready)}
            canonical_map = {self._canonical_path(fp): fp for fp in filtered_paths if fp}
            unique_canonical = list(canonical_map.keys())

            logger.info(f"[INDEX] Pre-fetching existing metadata database entries for {len(unique_canonical)} canonical paths...")
            t0 = time.time()
            chunk_size = 900
            for i in range(0, len(unique_canonical), chunk_size):
                chunk = unique_canonical[i:i+chunk_size]
                qs = ",".join(["?"] * len(chunk))
                cursor = conn.execute(
                    f"SELECT file_path, file_mtime, file_size, semantic_ready, dim, gps_lat, city FROM semantic_index WHERE file_path IN ({qs})",
                    chunk
                )
                for row in cursor.fetchall():
                    existing_meta[row[0]] = {
                        "mtime": row[1],
                        "size": row[2],
                        "semantic_ready": row[3],
                        "dim": row[4],
                        "gps_lat": row[5],
                        "city": row[6]
                    }
            logger.info(f"[INDEX] Pre-fetch completed in {time.time() - t0:.4f}s. Found {len(existing_meta)} matches in database.")

            @staticmethod
            def needs_reindex_local(cp, st):
                if cp not in existing_meta:
                    return True
                row = existing_meta[cp]
                if not self._mtime_matches(row["mtime"], st):
                    return True
                return int(row["size"]) != int(st.st_size)

            # 1.2 Identify files that actually need metadata extraction
            to_extract = []
            for fp in filtered_paths:
                if not fp:
                    continue
                canonical_fp = self._canonical_path(fp)
                try:
                    st = os.stat(canonical_fp)
                    if needs_reindex_local(canonical_fp, st):
                        to_extract.append((canonical_fp, st))
                    else:
                        row = existing_meta[canonical_fp]
                        sr = int(row["semantic_ready"] or 0)
                        dim = _coerce_int(self._row_value(row, "dim"))
                        needs_semantic = (sr == 0) or (
                            bool(run_semantic_embeddings) and sr == 1 and dim == 0
                        )
                        if needs_semantic:
                            pending_for_semantic.append((canonical_fp, st))
                        else:
                            skipped += 1
                except OSError:
                    failed += 1
                    continue

            logger.info(f"[INDEX] Needs metadata extraction: {len(to_extract)} files. Already indexed & skipped: {skipped} files.")

            # 1.3 Parallel extraction of metadata
            total_extract = len(to_extract)
            batch_writes = 0
            commit_every = 40

            if total_extract > 0:
                logger.info(f"[INDEX] Starting parallel metadata extraction for {total_extract} files using ThreadPoolExecutor...")
                self._acquire_load_throttle()
                if progress_callback:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Metadata", progress_indexed_base, progress_album_total),
                    )
                t_meta_start = time.time()
                from common_image_loader import index_metadata_worker_count

                sample_fp = to_extract[0][0] if to_extract else None
                max_workers = index_metadata_worker_count(total_extract, sample_fp)
                logger.info(
                    "[INDEX] Metadata workers=%d for %d files",
                    max_workers,
                    total_extract,
                )
                executor = _get_shared_index_executor()
                try:
                    @staticmethod
                    def extract_task(item):
                        self._wait_if_paused()
                        cp, st = item
                        try:
                            t_single_start = time.time()
                            meta = self._extract_exif_brief(
                                cp, include_face=False, file_stat=st
                            )
                            t_single_dur = time.time() - t_single_start
                            if t_single_dur > 2.0:
                                logger.warning(
                                    "[INDEX] Slow metadata extraction for %s: %.3fs "
                                    "(often EXIF cache lock vs folder sort refinement)",
                                    os.path.basename(cp),
                                    t_single_dur,
                                )
                            return cp, st, meta
                        except Exception as e:
                            logger.error(f"[INDEX] Failed to extract EXIF for {os.path.basename(cp)}: {e}")
                            return cp, st, None

                    futures = [executor.submit(extract_task, item) for item in to_extract]
                    try:
                        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                            self._wait_if_paused()
                            cp, st, meta = future.result()
                            if meta:
                                try:
                                    self._upsert_metadata(cp, st, meta, conn=conn)
                                    pending_for_semantic.append((cp, st))
                                    batch_writes += 1
                                    # Do not increment 'indexed' here; it will be incremented in Phase 2
                                    # when the semantic embedding is actually ready.

                                    if batch_writes >= commit_every:
                                        conn.commit()
                                        batch_writes = 0
                                except Exception as e:
                                    logger.error(f"[INDEX] Database upsert failed for {os.path.basename(cp)}: {e}")
                                    failed += 1
                            else:
                                failed += 1

                            if i % 100 == 0:
                                import gc
                                gc.collect()

                            if progress_callback and (i <= 2 or i >= total_extract or i % 10 == 0):
                                progress_callback(
                                    progress_indexed_base,
                                    progress_album_total,
                                    format_index_progress(
                                        "Metadata", progress_indexed_base + i, progress_album_total
                                    ),
                                )
                    except IndexAborted:
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                except IndexAborted:
                    raise

                if batch_writes > 0:
                    conn.commit()
                    batch_writes = 0
                logger.info(f"[INDEX] Completed metadata extraction for {total_extract} files in {time.time() - t_meta_start:.4f}s.")
                try:
                    from image_cache import get_image_cache
                    get_image_cache().flush_exif_disk_cache()
                except Exception:
                    pass

            face_pending = self._face_pending_paths(conn, unique_canonical) if face_scan_enabled() else []
            total_face = len(face_pending)
            total_sem = len(pending_for_semantic)
            if run_face_scan is None:
                run_face_scan = face_scan_enabled() and not self._defer_face_scan_during_build()
            run_face_inline = total_face > 0 and run_face_scan

            if total_face > 0 and not run_face_inline:
                logger.info(
                    "[INDEX] Deferring face scan for %d files until after semantic pass",
                    total_face,
                )

            # Phase 2: semantic embeddings (search-critical). Run before face scan.
            if total_sem > 0:
                self._wait_if_paused()
                if run_semantic_embeddings:
                    # Heavy GPU/RAM + thumbnail decode work: throttle ILM now (not
                    # at index start) so light/paused phases never starve the gallery.
                    self._acquire_load_throttle()
                    logger.info(f"[INDEX] Starting AI features neural pass (MobileCLIP) for {total_sem} files...")
                    if progress_callback:
                        progress_callback(
                            progress_indexed_base,
                            progress_album_total,
                            format_index_progress("Semantic", progress_indexed_base, progress_album_total),
                        )
                    sem_gpu_accel = False
                    sem_accel_detail = "unknown"
                    try:
                        if self.model_name.startswith("mobileclip-"):
                            backend = self._mobileclip_backend or resolve_mobileclip_backend()
                            self._mobileclip_backend = backend
                            if isinstance(backend, MobileCLIPONNXBackend):
                                try:
                                    selected = resolve_onnx_execution_providers(
                                        _ort_get_available_providers()
                                    )
                                except Exception:
                                    selected = ["CPUExecutionProvider"]
                                sem_gpu_accel = any(
                                    str(p) != "CPUExecutionProvider" for p in selected
                                )
                                sem_accel_detail = f"ONNX providers=[{', '.join(selected)}]"
                            elif isinstance(backend, MobileCLIPCoreMLBackend):
                                index_cu = _coreml_compute_units_setting_for_indexing()
                                search_cu = _coreml_compute_units_setting_for_search()
                                sem_gpu_accel = index_cu not in ("cpu", "cpuonly")
                                if index_cu == search_cu:
                                    sem_accel_detail = f"CoreML compute_units={index_cu}"
                                else:
                                    sem_accel_detail = (
                                        f"CoreML index={index_cu} search={search_cu}"
                                    )
                            else:
                                sem_accel_detail = backend.__class__.__name__
                        else:
                            sem_accel_detail = f"model={self.model_name}"
                    except Exception as accel_exc:
                        sem_accel_detail = f"unavailable ({accel_exc})"
                    logger.info(
                        "[INDEX][ACCEL] Semantic acceleration: gpu=%s (%s)",
                        "ON" if sem_gpu_accel else "OFF",
                        sem_accel_detail,
                    )
                    try:
                        if self.model_name.startswith("mobileclip-"):
                            backend = self._mobileclip_backend or resolve_mobileclip_backend()
                            self._mobileclip_backend = backend
                            if isinstance(backend, MobileCLIPONNXBackend):
                                img_session, txt_session = backend._require_sessions()
                                img_providers = img_session.get_providers()
                                txt_providers = txt_session.get_providers()
                                logger.info(
                                    "[INDEX][ACCEL] ONNX active session providers: image=%s text=%s",
                                    img_providers,
                                    txt_providers,
                                )
                    except Exception as provider_exc:
                        logger.debug(
                            "[INDEX][ACCEL] Could not read active ONNX session providers: %s",
                            provider_exc,
                        )
                    warm_paths = [cp for cp, _ in pending_for_semantic]
                    t_warm_start = time.time()
                    warm_count = self._warm_thumbnail_cache_for_semantic_index(
                        warm_paths,
                        progress_callback,
                        progress_album_total=progress_album_total,
                        progress_indexed_base=progress_indexed_base,
                    )
                    t_warm_elapsed = max(0.0, time.time() - t_warm_start)
                    throttle_sec = semantic_gpu_throttle_seconds()
                    batch_size = self._auto_select_semantic_batch_size(
                        pending_for_semantic,
                        sem_accel_detail,
                        progress_callback,
                        progress_album_total=progress_album_total,
                        progress_indexed_base=progress_indexed_base,
                    )
                    logger.info(
                        "[INDEX][SPEED] Semantic batch size: %d",
                        batch_size,
                    )
                    if throttle_sec > 0:
                        logger.info(
                            "[INDEX][SPEED] Semantic throttle enabled: %.0f ms/image",
                            throttle_sec * 1000.0,
                        )
                    t_sem_start = time.time()
                    sem_success = 0
                    sem_fail = 0
                    i = 0
                    t_encode_cpu = 0.0
                    for batch_start in range(0, total_sem, batch_size):
                        self._wait_if_paused()
                        batch_items = pending_for_semantic[batch_start : batch_start + batch_size]
                        batch_paths = [cp for cp, _ in batch_items]
                        try:
                            t_batch_neural = time.time()
                            vecs = self._encode_images_best_effort(batch_paths)
                            t_batch_dur = time.time() - t_batch_neural
                            t_encode_cpu += t_batch_dur
                            if t_batch_dur > 0.5:
                                logger.info(
                                    "[INDEX] MobileCLIP encoding batch=%d took %.4fs",
                                    len(batch_items),
                                    t_batch_dur,
                                )
                            for (canonical_fp, st), vec in zip(batch_items, vecs, strict=False):
                                i += 1
                                conn.execute(
                                    """
                                    UPDATE semantic_index
                                    SET dim = ?, embedding = ?, semantic_ready = 1, updated_at = ?
                                    WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
                                    """,
                                    (
                                        int(vec.size),
                                        self._to_blob(vec),
                                        float(time.time()),
                                        canonical_fp,
                                        self.model_name,
                                        int(st.st_size),
                                        self._mtime_ns_from_stat(st),
                                    ),
                                )
                                indexed += 1
                                sem_success += 1
                                batch_writes += 1
                                if batch_writes >= commit_every:
                                    conn.commit()
                                    batch_writes = 0
                                if throttle_sec > 0:
                                    time.sleep(throttle_sec)

                            if progress_callback:
                                progress_callback(
                                    int(progress_indexed_base + total_sem * 0.1 + i * 0.9),
                                    progress_album_total,
                                    format_index_progress("Semantic", int(progress_indexed_base + total_sem * 0.1 + i * 0.9), progress_album_total),
                                )
                        except Exception as batch_exc:
                            logger.warning(
                                "[INDEX] Batch semantic encode failed (size=%d): %s. Falling back to per-image.",
                                len(batch_items),
                                batch_exc,
                            )
                            for canonical_fp, st in batch_items:
                                i += 1
                                try:
                                    t_single_neural = time.time()
                                    vec = self._encode_image(canonical_fp)
                                    t_neural_dur = time.time() - t_single_neural
                                    t_encode_cpu += t_neural_dur
                                    if t_neural_dur > 0.5:
                                        logger.info(
                                            f"[INDEX] MobileCLIP encoding for {os.path.basename(canonical_fp)} took {t_neural_dur:.4f}s"
                                        )
                                    conn.execute(
                                        """
                                        UPDATE semantic_index
                                        SET dim = ?, embedding = ?, semantic_ready = 1, updated_at = ?
                                        WHERE file_path = ? AND model_name = ? AND file_size = ? AND mtime_ns = ?
                                        """,
                                        (
                                            int(vec.size),
                                            self._to_blob(vec),
                                            float(time.time()),
                                            canonical_fp,
                                            self.model_name,
                                            int(st.st_size),
                                            self._mtime_ns_from_stat(st),
                                        ),
                                    )
                                    indexed += 1
                                    sem_success += 1
                                    batch_writes += 1
                                    if batch_writes >= commit_every:
                                        conn.commit()
                                        batch_writes = 0
                                except Exception as e:
                                    logger.warning(
                                        "[INDEX] Skipping semantic embedding for %s: %s",
                                        os.path.basename(canonical_fp),
                                        e,
                                    )
                                    failed += 1
                                    sem_fail += 1
                                    try:
                                        self._mark_semantic_skipped(canonical_fp, st, conn)
                                        self._store_face_count(
                                            canonical_fp, 0, conn=conn, commit=False
                                        )
                                        batch_writes += 1
                                        if batch_writes >= commit_every:
                                            conn.commit()
                                            batch_writes = 0
                                    except Exception as mark_exc:
                                        logger.debug(
                                            "[INDEX] Could not mark skipped for %s: %s",
                                            os.path.basename(canonical_fp),
                                            mark_exc,
                                        )
                                if progress_callback and (
                                    i <= 2 or i >= total_sem or (i % 5 == 0)
                                ):
                                    progress_callback(
                                        int(progress_indexed_base + total_sem * 0.1 + i * 0.9),
                                        progress_album_total,
                                        format_index_progress("Semantic", int(progress_indexed_base + total_sem * 0.1 + i * 0.9), progress_album_total),
                                    )
                                if throttle_sec > 0:
                                    time.sleep(throttle_sec)

                        if i % 25 == 0 or i == total_sem:
                            elapsed_so_far = max(1e-9, time.time() - t_sem_start)
                            throughput_so_far = float(i) / elapsed_so_far
                            logger.info(
                                "[INDEX][SPEED] Semantic progress: %d/%d in %.3fs (%.2f img/s)",
                                i,
                                total_sem,
                                elapsed_so_far,
                                throughput_so_far,
                            )
                        import gc
                        gc.collect()
                    if batch_writes:
                        conn.commit()
                        batch_writes = 0
                    sem_elapsed = max(1e-9, time.time() - t_sem_start)
                    sem_avg = sem_elapsed / max(1, total_sem)
                    sem_throughput = float(total_sem) / sem_elapsed
                    encode_elapsed = max(1e-9, t_encode_cpu)
                    warm_throughput = (
                        float(len(warm_paths)) / max(1e-9, t_warm_elapsed)
                        if warm_paths
                        else 0.0
                    )
                    encode_throughput = float(total_sem) / encode_elapsed
                    wall_phases = max(1e-9, t_warm_elapsed + sem_elapsed)
                    logger.info(f"[INDEX] Completed AI neural pass in {sem_elapsed:.4f}s.")
                    logger.info(
                        "[INDEX][SPEED] Semantic pass stats: total=%d success=%d failed=%d elapsed=%.3fs avg=%.4fs/img throughput=%.2f img/s",
                        total_sem,
                        sem_success,
                        sem_fail,
                        sem_elapsed,
                        sem_avg,
                        sem_throughput,
                    )
                    logger.info(
                        "[INDEX][SPEED] Phase split: warm=%.3fs (%.2f img/s, warmed=%d) "
                        "encode=%.3fs (%.2f img/s) encode_wall=%.3fs warm_share=%.0f%%",
                        t_warm_elapsed,
                        warm_throughput,
                        int(warm_count),
                        encode_elapsed,
                        encode_throughput,
                        sem_elapsed,
                        100.0 * t_warm_elapsed / wall_phases,
                    )
                else:
                    logger.info(f"[INDEX] Skipping AI features neural pass (MobileCLIP) for {total_sem} files (metadata-only index).")

                    # Thumbnail warm-up runs once in the user semantic pass (before MobileCLIP).
                    # Skip here when semantic search is enabled to avoid duplicate work and a
                    # second misleading "Semantic" progress phase in the search field.
                    if semantic_embeddings_enabled():
                        pass
                    else:
                        # Skip background thumbnail warming for metadata-only index to avoid
                        # disk/CPU contention and locking up the gallery image loader.
                        pass

                    for canonical_fp, st in pending_for_semantic:
                        self._wait_if_paused()
                        self._mark_metadata_ready_without_embedding(canonical_fp, st, conn)
                        indexed += 1
                    conn.commit()

            # Phase 3: face scan (optional; skipped when deferred to a follow-up worker).
            if run_face_inline:
                # Face detection decodes images at scale — heavy, so throttle here.
                self._acquire_load_throttle()
                self.release_mobileclip_gpu_memory()
                logger.info(
                    "[INDEX] Face scanning %d files (semantic pending was %d)",
                    total_face,
                    total_sem,
                )
                if progress_callback:
                    progress_callback(
                        progress_indexed_base,
                        progress_album_total,
                        format_index_progress("Face", 0, total_face),
                    )
                self._run_parallel_face_scan(
                    face_pending,
                    conn,
                    progress_callback,
                    conn_lock=threading.Lock(),
                    commit_every=commit_every,
                    progress_album_total=progress_album_total,
                    progress_indexed_base=progress_indexed_base,
                )
        except IndexAborted:
            logger.info("[INDEX] Indexing aborted (folder scope changed)")
        finally:
            # Always drop the ILM throttle so concurrency is restored for the gallery.
            self._release_load_throttle()
            conn.close()

        duration = time.time() - t_start
        logger.info(f"[INDEX] Finished indexing process in {duration:.4f}s. Results -> indexed: {indexed}, skipped: {skipped}, failed: {failed}, total: {total}")
        faces_deferred = total_face > 0 and not run_face_inline
        return {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "faces_deferred": int(faces_deferred),
            "faces_pending": total_face if faces_deferred else 0,
        }

    def get_index_coverage(self, file_paths: Sequence[str]) -> dict[str, int]:
        """
        Return index coverage for the given file set.
        Metadata-lazy version: only checks database presence, does not touch disk.
        """
        if not file_paths:
            return {"total": 0, "indexed": 0, "missing": 0, "ready": 1}

        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        total = len(indexable_paths)
        canonical_paths = [self._canonical_path(p) for p in indexable_paths if p]

        # Bulk lookup in database
        ",".join(["?"] * len(canonical_paths))
        # SQLite has a limit on parameters (usually 999), so we batch if needed.
        batch_size = 900
        indexed_count = 0

        try:
            for i in range(0, len(canonical_paths), batch_size):
                batch = canonical_paths[i : i + batch_size]
                qs = ",".join(["?"] * len(batch))
                cursor = self._conn.execute(
                    f"SELECT COUNT(*) FROM semantic_index WHERE file_path IN ({qs}) AND semantic_ready IN (1, -1) AND model_name = ?",
                    [*batch, self.model_name],
                )
                indexed_count += cursor.fetchone()[0]
        except Exception:
            # Fallback if table doesn't exist or other error
            indexed_count = 0

        missing = max(0, total - indexed_count)
        return {
            "total": total,
            "indexed": indexed_count,
            "missing": missing,
            "ready": 1 if missing == 0 else 0,
        }

    def get_semantic_embedding_coverage(self, file_paths: Sequence[str]) -> dict[str, int]:
        """Count corpus rows with real CLIP embeddings (semantic_ready=1, dim>0)."""
        indexable_paths, _ = self._filter_duplicate_raw_companions(file_paths)
        total = len(indexable_paths)
        if not total:
            return {"total": 0, "embeddable": 0, "skipped": 0, "pending": 0}

        rows = self._fetch_rows_for_paths(indexable_paths)
        row_map = {self._canonical_path(str(r["file_path"])): r for r in rows}
        embeddable = skipped = pending = 0
        for p in indexable_paths:
            row = row_map.get(self._canonical_path(p))
            if not row:
                pending += 1
                continue
            sr = _coerce_int(self._row_value(row, "semantic_ready"))
            dim = _coerce_int(self._row_value(row, "dim"))
            if sr == 1 and dim > 0:
                embeddable += 1
            elif sr == -1:
                skipped += 1
            else:
                pending += 1
        return {
            "total": total,
            "embeddable": embeddable,
            "skipped": skipped,
            "pending": pending,
        }
