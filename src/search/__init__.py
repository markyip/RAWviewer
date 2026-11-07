"""Semantic search package."""

from search.backends import (
    MobileCLIPCoreMLBackend,
    MobileCLIPONNXBackend,
    _ClipBPETokenizer,
    copy_bgra_into_cvpixelbuffer,
    copy_numpy_into_mlmultiarray,
    mlmultiarray_to_numpy_f32,
    resolve_mobileclip_backend,
)
from search.engine import SemanticImageIndex
from search.indexer import (
    _get_shared_index_executor,
    face_scan_enabled,
    format_index_progress,
    metadata_auto_index_enabled,
    semantic_embeddings_enabled,
)
from search.queries import (
    CustomReverseGeocoder,
    get_semantic_capture_times_for_folder,
    get_semantic_capture_times_for_paths,
)
from search.types import IndexAborted, MetadataRow, ProgressCallback, SearchHit

__all__ = [
    "CustomReverseGeocoder",
    "IndexAborted",
    "MetadataRow",
    "MobileCLIPCoreMLBackend",
    "MobileCLIPONNXBackend",
    "ProgressCallback",
    "SearchHit",
    "SemanticImageIndex",
    "_ClipBPETokenizer",
    "_get_shared_index_executor",
    "copy_bgra_into_cvpixelbuffer",
    "copy_numpy_into_mlmultiarray",
    "face_scan_enabled",
    "format_index_progress",
    "get_semantic_capture_times_for_folder",
    "get_semantic_capture_times_for_paths",
    "metadata_auto_index_enabled",
    "mlmultiarray_to_numpy_f32",
    "resolve_mobileclip_backend",
    "semantic_embeddings_enabled",
]
