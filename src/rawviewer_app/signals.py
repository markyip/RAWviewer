"""Qt signal carriers."""
from PyQt6.QtCore import QObject, pyqtSignal

class ImageLoaded(QObject):
    """Signal carrier for image loading - thread to UI communication"""
    loaded = pyqtSignal(int, object, int)  # index, QImage, generation (convert to QPixmap in UI thread)

class GalleryMetadataSignals(QObject):
    """Signal carrier for background gallery metadata fetching"""
    ready = pyqtSignal(dict, str)  # meta dictionary, folder_path

class FolderLoadSignals(QObject):
    """Signal carrier for background folder scan/sort work."""
    ready = pyqtSignal(object, object, object, object, str, object, object, float, float)
    error = pyqtSignal(object, str, str)

class QuickFolderIndexSignals(QObject):
    """Fast scandir + mtime sort for single-file open (navigation before EXIF sort finishes)."""
    ready = pyqtSignal(object, object, object, int, float)  # token, files, file_stats, start_idx, scan_s

class FolderSortRefineSignals(QObject):
    """Signal carrier for background EXIF sort refinement."""
    ready = pyqtSignal(int, list, dict)  # token, sorted_files, bulk_metadata


class FolderResortSignals(QObject):
    """Signal carrier for background manual folder re-sort (sort button)."""
    ready = pyqtSignal(int, list, dict)  # token, sorted_files, bulk_metadata


class SemanticSearchResortSignals(QObject):
    """Signal carrier for background capture-time re-sort of search hits."""
    ready = pyqtSignal(int, str, list, object, bool, dict)  # token, query, ranked_paths, hits, used_semantic, bulk_meta


class WebpDecodeSignals(QObject):
    """Signal carrier for background animated WebP frame decode."""
    ready = pyqtSignal(int, str, list, list)  # token, file_path, QImage frames, durations ms
    failed = pyqtSignal(str, str)  # file_path, error message

class SemanticIndexSignals(QObject):
    """Signal carrier for background semantic index build."""
    progress = pyqtSignal(object, int, int, str)  # token, current, total, basename
    done = pyqtSignal(object, object)             # token, result dict
    error = pyqtSignal(object, str)               # token, error

class SemanticIndexPrepSignals(QObject):
    """Signal carrier for background semantic index prep (coverage + pending checks)."""
    done = pyqtSignal(object, object, int)        # coverage dict, pending list, face_pending count
    error = pyqtSignal(str)                       # error message

class ReleaseUpdateCheckSignals(QObject):
    """Signal carrier for background GitHub release version check."""
    finished = pyqtSignal(dict)

