"""Shared Qt widgets: thin facade over the widget submodules."""
from __future__ import annotations

from .batch_export_dialog import BatchExportDialog
from .confirm_dialog import CustomConfirmDialog, _ConfirmDialogButton
from .helpers import _filmstrip_ui_env_int, _norm_dir_key
from .macos import _activate_macos_foreground_app
from .metadata_bar import TopMetadataBar
from .overlays import LoadingOverlay, SingleImageViewOverlay
from .resize_grip import ResizeGripIndicator
from .share import (
    _open_windows_open_with_dialog,
    _resolve_windows_share_helper_exe,
    _share_windows_clipboard_cf_hdrop,
    _share_windows_clipboard_cf_hdrop_paths,
    _share_windows_clipboard_file_via_powershell,
    _share_windows_via_helper,
    _share_windows_via_winrt,
    _windows_shell_verb_suggests_share,
)
from .task_progress_dialog import TaskProgressDialog
from .thumbnail_label import ThumbnailLabel
from .title_bar import CustomTitleBar
from .warning_dialog import CustomWarningDialog

__all__ = [
    "_activate_macos_foreground_app",
    "_ConfirmDialogButton",
    "_filmstrip_ui_env_int",
    "_norm_dir_key",
    "_open_windows_open_with_dialog",
    "_resolve_windows_share_helper_exe",
    "_share_windows_clipboard_cf_hdrop",
    "_share_windows_clipboard_cf_hdrop_paths",
    "_share_windows_via_helper",
    "_share_windows_via_winrt",
    "_share_windows_clipboard_file_via_powershell",
    "_windows_shell_verb_suggests_share",
    "BatchExportDialog",
    "CustomConfirmDialog",
    "CustomTitleBar",
    "CustomWarningDialog",
    "LoadingOverlay",
    "ResizeGripIndicator",
    "SingleImageViewOverlay",
    "TaskProgressDialog",
    "ThumbnailLabel",
    "TopMetadataBar",
]
