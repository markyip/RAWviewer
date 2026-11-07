"""
Bootstrap / installer module: managed pixi-based installation of RAWviewer.

Provides InstallWorker (background install logic), InstallerGUI (setup wizard GUI),
and module-level helpers for the Windows installer executable.
"""

from .constants import (
    BUNDLE_DIR,
    DOWNLOAD_RETRIES,
    EXE_DIR,
    MIN_FREE_BYTES,
    MIN_FREE_BYTES_LITE,
    PIXI_DOWNLOAD_URL,
    PIXI_VERSION,
)
from .gui import InstallerGUI
from .helpers import (
    _apply_installer_process_icon,
    _check_disk_space,
    _cleanup_partial_install,
    _clear_existing_user_cache,
    _delete_reg_tree,
    _describe_download_error,
    _directory_size_bytes,
    _download_file_with_retry,
    _estimated_size_kb,
    _file_association_extensions,
    _human_bytes,
    _installed_app_icon_path,
    _installer_icon_path,
    _is_lite_installer,
    _kill_running_app_processes,
    _map_model_download_progress,
    _min_free_bytes_for_install,
    _parse_installer_progress_line,
    _patch_python_exe_branding,
    _prune_torch_link_libs,
    _register_file_associations,
    _run_logged_subprocess,
    _run_subprocess_with_retry,
    _user_cache_dir,
)
from .worker import InstallWorker


def main() -> None:
    """Entry point for the installer application."""
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    _apply_installer_process_icon()
    window = InstallerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
