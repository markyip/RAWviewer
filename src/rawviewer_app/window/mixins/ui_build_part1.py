"""ui_build mixin, part 1 of 3 (2 methods, 60 lines)."""
from __future__ import annotations

import sys

from PyQt6.QtCore import QSize

from core import theme

from .. import lazy
from ..lazy import qta
from ..qt_helpers import _qta_icon

lazy.register(sys.modules[__name__])


class UiBuildPart1:
    def _sync_burst_grouping_button_icon(self) -> None:
        btn = getattr(self, "burst_grouping_button", None)
        if btn is None:
            return
        enabled = self._is_burst_grouping_enabled()
        try:

            color = theme.INK if enabled else theme.INK_FAINT
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.setIcon(_qta_icon("fa5s.layer-group", color=color))
            btn.blockSignals(False)
        except Exception:
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.blockSignals(False)

    def _sync_burst_group_bottom_chrome(self) -> None:
        burst = bool(getattr(self, "_burst_group_view_active", False))
        vm = getattr(self, "view_mode", "single")
        if hasattr(self, "search_bottom_button"):
            if burst:
                self.search_bottom_button.hide()
            else:
                self.search_bottom_button.setVisible(
                    bool(getattr(self, "image_files", None))
                    and self._is_exif_sort_ready()
                    and vm == "gallery"
                )
        if hasattr(self, "sort_toggle_button"):
            if burst:
                self.sort_toggle_button.hide()
            else:
                self.sort_toggle_button.setVisible(vm == "gallery")
        if hasattr(self, "size_slider"):
            self.size_slider.setVisible(vm == "gallery")
        if hasattr(self, "slideshow_bottom_button"):
            if burst:
                self.slideshow_bottom_button.hide()
            else:
                self._sync_slideshow_button_visibility()
        if burst and hasattr(self, "search_expand_container"):
            self._ensure_gallery_search_collapsed()
            self.search_expand_container.hide()
        if hasattr(self, "view_mode_button"):
            if burst and vm == "gallery":
                self.view_mode_button.show()
                self.view_mode_button.setToolTip("Back to main gallery")
                if qta is not None:
                    try:
                        self.view_mode_button.setIcon(_qta_icon("fa5s.th", color=theme.INK_MUTED))
                        self.view_mode_button.setIconSize(QSize(20, 20))
                        self.view_mode_button.setText("")
                    except Exception:
                        self.view_mode_button.setText("Gallery")
                else:
                    self.view_mode_button.setText("Gallery")
            elif vm == "gallery":
                self.view_mode_button.hide()
        self._sync_burst_grouping_button_visibility()
