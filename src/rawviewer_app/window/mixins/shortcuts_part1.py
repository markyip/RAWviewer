"""shortcuts mixin, part 1 of 1 (19 methods, 574 lines)."""
from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QMessageBox

from core import theme

from .. import lazy
from ..paths import _require_str_path

lazy.register(sys.modules[__name__])


class ShortcutsPart1:
    def _format_shortcut_line_html(self, line: str) -> str:
        """Bold the shortcut keyword before the em dash for quick scanning."""
        line = line.strip()
        if " -- " in line:
            key, desc = line.split(" -- ", 1)
            return f"<b>{key.strip()}</b> -- {desc.strip()}"
        return f"<b>{line}</b>"

    def _keyboard_shortcuts_help_text(self):
        """Plain-text shortcuts list for tooltips and the shortcuts dialog."""
        if getattr(self, "view_mode", "single") == "compare":
            return (
                "Space -- Toggle synchronized zoom (F on: zoom each pane to its focus point)\n"
                "F -- Toggle focus overlays on both panes\n"
                "← / → -- Previous / next candidate image\n"
                "↑ -- Promote candidate to selected\n"
                "↓ -- Reject candidate (Shift + ↓ to reject selected)\n"
                "Delete -- Delete candidate image (Shift + Delete to delete selected)\n"
                "J -- Toggle clipping overlay on both panes\n"
                "G -- Cycle composition guide on both panes\n"
                "C / Esc -- Exit Compare mode"
            )
        from raw_adjustments import editing_features_enabled

        # Editor-open: show Adjust-relevant shortcuts (H/M are browse-only).
        if getattr(self, "_adjust_overlay_visible", False):
            lines = [
                "E / Esc -- Close Adjust panel",
                "Space / Double-click -- Toggle fit-to-window / 100% zoom",
                "Trackpad Pinch / Ctrl+Scroll -- Smooth zoom in/out",
                "Dodge/Burn/Heal armed: two-finger scroll -- Brush Size",
                "D / B / X / H -- Arm Dodge / Burn / Eraser / Heal (again to disarm)",
                "M -- Toggle Mask overlay (armed, or an existing dodge/burn/heal edit)",
                "← / → -- Nudge focused slider (or previous/next when none focused)",
                "J -- Show/hide highlight/shadow clipping",
                "G -- Cycle composition guide",
                "F -- Show/hide focus point",
            ]
            return "\n".join(lines)

        lines = [
            "Space / Double-click -- Toggle fit-to-window / 100% zoom",
            "Trackpad Pinch / Ctrl+Scroll -- Smooth zoom in/out",
            "← / → -- Previous / next image",
            "0–5 -- Star rating (0 clears; click bottom stars too)",
            "Ctrl/Cmd+A -- Gallery: select all shown images",
            "↓ -- Move image(s) to Discard folder",
            "Delete -- Delete image(s)",
            "Esc -- Gallery: clear selection / exit filters; single view: back to gallery",
            "Gallery: Ctrl/Cmd+click -- toggle selection; Shift+click two photos for range",
            "Gallery: ↑ / ↓ -- Scroll (↓ with selection: move selection to Discard)",
            "C -- Toggle Compare mode (requires multiple images selected)",
            "H -- Show/hide histogram",
            "J -- Show/hide highlight (red) and shadow (blue) clipping",
            "P -- RAW only: shadow/highlight recovery preview (fit only, half-res; session)",
            "G -- Cycle composition guide (off / 3×3 / diagonal / both / phi grid)",
            "F -- Show/hide focus point",
            "M -- Show/hide location map",
        ]
        if editing_features_enabled():
            lines.append("E -- Show/hide adjustment panel (XMP sidecar)")
        return "\n".join(lines)

    def _keyboard_shortcuts_tooltip(self) -> str:
        """Rich tooltip for the status-bar hint button (padded, no drag-and-drop note)."""
        lines = [
            ln.strip()
            for ln in self._keyboard_shortcuts_help_text().split("\n")
            if ln.strip()
        ]
        body = "<br/>".join(self._format_shortcut_line_html(ln) for ln in lines)
        return (
            "<html><head/><body style='margin:0;'>"
            "<div style='padding:10px 14px; line-height:1.45;'>"
            f"{body}"
            "</div></body></html>"
        )

    def _set_shortcuts_hint_hovered(self, hovered: bool):
        """Toggle hint prominence without affecting layout width."""
        if not hasattr(self, "shortcuts_hint_button"):
            return
        btn = self.shortcuts_hint_button
        if hovered:
            btn.setEnabled(True)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(f"""
                QPushButton {{
                    color: {theme.INK_MUTED};
                    font-size: 11px;
                    font-weight: 600;
                    padding: 0px;
                    border: none;
                    background: transparent;
                    border-radius: 11px;
                    min-width: 22px;
                    max-width: 22px;
                    min-height: 22px;
                    max-height: 22px;
                }}
                QPushButton:hover {{
                    color: {theme.INK};
                    background-color: rgba(255, 255, 255, 0.08);
                }}
                QPushButton:pressed {{
                    background-color: rgba(255, 255, 255, 0.12);
                }}
            """)
        else:
            btn.setEnabled(False)
            btn.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            # Keep the button in layout so metadata text does not shift.
            btn.setStyleSheet("""
                QPushButton {
                    color: rgba(136, 136, 136, 0);
                    font-size: 11px;
                    font-weight: 600;
                    padding: 0px;
                    border: none;
                    background: transparent;
                    border-radius: 11px;
                    min-width: 22px;
                    max-width: 22px;
                    min-height: 22px;
                    max-height: 22px;
                }
            """)

    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts dialog"""
        lines = [
            ln.strip()
            for ln in self._keyboard_shortcuts_help_text().split("\n")
            if ln.strip()
        ]
        body = "<br/>".join(self._format_shortcut_line_html(ln) for ln in lines)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setText("Available Keyboard Shortcuts:")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setInformativeText(
            f"<div style='line-height:1.5;padding:4px 2px 0 2px;'>{body}</div>"
        )
        msg_box.exec()

    def _shortcut_blocked_by_text_input(self) -> bool:
        focused = self.focusWidget()
        from PyQt6.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit

        if not focused or not isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return False
        # Locked/read-only search field (indexing) must not swallow E / Esc /
        # nav shortcuts -- otherwise the editor looks unreachable after search.
        try:
            if focused.isReadOnly():
                return False
        except Exception:
            pass
        return True

    def _should_route_global_shortcut(self, obj) -> bool:
        """True when app shortcuts should run via the event filter (before child widgets)."""
        if self._shortcut_blocked_by_text_input():
            return False
        if not self.isVisible():
            return False
        app = QApplication.instance()
        active = app.activeWindow() if app is not None else None
        if active is not self:
            return False
        if obj is self:
            return True
        if self._is_widget_in_main_window(obj):
            return True
        try:
            from PyQt6.QtGui import QWindow

            wh = self.windowHandle()
            if isinstance(obj, QWindow) and wh is not None and obj is wh:
                return True
        except Exception:
            pass
        # App-level filter: receiver may be QApplication while our window is active.
        return active is self and app is not None and obj is app

    def _shortcut_activate_raw_recovery(self) -> None:
        if getattr(self, "view_mode", "") != "single":
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._toggle_auto_tone_preview()

    def _shortcut_activate_adjust_panel(self) -> None:
        from raw_adjustments import editing_features_enabled

        if not editing_features_enabled():
            return
        if getattr(self, "view_mode", "") != "single":
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._toggle_adjust_panel()

    def _nav_shortcut_defers_to_slider(self, step: int) -> bool:
        """When an Adjust-panel slider has (click) focus, Left/Right nudge its
        value by one step instead of navigating images. The nav shortcuts are
        ApplicationShortcut-scoped, so they fire before the focused widget
        ever sees the key -- the deferral has to happen here."""
        try:
            from PyQt6.QtWidgets import QApplication

            from rawviewer_ui.widgets.adjust import AdjustSlider

            w = QApplication.focusWidget()
            if isinstance(w, AdjustSlider) and w.isVisible():
                w.setValue(w.value() + step * w.singleStep())
                # Arrow nudges never pass through sliderReleased; persist the
                # same way a click-release does.
                w.sliderReleased.emit()
                return True
        except Exception:
            pass
        return False

    def _shortcut_activate_nav_prev(self) -> None:
        if self._nav_shortcut_defers_to_slider(-1):
            return
        if getattr(self, "view_mode", "") == "compare":
            session = getattr(self, "_comparison_session", None)
            if session is not None and not session.complete:
                session.step_candidate(-1)
                self._refresh_compare_panes(changed="candidate")
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._navigate_single_view_step(-1)

    def _shortcut_activate_nav_next(self) -> None:
        if self._nav_shortcut_defers_to_slider(+1):
            return
        if getattr(self, "view_mode", "") == "compare":
            session = getattr(self, "_comparison_session", None)
            if session is not None and not session.complete:
                session.step_candidate(1)
                self._refresh_compare_panes(changed="candidate")
            return
        if getattr(self, "view_mode", "single") != "single":
            return
        if self._shortcut_blocked_by_text_input():
            return
        self._navigate_single_view_step(1)

    def _shortcut_activate_escape(self) -> None:
        if self._shortcut_blocked_by_text_input():
            return
        vm = getattr(self, "view_mode", "single")
        if vm == "compare":
            self._exit_compare_mode()
            return
        if vm == "gallery":
            if getattr(self, "_burst_group_view_active", False):
                session = getattr(self, "_comparison_session", None)
                scroll_to = getattr(session, "select", None) if session else None
                if not scroll_to:
                    members = self._burst_group_view_visible_members()
                    scroll_to = members[0] if members else getattr(self, "current_file_path", None)
                self._exit_burst_group_view(scroll_to=scroll_to)
                return
            if self._gallery_has_selection():
                self._clear_gallery_selection()
                return
            if self._clear_gallery_bookmark_filter():
                return
            if self._clear_gallery_rating_filter():
                return
        elif vm == "single":
            if getattr(self, "_adjust_overlay_visible", False):
                self._close_adjust_panel_from_ui()
                return
            if getattr(self, "_burst_group_view_active", False):
                if not self._path_in_burst_group_view(getattr(self, "current_file_path", None)):
                    self._exit_burst_group_view()
                self.view_mode = "gallery"
                self._show_gallery_view()
                return
            if self._is_exif_sort_ready():
                self.toggle_view_mode()

    def _shortcut_activate_undo(self) -> None:
        if self._shortcut_blocked_by_text_input():
            return
        path = getattr(self, "current_file_path", None)
        if not path:
            return
        stack = getattr(self, "_undo_stack", {})
        history = stack.get(path, [])
        if not history:
            if hasattr(self, "status_bar"):
                self.status_bar.showMessage("Nothing to undo for this image", 2500)
            return

        old_adj = history.pop()

        self._disable_undo_tracking = True
        try:
            panel = getattr(self, "single_image_adjust_panel", None)
            if panel is not None:
                panel.set_adjustments(old_adj)
            # Mask is not a panel slider -- restore the live buffer before
            # editing_finished re-serializes whatever is in _dodge_burn_mask.
            self._apply_dodge_burn_mask_from_adj(old_adj)
            self._apply_spot_heal_mask_from_adj(old_adj)
            if panel is not None:
                self._on_adjust_panel_editing_finished(old_adj)
        finally:
            self._disable_undo_tracking = False

        if hasattr(self, "status_bar"):
            self.status_bar.showMessage("Undo applied", 2500)

    def _shortcut_activate_compare(self) -> None:
        if self._shortcut_blocked_by_text_input():
            return
        if getattr(self, "view_mode", "") == "compare":
            self._exit_compare_mode()
            return
        if self._compare_session_plan() is not None:
            self._enter_compare_mode()

    def _shortcut_activate_gallery_up(self) -> None:
        if self._shortcut_blocked_by_text_input():
            return
        vm = getattr(self, "view_mode", "single")
        if vm == "compare":
            self._compare_handle_promote()
            return
        if vm == "single":
            # Single view uses star ratings (0–5 / bottom stars); ↑ is unused here.
            if hasattr(self, "status_bar") and self.status_bar is not None:
                self.status_bar.showMessage(
                    "Rate with 1–5 (0 clears). ↑ is not used in single view.",
                    3500,
                )
            return
        if vm != "gallery":
            return
        # Gallery: ↑ scrolls only. Bookmarks are retired (no filter UI); use
        # bottom rating stars to narrow the grid.
        self._scroll_gallery_vertical(-1)

    def _shortcut_activate_gallery_down(self) -> None:
        if self._shortcut_blocked_by_text_input():
            return
        vm = getattr(self, "view_mode", "single")
        if vm == "compare":
            from PyQt6.QtWidgets import QApplication

            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                self._compare_handle_reject_select()
            else:
                self._compare_handle_reject_candidate()
            return
        if vm == "gallery":
            if self._gallery_has_selection():
                self.discard_gallery_selection()
                return
            self._scroll_gallery_vertical(1)
            return
        if vm == "single":
            self.move_current_image_to_discard()

    def _handle_app_shortcut(self, event):
        """Handle application-wide shortcuts for better consistency and focus-resilience."""
        focused = self.focusWidget()
        from PyQt6.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit
        if focused and isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return False
        # Toggle-style keys here (P, H, M, F, G, J, ...) must fire once per physical
        # press, matching the sibling QShortcut objects for the same keys which
        # explicitly disable auto-repeat. This function is reached via both
        # keyPressEvent and an eventFilter, neither of which gets that guard for
        # free -- an OS key-repeat from holding the key even slightly past the
        # repeat-delay threshold would silently double-toggle (e.g. "P" for RAW
        # recovery: on, then immediately off again from the repeat, netting no
        # visible change on that press).
        if event.isAutoRepeat():
            return False

        key = event.key()
        modifiers = event.modifiers()
        vm = getattr(self, "view_mode", "single")

        if vm == "single" and getattr(self, "_adjust_overlay_visible", False):
            ctrl_or_cmd = bool(modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
            if ctrl_or_cmd:
                if key == Qt.Key.Key_C:
                    panel = getattr(self, "single_image_adjust_panel", None)
                    if panel is not None:
                        panel._on_copy_settings_clicked()
                        if hasattr(self, "status_bar") and self.status_bar:
                            self.status_bar.showMessage("Edit settings copied", 2000)
                        return True
                elif key == Qt.Key.Key_V:
                    panel = getattr(self, "single_image_adjust_panel", None)
                    if panel is not None:
                        from rawviewer_ui.widgets.adjust import _edit_settings_clipboard
                        if _edit_settings_clipboard() is not None:
                            panel._on_paste_settings_clicked()
                            if hasattr(self, "status_bar") and self.status_bar:
                                self.status_bar.showMessage("Edit settings pasted", 2000)
                        else:
                            if hasattr(self, "status_bar") and self.status_bar:
                                self.status_bar.showMessage("No edit settings to paste", 2000)
                        return True
            if key == Qt.Key.Key_M and not ctrl_or_cmd:
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None:
                    panel.toggle_dodge_burn_show_mask()
                    return True
            if key == Qt.Key.Key_D and not ctrl_or_cmd:
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None and hasattr(panel, "set_dodge_burn_mode"):
                    panel.set_dodge_burn_mode("dodge")
                    return True
            if key == Qt.Key.Key_B and not ctrl_or_cmd:
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None and hasattr(panel, "set_dodge_burn_mode"):
                    panel.set_dodge_burn_mode("burn")
                    return True
            if key == Qt.Key.Key_X and not ctrl_or_cmd:
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None and hasattr(panel, "set_dodge_burn_mode"):
                    panel.set_dodge_burn_mode("erase")
                    return True
            if key == Qt.Key.Key_H and not ctrl_or_cmd:
                panel = getattr(self, "single_image_adjust_panel", None)
                if panel is not None and hasattr(panel, "set_dodge_burn_mode"):
                    panel.set_dodge_burn_mode("heal")
                    return True

        if key == Qt.Key.Key_Escape:
            gv = getattr(self, "gpu_view", None)
            if gv is not None and gv.is_color_pick_mode():
                self._cancel_wb_picker()
                return True

        if key in (Qt.Key.Key_H, Qt.Key.Key_P, Qt.Key.Key_M) and vm == "compare":
            return True

        if Qt.Key.Key_0 <= key <= Qt.Key.Key_5 and vm == "single":
            rating = key - Qt.Key.Key_0
            self.rate_current_image(rating)
            return True

        if key == Qt.Key.Key_Space:
            if vm == "compare":
                self._compare_toggle_zoom_mode_a()
                return True
            if vm == "single":
                if (
                    hasattr(self, "current_file_path")
                    and self.current_file_path
                    and self.is_animated_image(_require_str_path(self.current_file_path))
                ):
                    return True
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[TRACK] User pressed spacebar to toggle zoom - file: "
                    f"{os.path.basename(_require_str_path(self.current_file_path)) if hasattr(self, 'current_file_path') and self.current_file_path else 'Unknown'}"
                )
                if getattr(self, "_focus_subject_outline_active", False):
                    # When already zoomed in, Space must still zoom out like normal;
                    # only from fit-to-window does Space jump to the focus box center.
                    if self._single_view_is_fit_mode():
                        if not self._focus_jump_to_subject_center():
                            self.toggle_zoom()
                    else:
                        self.toggle_zoom()
                    return True
                self.toggle_zoom()
                return True
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if vm == "single" and self._filmstrip_handles_input():
                bar = self._filmstrip_bar()
                if bar:
                    bar.commit_selection()
                    return True
        elif key == Qt.Key.Key_Delete or (
            sys.platform == "darwin" and key == Qt.Key.Key_Backspace
        ):
            # Mac: Backspace key is labeled "delete"; fn+Delete is Key_Delete (forward delete).
            if vm == "compare":
                shift_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self._compare_handle_delete(shift_pressed)
                return True
            if vm == "gallery" and self._gallery_has_selection():
                self.delete_gallery_selection()
                return True
            if vm == "single":
                self.delete_current_image()
                return True
        elif key == Qt.Key.Key_Escape:
            if self._gallery_has_selection():
                if vm == "gallery":
                    self._clear_gallery_selection()
                else:
                    self._request_clear_gallery_selection()
                return True
            self._shortcut_activate_escape()
            return True
        elif key == Qt.Key.Key_F:
            if vm == "single":
                self._toggle_focus_subject_outline()
                return True
        elif key == Qt.Key.Key_G:
            if vm == "single":
                self._toggle_composition_grid()
                return True
        elif key == Qt.Key.Key_A:
            ctrl_a = bool(
                modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
            )
            if ctrl_a and vm == "gallery" and self._gallery_files_for_display():
                self._gallery_select_all_displayed()
                return True
        elif key == Qt.Key.Key_J:
            if vm == "single":
                self._toggle_clipping_overlay()
                return True
        elif key == Qt.Key.Key_E:
            if vm == "single":
                self._shortcut_activate_adjust_panel()
                return True
        elif key == Qt.Key.Key_Left:
            self._shortcut_activate_nav_prev()
            return True
        elif key == Qt.Key.Key_Right:
            self._shortcut_activate_nav_next()
            return True
        elif key == Qt.Key.Key_C:
            if not (modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)):
                self._shortcut_activate_compare()
                return True
        elif key == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._shortcut_activate_undo()
            return True
        elif key == Qt.Key.Key_Up:
            self._shortcut_activate_gallery_up()
            return True
        elif key == Qt.Key.Key_Down:
            self._shortcut_activate_gallery_down()
            return True
        # Handle H (Histogram) separately to ensure it works even when no image is loaded
        # but only in single view mode.
        if key == Qt.Key.Key_H and vm == "single":
            self._shortcut_activate_histogram()
            return True

        if key == Qt.Key.Key_M and vm == "single":
            self._shortcut_activate_location_map()
            return True

        if key == Qt.Key.Key_P and vm == "single":
            self._shortcut_activate_raw_recovery()
            return True

        return False

    def keyPressEvent(self, event):
        if getattr(self, '_is_delete_dialog_open', False):
            event.ignore()
            return
        event.key()
        getattr(self, "view_mode", "single")

        # Try to handle common app shortcuts first
        if self._handle_app_shortcut(event):
            event.accept()
            return

        # Mode-specific keys handled via WindowShortcut (arrows, escape, up/down).
        super().keyPressEvent(event)
