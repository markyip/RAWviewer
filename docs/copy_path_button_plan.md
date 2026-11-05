# Gallery hover "Copy path" button — update plan

## Rephrased requirement

> **Gallery hover "copy path" button.** In the gallery grid view, when the pointer
> hovers over a thumbnail that has a loaded image, a small circular button with a
> copy icon appears at the tile's **bottom-right corner**. Clicking it copies that
> image file's absolute path to the system clipboard and shows a brief
> "Path copied to clipboard" status message. Clicking the button must **not** open
> the image, toggle selection, or start a drag.

## Scope

Minimal: one widget, one gallery press/release interception, one test. No new
dependencies, no new files at runtime, no changes to the clipboard helper.

---

## Proof-backed assumptions

| # | Assumption | Evidence |
|---|---|---|
| 1 | **Placement: bottom-right corner.** Top-left is the selection check (`QRect(6,6,20,20)`), top-right is the edited badge (`_position_edited_badge`, 16 px, margin 2), bottom-left is the rating badge (`_position_rating_badge`, margin 4). Bottom-right is the only free corner. | `widgets.py:37-38, 128-132, 328-342` |
| 2 | **Paint it, don't add a child widget.** The selection check was deliberately moved from per-tile QSS/child widgets to `paintEvent` because per-tile style re-polish was a top cost; the gallery is virtualized/pooled. | `widgets.py:144-157` comment; `_visible_widgets`/`_widget_pool` at `gallery_view.py:417-418` |
| 3 | **Hit-test belongs in the per-widget press wrapper**, not in `ThumbnailLabel`, because the wrapper already intercepts the check circle and swallows the release via `_press_on_check`. | `gallery_view.py:3667-3685, 3714-3722` |
| 4 | **Reuse the existing clipboard helper** `_copy_paths_to_clipboard([path])` — it already sets `QMimeData` with absolute path text **and** `QUrl.fromLocalFile` URLs, with a plain-text fallback. The existing "Copy file paths" menu action uses it. | `main.py:26022-26046`, wired at `main.py:25964-25968` |
| 5 | **No new 3rd-party library is needed.** `qtawesome` is already a declared dependency and `fa5s.copy` exists. | `pixi.toml:47`, `build.py:471`; verified `qta.icon('fa5s.copy')` succeeds |
| 6 | **Gate on loaded pixels**, like every existing badge, because tiles exist (and get metadata) before pixels arrive. | `widgets.py:223-227` |
| 7 | **Status feedback via `status_bar.showMessage(..., 2500)`**, matching the existing copy action. | `main.py:25968, 25998` |
| 8 | **The gallery's `parent_viewer` is the `RAWImageViewer`** and has `status_bar` and `_copy_paths_to_clipboard`. | `gallery_view.py:401, 895`; `main.py:9109` |

---

## Change list

### 1. `src/rawviewer_ui/widgets.py` — `ThumbnailLabel`

**State** (`__init__`, ~line 44):
- `_tile_hovered = False`, `_copy_hover = False`, `_press_on_copy_path = False`
- `_copy_icon_pixmap = None`, `_copy_icon_size = 0`

**Constants** (line 37-38):
- `_COPY_MARGIN = 6`, `_COPY_SIZE = 20` (match the check circle)

**New methods:**
- `copy_path_rect() -> QRect` — bottom-right, computed from current size
- `hits_copy_path_button(pos) -> bool` — circle hit test, radius = half + 4 px
  (same pattern as `hits_selection_check`)
- `_set_tile_hovered(hovered)` / `_set_copy_hover(hovered)` — bool guard + `update()`
- `_copy_icon(size) -> QPixmap` — lazy, cached;
  `qta.icon("fa5s.copy", color=theme.INK).pixmap(size, size)` with a hand-drawn
  two-rounded-rect fallback if qtawesome fails

**Event handlers:**
- `mouseMoveEvent` — also compute `_set_copy_hover(hits_copy_path_button(pos))`
- `enterEvent` — also set `_tile_hovered = True` and copy hover from pos
- `leaveEvent` — clear `_tile_hovered` and `_copy_hover`

**`paintEvent`** (after the existing check overlay, ~line 190):
- If `file_path and _thumb_pixels_ready() and _tile_hovered`: draw translucent
  backing circle + centered copy icon (icon = `_COPY_SIZE * 0.62`)

### 2. `src/rawviewer_ui/gallery_view.py`

- **`_clear_widget_thumbnail`** (line 949): reset `widget._tile_hovered = False`,
  `_copy_hover = False`, `_press_on_copy_path = False`
- **`_on_thumb_press`** (line 3667): after the check-circle test, add a
  `hits_copy_path_button` branch that sets `_press_on_copy_path`, clears
  `_drag_start_pos`, calls `pv._copy_paths_to_clipboard([target_path])` and
  `pv.status_bar.showMessage("Path copied to clipboard", 2500)`, then returns
- **`_on_thumb_release`** (line 3714): swallow the release when
  `_press_on_copy_path` is set (mirrors the `_press_on_check` branch)

### 3. New test — `testplan/auto/t_gallery_copy_path_button.py`

Following the `t_gallery_batch_selection.py` pattern (`check()` helper, offscreen
Qt, `sys.path` insert):

1. `copy_path_rect()` is in the bottom-right (x > width/2, y > height/2)
2. `copy_path_rect()` does not intersect `selection_check_rect()`
3. Hit-test true at rect center, false at tile center
4. `_copy_icon(size)` returns a non-null pixmap of the requested size
5. Hover painting with a real pixmap + `file_path` set → no exception
6. Hover painting makes **no** `setStyleSheet` call (style counter)
7. `leaveEvent` clears `_tile_hovered` / `_copy_hover`
8. `RAWImageViewer._copy_paths_to_clipboard(None, [path])` puts the absolute path
   on `QApplication.clipboard().text()` (self is unused, so no viewer
   construction needed)

### 4. Third-party library

**None required.** No install step. `qtawesome` is already present
(`pixi.toml:47`, `build.py:471`) and `fa5s.copy` is available.

---

## Optional polish (out of minimal scope)

- Pointing-hand cursor while over the button
- Native tooltip ("Copy file path")
- Hide the button on very small tiles (< 64 px) where a 20 px button dominates
- Fade animation (the codebase does not animate tile overlays today)

---

## Acceptance criteria

- Hovering a loaded thumbnail shows the copy button in the bottom-right corner.
- Moving the pointer off the tile hides it.
- Clicking the button copies the file's absolute path as plain text (and as a
  file URL via `QMimeData`).
- A "Path copied to clipboard" status message appears for ~2.5 s.
- Clicking the button does not open the image, change selection, or drag.
- The button does not overlap the selection check, rating, or edited badges.
- Hovering does not re-polish styles (`setStyleSheet` is never called).
- No new dependency is added.
