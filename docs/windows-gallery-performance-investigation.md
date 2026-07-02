# Windows 外接碟 Gallery 效能與穩定性調查

> 調查期間：2026-06-05 — 2026-06-30  
> 平台：Windows 11，PyQt6 + QOpenGLWidget（GPU 單張視圖），RAWviewer v2.5.0  
> 主要測試資料：`I:\Photos\London`（1136 張 ARW，33MP）、`K:\Photos\Canada`（3237 張）、`K:\Photos\Japan Trip`（6886 張）  
> 對照基準：macOS（commit `eb7f762` 一帶），M1 Air 16GB，同類外接/網路資料夾 gallery 正常

**狀態（2026-07-01 更新）：** 保留的修復 = `d8a35c4`（GL teardown + scroll-jump fix）、Windows 檔案對話框 crash fix（native dialog → Qt picker）。**效能優化已於 2026-07-01 revert**（`16b6788` 並發調高、`7cbd816`/`dbb9b94` byte-scan-first）—— 見下方第 8 節；決定先聚焦穩定性，之後再重新評估效能。手動驗證（TODO 項目 1）仍建議在發布前執行一次。

---

## 摘要

| 項目 | 結論 |
|------|------|
| 主要症狀 | 單張全解後切換 gallery → 捲動時程式 **native abort**（`-805306369` / `0xCFFFFFFF`），無 Python traceback |
| 根因（最終確認） | **非** gallery 並發 cap 本身，而是 Windows `QOpenGLWidget` 在 mode switch 時對 **33MP live GL pixmap** 做破壞性操作（clear / grab / crossfade snapshot） |
| 次要瓶頸 | 背景 metadata/semantic 索引與 gallery 縮圖搶 I/O；metadata-only 後誤判 semantic 就緒；雙重 `build_gallery`；LibRaw process pool 與 GPU 解碼並行 |
| Mac 為何較穩 | 無 moderate external cap、預設無 process pool、Metal GL 較穩、較高 gallery 並發 |
| 目前最佳結果 | Mac-like env + GL teardown 修復後，自動化測試 **gl_cleared=True**，gallery 捲動 30s+ 無 crash |

標準重現流程（所有測試一致）：**先開單張 → 等全解 → 切 gallery → 捲動**（非直接進 gallery）。

---

## 錯誤碼對照

| 錯誤碼 | 十六進位 | 意義 | 常見情境 |
|--------|----------|------|----------|
| `-805306369` | `0xCFFFFFFF` | 原生 abort（SIGABRT 類） | Gallery 切換、GL pixmap 操作、大資料夾暖機 |
| `-1073741819` | `0xC0000005` | ACCESS_VIOLATION | 快速導航 + GPU 解碼 + 背景索引並行（NEF 等） |

兩者皆為 **C++/驅動層崩潰**，log 通常看不到 Python `Traceback`。請搭配 `RAWVIEWER_FILE_LOG=1`、`RAWVIEWER_FATAL_DUMP=1` 與 `%LOCALAPPDATA%\RAWviewer\logs\` 分析。

---

## 發現的問題

### 1. Gallery 切換時 OpenGL pixmap 未安全釋放（**主因**）

**現象：** 單張模式以 GPU view 顯示 7008×4672 全解影像後切 gallery，數秒內 process 結束，exit `-805306369`。

**機制：**
- `GpuImageView` 使用 `QOpenGLWidget` viewport，大圖以 GL texture 形式常駐 GPU
- 若在 widget **仍可見** 時呼叫 `clear()`、`grabFramebuffer()`、或 crossfade 用的 viewport snapshot，多數 Windows GL 驅動會 **直接 abort**
- `fatal_dump` 曾指向 `_grab_viewport_snapshot_for_crossfade` 同類路徑

**證據：**
- 僅調低 `active_cap` 無法根治；完全跳過 GL clear（`gpu_gl_skip=True`）可延長存活但 GPU 記憶體不釋放
- 修復後 log：`gl_cleared=True gl_deferred=False gl_pending=False`

---

### 2. Gallery 進入時資源競爭過大

**現象：** 大資料夾（3000+ 張）切 gallery 後，`_update_gallery_view` 延遲十數秒才執行，隨後崩潰；或 layout 完成但 `build_gallery` / `load_visible_images` 未出現。

**機制：**
- `set_images()` 非同步排程 `build_gallery`，與縮圖載入、face/semantic 索引、LibRaw pool 同時搶 CPU/磁碟/Qt 事件迴圈
- 暖機期若無 cap，一次可排 40+ thumbnail decode
- 雙重 `build_gallery`（重複進入 `_show_gallery_view`）加劇壓力

---

### 3. 背景索引與使用者操作搶資源

**現象：**
- 快速方向鍵導航時 crash（`0xC0000005`）
- Gallery 載入期間 metadata 索引掃描 1136 路徑 pending DB
- Semantic thumbnail warm-up 16 workers 在導航期間仍跑 rawpy/TIFF

**機制：**
- `_warm_thumbnail_cache_for_semantic_index()` 未尊重 `pause_indexing`
- LibRaw/rawpy 非 thread-safe 並行 `imread`/`postprocess`
- GPU CUDA decode 與背景 I/O 疊加

---

### 4. Metadata / Semantic 就緒判斷錯誤

**現象：** `run_debug_full.bat` 下 metadata-only 完成後，UI 仍顯示 semantic skipped 或誤判已就緒。

**機制：**
- `_is_gallery_semantic_search_ready()` 在 metadata-only 後使用 stale session cache
- `build_index()` 在 metadata-only 路徑仍呼叫 `log_inference_acceleration_profile()`
- DB 已滿時仍啟動 worker 掃描 broad pending paths

---

### 5. Windows 與 macOS 預設行為差異（效能面）

| 項目 | Windows 預設 | macOS |
|------|--------------|-------|
| `moderate_external_cap` | 開（外接碟額外限速） | 無 |
| LibRaw process pool | 19 workers | 通常關閉 |
| Gallery active cap（快外接碟） | 受 `EXTERNAL_GALLERY_ACTIVE_CAP` 限制 | 可達 44–64 |
| Gallery active cap（**慢**外接碟） | `SLOW_GALLERY_ACTIVE_CAP` 預設 **8** | 同邏輯但 probe 常為 fast |
| OpenGL 穩定性 | 脆弱（mode switch / grab） | Metal 較穩 |

**實測：** `I:\Photos\London` 被 probe 為 **slow（~63.5 MB/s）**，因此即使設定 `RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP=24`，實際 active 仍為 **8**（走 slow tier，非 external tier）。

---

### 6. Semantic 索引與 Gallery 縮圖快取重用不足

**現象：** 1133 張 London 資料夾，semantic warm-up 僅 **430/1133** 命中 `ImageCache`，其餘 703 張重解 ~25s。

**機制：** Gallery 閒置前背景 metadata 已啟動；使用者捲動 gallery 產生的 preview 尚未大量進入 cache 時 semantic warm-up 已開始。

---

## 已實作的修復

### A. GL teardown（gallery 切換 — **核心修復**）

| 變更 | 檔案 | 說明 |
|------|------|------|
| `_release_single_view_heavy_buffers_for_gallery()` | `src/main.py` | 進 gallery 前取消 crossfade、釋放 full-res cache、呼叫 GPU view teardown |
| 先 hide 單張容器，延遲 50ms 再釋放 | `src/main.py` `_show_gallery_view` | 讓 Qt 完成 GL surface hide 後再 clear |
| `release_for_gallery_entry()` | `src/rawviewer_ui/gpu_image_view.py` | Windows + GL viewport：僅在 hidden 時 clear |
| `_safe_clear_opengl_pixmap()` | `gpu_image_view.py` | `makeCurrent()` → `clear()` → `doneCurrent()` |
| 延遲重試 clear（0 / 50 / 150 ms） | `gpu_image_view.py` | `_clear_for_gallery_if_hidden` |
| Windows crossfade GL skip | `gpu_image_view.py`, `main.py` | `capture_viewport_pixmap()` 在 GL 上回傳 None，避免 grab |
| Log 欄位 | `main.py` | `gl_cleared` / `gl_deferred` / `gl_pending` |

成功 log 範例：

```text
[GALLERY] Released single-view full-res buffers for gallery entry (DSC00734.ARW, gl_cleared=True gl_deferred=False gl_pending=False)
[GPU_VIEW] Clearing hidden OpenGL pixmap (4672x7008) for gallery entry
```

---

### B. Gallery 暖機與排程節流

| 變更 | 檔案 | 說明 |
|------|------|------|
| `begin_gallery_warmup()` | `gallery_view.py` | 依檔案數 2–10s 暖機，cap widgets/tasks/active |
| `_gallery_warmup_scheduling_budgets()` | `gallery_view.py` | 1136 張 → 約 14/10/18；2500+ → 6/4/8 |
| 暖機期 skip prefetch、延遲 thumbnail slot | `gallery_view.py` | 避免 `build_gallery` 期間重入 |
| 暖機期 skip viewport resize rebuild | `gallery_view.py` | 減少 layout 風暴 |
| `enter_gallery_warmup_throttle()` | `image_load_manager.py` | 暖機期降低 worker / RAW 並行 |
| `_defer_background_indexing_for_gallery()` | `main.py` | 暫停 semantic、作廢 face token、延長恢復 8–30s |
| `_gallery_search_needs_index_work` defer | `main.py` | Gallery 壓力期跳過 1136 路徑 pending scan |

---

### C. 載入佇列與 prefetch 清理

| 變更 | 檔案 | 說明 |
|------|------|------|
| `cancel_non_gallery_tasks()` | `image_load_manager.py` | 取消進行中 non-gallery 任務 |
| `cancel_queued_non_gallery_tasks()` | `image_load_manager.py` | 清空佇列中 stale full-res |
| `_cancel_filmstrip_prefetch_for_gallery()` | `main.py` | 停止 filmstrip/neighbor prefetch |
| `build_gallery` 去重 | `gallery_view.py` | 避免雙重 layout |

---

### D. 索引 / metadata 正確性

| 變更 | 檔案 | 說明 |
|------|------|------|
| `get_metadata_extraction_pending_paths()` | `semantic_search.py` | 只回傳需 EXIF 抽取的檔案 |
| metadata-only skip worker | `main.py` | `extract_pending=0` → `_finish_silent_metadata_without_worker()` |
| `log_inference_acceleration_profile` 條件化 | `semantic_search.py` | 僅 `run_semantic_embeddings=True` 時呼叫 |
| semantic ready 快取修正 | `main.py` | metadata-only 後清除 stale session cache |
| warm-up `_wait_if_paused()` | `semantic_search.py` | 導航期間暫停 thumbnail warm-up |
| `libraw_io_lock()` / `pause_gpu_decode()` | `gpu_raw_processor.py` 等 | 序列化 LibRaw 與 GPU 解碼 |

---

### E. Mac-like 開發環境預設（Windows）

已寫入 `pixi.toml` `[target.win-64.activation.env]` 與 `scripts/Launch/bat/run_debug.bat`：

```text
RAWVIEWER_MODERATE_EXTERNAL_CAP=0
RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP=24
RAWVIEWER_USE_PROCESS_POOL=0
```

**注意：** slow 外接碟仍需額外設定才會提高 active：

```text
RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP=24
RAWVIEWER_SLOW_GALLERY_MAX_TASKS=12
```

---

## 測試結果

### 測試矩陣（`I:\Photos\London`，1136 ARW，流程：單張全解 → gallery → 捲動）

| 情境 | 環境 | 結果 | 備註 |
|------|------|------|------|
| 舊版 + active=8 + 永不 GL clear | 預設 Windows | **Crash** ~8–11s | GPU texture 未釋放或 clear 時 abort |
| GL 修復 + active=8 | `gpu_gl_skip` 時期 | **成功** ~5 min，exit 0 | `gl_cleared=True`；當時跳過 clear |
| GL 修復 + active=16，無 Mac env | 預設 | **Crash** ~18:09 | 雙重 `build_gallery`、GL 未釋放 |
| GL 強化 + Mac-like env | `MODERATE=0`, `ACTIVE=24`, `POOL=0` | **GL OK**，捲動 30s+ | `gl_cleared=True gl_deferred=False gl_pending=False`；active 實測 **8**（slow tier） |
| 自動化 stress（`stress_main_auto.py`） | 同上 + `AUTO_METADATA=0` | GL teardown 通過；腳本 ~30s 後 UI 凍結 | 建議以手動操作做最終確認 |

### 成功 log 特徵

```text
[LOAD] LibRaw process pool disabled
[GALLERY] Released single-view full-res buffers for gallery entry (..., gl_cleared=True gl_deferred=False gl_pending=False)
[INDEX] Deferring background metadata indexing (gallery warming/loading)
Broad pending=N, need EXIF extract=0 → skipping index worker   # DB 已滿時
```

### macOS 對照

- 同資料夾（更多圖片）gallery 正常，無 `-805306369`
- 無 moderate external cap、無 process pool 預設負擔較小
- 未觀察到 mode switch 時 GL pixmap clear 導致 abort

---

## 環境變數參考

### 穩定性 / 隔離

| 變數 | 建議值 | 用途 |
|------|--------|------|
| `RAWVIEWER_FILE_LOG` | `1` | 寫入 `%LOCALAPPDATA%\RAWviewer\logs\` |
| `RAWVIEWER_FATAL_DUMP` | `1` | Native crash 時傾印線程 |
| `RAWVIEWER_DISABLE_CROSSFADE` | `1` | 關閉 crossfade（減少 GL grab） |
| `RAWVIEWER_GPU_VIEW_NO_GL` | `1` | 強制非 GL viewport（隔離 GL 問題） |
| `RAWVIEWER_AUTO_METADATA_INDEX` | `0` | 測試時關閉背景 metadata |

### Gallery 並發

| 變數 | 預設 | 說明 |
|------|------|------|
| `RAWVIEWER_GALLERY_ACTIVE_CAP` | 64 | 本機 / 非外接 |
| `RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP` | 16（dev 設 24） | 快外接碟 |
| `RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP` | **8** | **慢**外接碟（probe < 120 MB/s） |
| `RAWVIEWER_MODERATE_EXTERNAL_CAP` | `1`（dev 設 `0`） | Windows 外接碟 moderate 限速 |
| `RAWVIEWER_GALLERY_INDEX_DEFER_ACTIVE` | 4 | gallery active ≥ 此值時 defer metadata |
| `RAWVIEWER_METADATA_INDEX_GALLERY_RETRY_MS` | 2000 | defer 重試間隔 |

### 載入器

| 變數 | 說明 |
|------|------|
| `RAWVIEWER_USE_PROCESS_POOL` | `0` = 關 LibRaw process pool（Mac-like） |
| `RAWVIEWER_GALLERY_WARMUP_MAX_WORKERS` | 暖機期 worker 上限 |

---

## 關鍵程式位置

```
src/main.py
  _release_single_view_heavy_buffers_for_gallery()   ~11517
  _show_gallery_view()                               ~11587  (hide + 50ms delay)
  _defer_background_indexing_for_gallery()
  _maybe_start_background_metadata_index()

src/rawviewer_ui/gpu_image_view.py
  release_for_gallery_entry()                        ~373
  _safe_clear_opengl_pixmap()                        ~406
  capture_viewport_pixmap()                          ~443  (Windows GL → None)

src/rawviewer_ui/gallery_view.py
  _apply_external_gallery_caps()                     ~84
  begin_gallery_warmup()                             ~1543
  _gallery_warmup_scheduling_budgets()               ~133

src/image_load_manager.py
  cancel_non_gallery_tasks() / cancel_queued_non_gallery_tasks()
  enter_gallery_warmup_throttle()

src/semantic_search.py
  get_metadata_extraction_pending_paths()
  build_index() metadata-only 路徑
```

---

## 測試腳本

| 腳本 | 用途 |
|------|------|
| `scripts/stress_main_auto.py` | 全 GUI：單張 → 全解 → gallery → 捲動（subprocess 捕獲 exit code） |
| `scripts/stress_main_worker.py` | 上述 worker 實作 |
| `scripts/stress_gallery_thumbs.py` | Headless 縮圖壓力（`QT_QPA_PLATFORM=offscreen`） |
| `scripts/run_gallery_stress_tests.ps1` | 多組 env 隔離測試 |
| `scripts/Launch/bat/run_debug.bat` | 日常 dev，已含 Mac-like env |

手動驗證命令（`pixi shell`）：

```powershell
$env:RAWVIEWER_MODERATE_EXTERNAL_CAP = "0"
$env:RAWVIEWER_EXTERNAL_GALLERY_ACTIVE_CAP = "24"
$env:RAWVIEWER_USE_PROCESS_POOL = "0"
$env:RAWVIEWER_AUTO_METADATA_INDEX = "0"
pixi run python src/main.py I:\Photos\London\DSC00734.ARW
```

---

## 待辦與建議

1. **手動確認**（仍待辦）：Mac-like env + `I:\London` 全解 → gallery → 捲動 3–5 分鐘，確認無 crash 且 log 有 `gl_cleared=True`。自動化 stress（cold cache、預設 Windows 設定、Mac-like 設定）已通過，但尚未有人手動互動驗證。
2. **Slow 碟並發**：`RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP` 預設已從 8 提高到 32（見第 8 節），`I:\` 上的並發已對齊 fast-scroll 預算，不再需要手動覆寫。
3. ~~**Commit 時機**~~：已完成 — `d8a35c4`（GL teardown + scroll-jump 修復）、`10afee6`（docs/scripts）、`16b6788`（並發調高），rebase 到 `eb7f762`（XMP sidecar）之上後 push 至 `origin/main`。
4. **Profile 分離**：考慮「穩定 profile」（低並發、無 pool）與「效能 profile」（高並發）分開，避免 dev 預設影響所有 Windows 使用者。尚未實作。
5. **自動化 stress 凍結**：`stress_main_auto.py` 約 30s 後 UI 無輸出，可能與自動捲動實作有關，不影響 GL 修復驗證結論。尚未調查根因。

---

## 7. Scrollbar 大幅跳轉後新可視區延遲載入（**2026-06-30 新發現並修復**）

**現象：** 使用者直接拖曳 scrollbar 到遠處（而非連續捲動）時，新可視區的縮圖不會立即開始載入，而是要等待原本（已離開畫面）位置的縮圖載入完才會開始 — 感覺像「卡住」幾秒。

**根因（兩處，皆在 `gallery_view.py` `load_visible_images()`）：**

1. 大跳轉偵測 (`abs(scroll_y - self._last_scheduled_scroll_y) > v_h*3`) 觸發時，除了呼叫 `load_manager.flush_queue()`（只清空 **load manager** 自己的佇列/追蹤），還會 `self._requested_thumbnail_paths.clear()`。但 gallery 自己另有一份 `self._active_tasks`（用於 `active_cap` 計數），其清理邏輯（`to_cancel = self._requested_thumbnail_paths - wanted_paths`）正好需要靠 `_requested_thumbnail_paths` 才能算出該清掉哪些路徑。清空它等於讓這段清理邏輯失效，舊位置的 `_active_tasks` entries 永遠不會被正確地 `pop()`。
2. 即使 (1) 修掉，`_gallery_prefetch_center_index()` 在「entry anchor 尚未進入可視範圍」（`_entry_prefetch_active()`，純粹用矩形位置判斷，無時間/跳轉感知）時，會持續把 `center_idx` 釘在進入 gallery 時開啟的那張圖（如 `DSC00734.ARW`）。大跳轉後該圖永遠不會回到可視範圍，所以這個錨定**永久生效**，`center_paths` 持續把舊位置一批路徑塞回 `wanted_paths`，讓它們躲過所有 cancel 邏輯，持續佔用 `active_cap` 名額。

**修復：**

| 變更 | 位置 | 說明 |
|------|------|------|
| 移除大跳轉時的 `_requested_thumbnail_paths.clear()` | `load_visible_images()` | 讓既有的 `to_cancel` diff 邏輯能正確清掉 gallery 自己的 `_active_tasks` |
| 新增 `_entry_prefetch_abandoned` flag | `__init__`、`begin_gallery_warmup()`（重置為 False）、大跳轉區塊（設為 True） | 大跳轉後放棄 entry-anchor 釘選，讓 `_gallery_prefetch_center_index()` 改用真實 viewport 中心 |

**驗證：** 用合成測試（直接呼叫 `_on_slider_pressed/_on_slider_released` 模擬拖曳跳轉到 75% 位置，而非 `stress_main_auto.py` 的平滑捲動，因為平滑 `setValue()` 不會觸發 `sliderPressed/Released`，測不到這個路徑）：

- 修復前：跳轉後 16–32 個舊位置 entries 卡在 `_active_tasks` 中 4.5 秒以上，新可視區完全沒有 `[GALLERY] load_visible_images scheduled=` log（`scheduled_tasks=0`，因為 active_cap 被舊 entries 佔滿，新 CURRENT 任務在排程迴圈中直接 `break`）。
- 修復後：跳轉後 50ms 內 `stale_overlap_with_pre_jump=0`，且同一個同步呼叫內就 `scheduled=18`（新可視區縮圖立即開始載入）。

---

## 8. 提高 gallery 並發以改善捲動流暢度（**2026-07-01，commit `16b6788` — 已 REVERT**）

> **已 revert（2026-07-01）：** 此並發調高與後續 byte-scan-first（原第 9 節）效能優化皆已回退，決定先聚焦穩定性。以下為當時的調查記錄，保留供日後重新評估。byte-scan-first 的關鍵發現（`_rawpy_global_lock` 序列化 RAW 縮圖解碼；byte-scan 對 Sony ARW / Nikon NEF 有效、其他格式 fallback 到 LibRaw；實測 ~3.4x）記錄於 session memory，未來若重做可參考。

**動機：** 使用者反映即使 GL crash 與 scroll-jump 修復後，`I:\Photos\London`（slow tier 外接碟）上的 gallery 捲動仍不夠流暢。

**發現：** fast-scroll 排程預算（`_gallery_scheduling_budgets(fast=True)`）在快速捲動時才是真正的瓶頸 —— 即使是 slow-tier 外接碟，`_apply_external_gallery_caps()` 用 `min()` 取 fast-scroll 預算與 slow-tier 預算兩者較小值；當時 slow-tier 的 `SLOW_GALLERY_MAX_TASKS=16` 比 fast-scroll 預算還低，變成隱性的真正上限。gallery 縮圖不受 `_raw_load_limit`（只限制 `full` stage 的重解）影響，所以真正的並發上限是 `_thread_pool.maxThreadCount()` 與 gallery 自己的 `active_cap`。

**變更：**

| 設定 | 舊值 | 新值 | 位置 |
|------|------|------|------|
| `RAWVIEWER_GALLERY_ACTIVE_CAP_FAST` | 24 | 32 | `gallery_view.py` |
| `RAWVIEWER_GALLERY_MAX_TASKS_FAST` | 16 | 24 | `gallery_view.py` |
| `RAWVIEWER_GALLERY_MAX_WIDGETS_FAST` | 12 | 16 | `gallery_view.py` |
| `RAWVIEWER_SLOW_GALLERY_ACTIVE_CAP` | 24 | 32 | `gallery_view.py`（提高以免蓋過上面的 fast-scroll 預算） |
| `RAWVIEWER_SLOW_GALLERY_MAX_TASKS` | 16 | 24 | `gallery_view.py` |
| `RAWVIEWER_SLOW_GALLERY_MAX_WIDGETS` | 32 | 40 | `gallery_view.py` |
| `RAWVIEWER_SLOW_VOLUME_MAX_WORKERS` | 8 | 12 | `image_load_manager.py`（實際 QThreadPool 執行緒數上限） |

**安全性：** 這些變更完全不涉及 `_raw_load_limit`（重解限流）或 GL 相關程式碼，所以不會重新引入本文件第 1 節的 Windows GL crash。已用 120s cold-cache stress 驗證（預設 Windows 設定、Mac-like 設定皆通過），並重新跑過第 7 節的 scroll-jump 合成測試確認兩個修復疊加無衝突（`active_count` 穩定維持在新的 32 上限）。

**跨平台影響：** 這些 cap 沒有 platform gate（`moderate_external_cap_enabled()` 只影響「快速」外接碟，不影響「confirmed-slow」層級的 cap，後者本來就是全平台適用）。因為不涉及 GL/RAW 解碼限流，對 macOS 沒有已知風險，且 macOS 從未出現過本調查的 GL crash，理論上安全邊際更大。

---

## 已 commit / 已 push 的變更檔案清單

`d8a35c4` fix + `10afee6` docs/scripts（rebase 前 hash：`7ea6af2`/`e6936cc`）：

- `src/main.py`
- `src/rawviewer_ui/gallery_view.py`
- `src/rawviewer_ui/gpu_image_view.py`
- `src/image_load_manager.py`
- `src/image_cache.py`
- `src/burst_grouping.py`
- `src/common_image_loader.py`
- `src/enhanced_raw_processor.py`
- `src/semantic_search.py`
- `src/unified_image_processor.py`
- `scripts/Launch/bat/run_debug.bat`
- `scripts/stress_main_auto.py` / `stress_main_worker.py` / `stress_gallery_*.py`（本地測試用，已納入版控）
- `pyrightconfig.json`, `typings/objc.pyi`（開發期型別檢查用）

`16b6788` perf（第 8 節，並發調高）：

- `src/rawviewer_ui/gallery_view.py`
- `src/image_load_manager.py`

未變更：`pixi.toml`（原計畫的 Mac-like env 覆寫已不需要，見待辦事項 2）。

---

## 9. Gallery 直向照片被 crop-fit 進橫向 tile（**2026-06-01**）

**症狀：** 部分**直向**照片在相簿 justified tile 中以**橫向外框**顯示（內容是正的，但被 center-crop 填滿橫向矩形）。常見於**剛進相簿**或**scrollbar 大跳**到新區域時。與 v2.4.1「整張圖 sideways」不同 — 這裡是 **tile 幾何 aspect 錯**，不是縮圖旋轉錯。

**根因（`gallery_view.py`）：**

1. **`_layout_aspect_for_path`** 曾把 global cache 解出的**未 orient** embedded JPEG 寬高當 ground truth，寫入 `_measured_raw_aspects`，後續 rebuild 一直用橫向 aspect（Sony ARW 等 RAW 常見）。
2. **`load_visible_images`** 從 `get_grid` / `get_thumbnail` 取圖時可**繞過** `on_thumbnail_ready` 的 orientation 修正，直接 `_fit_tile_pixmap` crop-fit。
3. **`_reconcile_tile_aspect`** 只更新 layout item 的 `aspect` 欄位；widget 的 `rect` 要等 `build_gallery` rebuild。rebuild 預設 debounce 500ms+，fast scroll 時再延後，跳轉後短時間內可見錯誤 frame。

**修復要點：**

| 機制 | 說明 |
|------|------|
| `_raw_aspect_from_pixmap` | 像素 aspect 與 EXIF display aspect 直/橫不一致時，以 EXIF 為 layout 依據 |
| `_orient_gallery_thumbnail_array` / `_orient_gallery_base_pixmap` | global cache → gallery base pixmap 一律套用 container orientation |
| `_global_cache_to_base_pixmap` | 統一 cache 取圖 + orient 路徑 |
| portrait ↔ landscape flip | `_reconcile_tile_aspect` 偵測到方向翻轉時縮短 rebuild debounce（idle 0ms / scroll 250ms） |
| `on_exif_ready` | EXIF 晚到時清除衝突的 `_measured_raw_aspects` 並 re-orient 已快取 base |

**驗證：** 含直向 ARW 的大資料夾 — 按 **G** 進相簿、scrollbar 跳到後段 — tile 外框應與直/橫一致。若個別檔仍錯，跑一次 `clear_cache.bat` 清除混有舊版未 orient cache 的項目。

**相關檔案：** `src/rawviewer_ui/gallery_view.py`（`_layout_aspect_for_path`、`_reconcile_tile_aspect`、`load_visible_images`、`warm_thumbnails_from_global_cache`）

---

## 修訂紀錄

| 日期 | 說明 |
|------|------|
| 2026-06-30 | 初版：彙整 Windows 外接碟 gallery crash、GL teardown、索引競爭、Mac-like env 與測試結果 |
| 2026-07-01 | 新增第 7 節（scroll-jump 修復）、第 8 節（並發調高）；commit `d8a35c4`/`10afee6`/`16b6788` 並 push 至 `origin/main`；更新待辦狀態與跨平台影響評估 |
| 2026-06-01 | 新增第 9 節：Gallery 直向/橫向 tile aspect 與 global cache orientation 時序 |
