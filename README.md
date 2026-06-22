# RAWviewer v2.5

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer Icon" width="256">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.5-blue" alt="Version">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total" alt="Downloads">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Donate-orange?logo=buy-me-a-coffee" alt="Buy Me a Coffee">
  </a>
</p>

**RAWviewer** is a fast photo viewer for **Windows and macOS**. Browse folders of RAW and JPEG files, check sharpness, cull rejects, and search your library — **on your computer, no cloud upload.**

Download: **[GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)**

---

## Using RAWviewer

Open a folder (menu, drag-and-drop, or double-click a photo). Scroll the **gallery**; click a thumbnail for full-screen view.

| Key | Action |
|-----|--------|
| **Space** / **Double-click** | Toggle fit-to-window / 100% zoom |
| **Pinch** / **Ctrl+scroll** | Zoom in or out |
| **←** / **→** | Previous / next image |
| **Scroll wheel** | Previous / next (single view, fit mode) |
| **↑** | Bookmark / unbookmark (bottom **star** in single view too) |
| **↓** | Move to Discard folder |
| **Delete** | Delete image(s) |
| **Esc** | Gallery: clear selection → exit bookmark filter · Single view: back to gallery |
| **Ctrl/Cmd+click** | Gallery: toggle selection |
| **Shift+click** | Gallery: select range (visible order) |
| **G** | Cycle composition guide |
| **H** | Show / hide histogram |
| **F** | Show / hide focus overlay (supported files) |

**Gallery bookmarks:** Click the outline **star** (nothing selected) to show bookmarked shots only; gold star = filter on. With photos selected, **↑** or the star toggles bookmarks on the selection.

**Gallery zoom slider:** Adjust the custom wedge-shaped slider next to the sort button to change thumbnail sizes in the gallery. Resizing is debounced (100ms) for smooth layout transitions and supports zoom-in only (minimum height 220px).

**Search:** gallery search icon — `camera:sony`, `iso<800`, … (**Full** also accepts `sunset on beach`). **Share:** bottom **Share / Open** button, or drag gallery / film-strip thumbnails out.

Search syntax → [Advanced reference](#advanced-reference).

---

## Lite vs Full

Both editions share the same viewer, culling tools, bookmarks, and metadata search. **Full** adds offline AI search and face filters.

| | Lite | Full |
|---|:--:|:--:|
| Gallery, film strip, zoom, histogram, bookmarks, culling | ✅ | ✅ |
| Metadata search (`camera:`, `iso:`, `date:`, …) | ✅ | ✅ |
| Plain-language search | — | ✅ |
| Face filters (`has:face`, …) | — | ✅ |

Pick **Lite** for a smaller install and browse-by-eye workflow. Pick **Full** to search with everyday words — still 100% offline.

---

## GPS map & geotagging — [LocateIt](https://github.com/markyip/LocateIt)

A GPS cluster map was prototyped inside RAWviewer, but it did not perform well enough on large folders to keep in the main app. **[LocateIt](https://github.com/markyip/LocateIt)** is a separate tool for that job: open an album, see where shots were taken on an interactive map, and **geotag photos missing GPS** by dragging them onto the map and saving back to the original files (JPEG and RAW).

---

## Download & install

### Windows

1. Download **`RAWviewer_Setup.exe`** from [Releases](https://github.com/markyip/RAWviewer/releases/latest).
2. Choose **Full (CUDA)**, **Full (DirectML)**, or **Lite** in the wizard. **Full** also downloads AI models (~600 MB).
3. Launch **`RAWviewer.exe`** or the Desktop shortcut (not the Setup file again).

Registers **Open with** for common photo formats. Uninstall: Settings → Apps, or **`uninstall.bat`** in `%LOCALAPPDATA%\RAWviewer`.

### macOS (13+)

1. Download **`RAWviewer-v2.5-macOS.zip`** (Full) or **`RAWviewer-v2.5-macOS-Lite.zip`** (Lite) from **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** and extract the zip.
2. Open **Terminal**, go to the extracted folder (`cd ` then drag the folder onto Terminal), and run:

```bash
bash install_macos_app.sh
```

3. Click **Install**, then **Open** in the dialogs. RAWviewer is copied to **Applications**.

**Full edition:** The first time you use gallery **Search**, RAWviewer may prompt to download the offline AI models from [Hugging Face](https://huggingface.co/) (~150 MB on macOS, one-time, needs internet). Without a Hugging Face account, that download may take longer. Click **Download** when prompted — progress appears in the search bar as `Downloading... N%`. Windows setup fetches the same models automatically during install.

Uninstall: **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** in the zip (keeps cache cleared; Trash alone does not).

### Requirements

Windows 10+ · macOS 13+ · 8 GB RAM (16 GB+ recommended for **Full** + large folders) · ~500 MB disk (**Lite**) or ~1.5 GB+ (**Full** with models)

To clear thumbnails only: **`scripts\Launch\bat\clear_cache.bat`** (Windows) · **`scripts/Launch/shell/clear_cache.sh`** (Mac)

---

## Supported formats

**RAW:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF, and other LibRaw types · **Standard:** JPEG, TIFF, HEIF

---

## Troubleshooting

### Windows

| Problem | What to do |
|---------|------------|
| SmartScreen warning | More info → Run anyway |
| Slow AI search (**Full**) | Prefer **DirectML** on most PCs; use **CUDA** only with NVIDIA + CUDA |
| Installer stuck on “Downloading models” (**Full**) | AI models (~600 MB) can take several minutes. Check firewall, VPN, or proxy if it fails — browsing still works; open gallery **Search** later to retry |
| Opened Setup again instead of the app | Launch **`RAWviewer.exe`** or the Desktop shortcut — not **`RAWviewer_Setup.exe`** |
| AI search missing after install (**Full**) | Open gallery **Search** → accept the download prompt |
| RAWviewer not in Open with | Re-run the installer (repair), or reinstall |
| Leftover cache after uninstall | Run **`uninstall.bat`** again, or delete `%USERPROFILE%\.rawviewer_cache` manually |
| Out of memory during AI indexing | See [Automatic memory tuning](#automatic-memory-tuning); use **Lite** on 8 GB PCs or set `RAWVIEWER_MEMORY_TIER_AUTO=0` and lower workers manually |
| App slow or exits after reopening last folder | **v2.4.1+** staggers full decode and prefetch on session restore. Still tight on 8 GB? Use **Lite**, or `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| Crash | Enable file logging with `RAWVIEWER_FILE_LOG=1`, then check the install folder |

### macOS

| Problem | What to do |
|---------|------------|
| macOS blocks the app (“damaged” / won’t open) | In the extracted folder, run `bash install_macos_app.sh` (see install steps above) |
| `bash: command not found` | Type `cd `, drag the extracted folder onto Terminal, press Return, then run the command again |
| Can’t read Desktop/Documents | System Settings → Privacy → **Full Disk Access** → add RAWviewer |
| Search says models missing (**Full**) | Open gallery search and click **Download** when prompted (needs internet once) |
| Download failed (SSL / certificate error) | Update to **v2.5** or newer (bundles certifi). On a corporate VPN or proxy, add your organization’s root certificate to **Keychain Access** and set it to **Always Trust** |
| Need to uninstall completely | Use **`uninstall_macos_app.sh`** or **`Uninstall RAWviewer.command`** from the release zip — not Trash alone |
| Uninstall scripts missing | Re-download the release zip from [Releases](https://github.com/markyip/RAWviewer/releases/latest); scripts are inside the extracted folder |
| macOS “out of memory” / heavy swap during indexing | See [Automatic memory tuning](#automatic-memory-tuning). On 8 GB Macs, prefer **Lite** or wait for indexing to finish before opening gallery on huge folders |
| Killed on relaunch (`Killed: 9` / exit 137 in Terminal) | **v2.4.1+** fixes most session-restore bursts. Try **Lite**, `RAWVIEWER_DISABLE_SESSION_RESTORE=1`, or `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| Gallery stutters on a huge folder | Update to **v2.5**. If it persists, run **`clear_cache.sh`** and reopen the folder |
| Thumbnails sideways or wrong way up (portrait shots) | Update to **v2.5**. Run **`clear_cache.sh`** once if old thumbnails were cached before the fix |

More detail: [`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## Advanced reference

*Optional — for search power users and troubleshooting.*

### Gallery search syntax

Separate words with spaces. Use `key:value` filters:

| Kind | Example |
|------|---------|
| Free text + filter | `jet takeoff camera:sony iso<800` *(Full: free text uses AI)* |
| Camera / lens | `camera:canon` · `lens:70-200` |
| ISO / year | `iso<=800` · `year>=2024` |
| Place | `city:tokyo` · `country:jp` |
| File name | `filename:_dsc` |
| Format | `format:raw` · `format:jpeg` · `format:cr3` |
| Date | `date:2024-05` |
| GPS / faces | `has:gps` · `has:face` · `no:face` *(face filters: Full only)* |

**Face vs semantic search:** `face`, `people`, `person`, etc. use stored face counts (`has:face`), not the neural search.

**Indexing:** On **Full** builds, semantic search and face counts run in the background on large folders (metadata + AI first, faces after). The gallery becomes searchable before face tagging finishes.

### Automatic memory tuning

On every launch, RAWviewer reads **installed system RAM** (not free RAM at that moment) and applies conservative defaults for load concurrency, preview cache, prefetch, and indexing — **only when you have not already set the same environment variables yourself**.

| Tier | Installed RAM | Typical Mac | What changes (summary) |
|------|---------------|-------------|-------------------------|
| **low** | &lt; 10 GB | 8 GB MacBook Air | Face scan off during indexing; fewer parallel workers; smaller preview cache; less idle prefetch |
| **medium** | 10–14 GB | 12 GB unified | Moderate limits on workers and cache |
| **balanced** | 14–20 GB | 16 GB | Light tuning (default on many laptops) |
| **high** | 20–28 GB | 24 GB | Slightly higher cache / worker caps |
| **ultra** | ≥ 28 GB | 32 GB+ studio machines | Stock app defaults (no overrides) |

**What you might notice**

- Startup log (dev / Terminal): `[PROFILE] memory tier=balanced (16.0 GB RAM)`
- A note file: `~/.rawviewer_cache/memory_tier.json` (tier, RAM, how many defaults were applied)
- **Lite** builds still use Lite profile defaults first; RAM tier only fills in gaps
- **Full** on an 8 GB Mac: semantic AI search can still run, but face indexing is disabled automatically to reduce memory pressure
- **Relaunch (v2.4.1+):** Session restore staggers full decode and prefetch so reopening the last folder is less likely to OOM; see release notes if fit view stays soft for a few seconds

**Disable auto-tuning** (use only your own env vars or scripts):

```bash
export RAWVIEWER_MEMORY_TIER_AUTO=0
```

**Force a specific override** (wins over auto-tuning — examples for low-RAM machines):

```bash
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_SEMANTIC_PREP_WORKERS=2
export RAWVIEWER_MEMORY_PREVIEW_MAX=1280
export RAWVIEWER_IDLE_DISPLAY_PREFETCH=0
```

Semantic batch/chunk size for AI indexing is **auto-tuned separately** on first index pass (Core ML on macOS, ONNX on Windows); results are cached under `~/.rawviewer_cache/semantic_batch_tuning.json`.

### MobileCLIP models (Full — AI search)

| Platform | When downloaded | Change variant (Windows) |
|----------|-----------------|--------------------------|
| **Windows Full** | During setup (~600 MB) | Set `RAWVIEWER_MOBILECLIP_VARIANT` to `s0`, `s2`, `b`, or `l14` |
| **macOS Full** | First gallery search (~150 MB) | Dev helper: `python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml` |

**Lite builds** do not use MobileCLIP models.

### Focus overlay (`F`) by brand

| Brand | Support |
|-------|---------|
| Canon CR2/CR3, Nikon NEF, Sony ARW, Olympus ORF, Panasonic RW2 | Yes (maker AF) |
| JPEG / TIFF / HEIF | Sometimes (EXIF SubjectArea) |
| Fujifilm RAF, Hasselblad 3FR, Pentax PEF, Samsung SRW, Sigma X3F | No |
| Typical Adobe DNG | Usually no |

Requires **pyexiv2** for maker-note AF on RAW.

### Environment variables

<details>
<summary><strong>Click to expand — dev / tuning flags</strong></summary>

| Variable | Effect |
|----------|--------|
| `RAWVIEWER_MEMORY_TIER_AUTO=1` | **Default.** Tune workers, cache, and prefetch from installed RAM at startup |
| `RAWVIEWER_MEMORY_TIER_AUTO=0` | Disable RAM-tier defaults; only explicit env vars apply |
| `RAWVIEWER_MOBILECLIP_VARIANT` | Windows ONNX model: `b` (default), `s0`, `s2`, `l14` |
| `RAWVIEWER_GPU_VIEW=1` | GPU single-image viewport (OpenGL zoom/pan; on by default in release builds) |
| `RAWVIEWER_GPU_VIEW=0` | Force legacy scroll-area single-image view |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | Same color pipeline for fit vs 100% zoom on RAW (default on) |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`, `pyexiv2`, or `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS: Qt share menu (recommended) |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS: try native share sheet first |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | Defer face scan until after semantic index (default) |
| `RAWVIEWER_SEMANTIC_PREP_WORKERS` | Parallel CPU workers before AI encode (RAM tier may set this) |
| `RAWVIEWER_SEMANTIC_BATCH_AUTO=1` | Auto-tune AI batch/chunk size on index (default) |
| `RAWVIEWER_SEMANTIC_BATCH_CANDIDATES` | Candidate batch sizes for auto-tune (default `8,16,32,64,128`) |
| `RAWVIEWER_PREVIEW_CACHE_ITEMS` | Cap in-memory preview LRU count |
| `RAWVIEWER_MEMORY_PREVIEW_MAX` | Max long edge for in-memory RAW/JPEG preview (pixels) |
| `RAWVIEWER_IDLE_DISPLAY_PREFETCH=0` | Disable idle neighbor prefetch in single view |
| `RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD=1` | **Default.** After relaunch, delay full decode and neighbor prefetch (see v2.4.1 release notes) |
| `RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS` | Milliseconds to wait after first paint before full decode on session restore (default `2500`) |
| `RAWVIEWER_DISABLE_SESSION_RESTORE=1` | Do not reopen the last folder/file on launch |

Full list and dev defaults: [`scripts/Launch/README.md`](scripts/Launch/README.md), [`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md).

</details>

### macOS version support

| Your Mac | Official `.zip` | Build from source |
|----------|-----------------|-------------------|
| macOS 13 Ventura (Intel) | ✅ | `build_macos_full.sh` or Pixi |
| macOS 13 Ventura (Apple Silicon) | ✅ | Use **`build_macos_full.sh`** (Pixi needs 14+) |
| macOS 14 Sonoma+ | ✅ | Pixi or `build_macos.sh` |
| macOS 12 Monterey or older | ❌ | ❌ |

### Upcoming (development branch)

Not in a release yet — tracked on a separate development branch.

**GPU-accelerated RAW decoding** — Early GPU decode works, but **correct color rendering** (matching the current LibRaw / embedded-JPEG pipeline) is still unresolved. We will only ship it if color accuracy and maintenance cost are acceptable.

This is separate from the GPU **viewport** (OpenGL zoom/pan on decoded pixels, on by default in release builds; set `RAWVIEWER_GPU_VIEW=0` to disable); the upcoming work targets **RAW decode** itself.

---

## For developers

Scripts and build matrix: [`scripts/Launch/README.md`](scripts/Launch/README.md)

### Quick start

```bash
pixi install
pixi run start          # full profile (default)
```

**Windows**

| Task | Script |
|------|--------|
| Run (full) | `scripts\Launch\bat\launch_dev_full.bat` |
| Run (lite) | `scripts\Launch\bat\launch_dev_lite.bat` |
| Build Full installers | `scripts\Launch\bat\build_windows_full.bat` (CUDA) or `build_windows_full.bat directml` |
| Build Lite installer | `scripts\Launch\bat\build_windows_lite.bat` |
| Build both Full backends | `scripts\Launch\bat\build_windows_all.bat` |

**macOS**

| Task | Script |
|------|--------|
| Run (full) | `./scripts/Launch/shell/launch_dev_full.sh` |
| Run (lite) | `./scripts/Launch/shell/launch_dev_lite.sh` |
| Build Full | `./scripts/Launch/shell/build_macos_full.sh` → `dist/RAWviewer.app` |
| Build Lite | `./scripts/Launch/shell/build_macos_lite.sh` → `dist/RAWviewer_Lite.app` |

Build outputs:

| Profile | Windows | macOS |
|---------|---------|-------|
| **Full / Unified** | `dist/RAWviewer_Setup.exe` (includes Full & Lite options) | `dist/RAWviewer-v2.5-macOS.zip` |
| **Lite** | (Select Lite option in `RAWviewer_Setup.exe`) | `dist/RAWviewer-v2.5-macOS-Lite.zip` |

Dependencies are in `pixi.toml`. Packaging scripts use a local `rawviewer_env/` venv when building release artifacts.

<details>
<summary><strong>Build from source (commands)</strong></summary>

**Windows**
```batch
scripts\Launch\bat\build_windows_full.bat
scripts\Launch\bat\build_windows_lite.bat
```

**macOS**
```bash
./scripts/Launch/shell/build_macos_full.sh
./scripts/Launch/shell/build_macos_lite.sh
# or: pixi install && pixi run python build.py --profile full
```

</details>

### Architecture (brief)

- **ImageLoadManager** — threaded load queue  
- **UnifiedImageProcessor** — RAW/JPEG/TIFF via one path  
- **Cache** — memory-first; optional disk cache via env; **RAM-tier defaults** at startup (`rawviewer_profile.py`)
- **Semantic index** — SQLite + local embeddings (Core ML on macOS, ONNX on Windows; Full builds only)

---

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Pull requests welcome on [GitHub](https://github.com/markyip/RAWviewer).

## Support

1. Check [Troubleshooting](#troubleshooting) above  
2. Search [existing issues](https://github.com/markyip/RAWviewer/issues)  
3. Open a new issue with OS version, steps, and logs if possible  

## ☕ Buy Me a Coffee

If RAWviewer helps your workflow, you can [buy me a coffee](https://www.buymeacoffee.com/markyip) ☕

---

**Enjoy your photos.** 📸

---

# RAWviewer 說明文件 (繁體中文)

<p align="center">
  <img src="icons/appicon.ico" alt="RAWviewer 圖示" width="256">
</p>

**RAWviewer** 是一款適用於 **Windows 和 macOS** 的快速相片檢視器。瀏覽包含 RAW 和 JPEG 檔案的資料夾、檢查清晰度、篩選/刪除相片、搜尋您的影像庫 —— **完全在您的電腦上本機運行，無需上傳雲端。**

下載連結：**[GitHub 最新發布版本](https://github.com/markyip/RAWviewer/releases/latest)**

---

## 使用 RAWviewer

開啟資料夾（透過選單、拖放檔案，或按兩下相片）。滾動**藝廊**；按一下縮圖進入全螢幕檢視模式。

| 按鍵 | 動作 |
|-----|--------|
| **空白鍵** / **按兩下** | 切換 適合視窗 / 100% 縮放 |
| **雙指捏合** / **Ctrl+滾輪** | 放大或縮小 |
| **←** / **→** | 上一張 / 下一張影像 |
| **滾輪** | 上一張 / 下一張（單張檢視，適合視窗模式下） |
| **↑** | 加入 / 取消書籤（在單張檢視的右下角也有**星號**按鈕） |
| **↓** | 移動到 捨棄 (Discard) 資料夾 |
| **Delete** | 刪除影像 |
| **Esc** | 藝廊：清除選取 → 退出書籤篩選 · 單張檢視：返回藝廊 |
| **Ctrl/Cmd+按一下** | 藝廊：切換選取狀態 |
| **Shift+按一下** | 藝廊：選取範圍（依據可見順序） |
| **G** | 切換構圖輔助線 |
| **H** | 顯示 / 隱藏直方圖 |
| **F** | 顯示 / 隱藏對焦覆蓋區域（僅支援的檔案） |

**藝廊書籤：** 在未選取任何相片時按一下**空心星號**，將只顯示有書籤的相片（星號會變為**金色**）。再次按一下或按 **Esc** 返回完整網格。選取縮圖時，按 **↑** 或星號可對選取的相片切換書籤狀態。

**藝廊縮放滑桿：** 調整排序按鈕旁邊的自訂楔形滑桿，可改變藝廊中的縮圖大小。調整大小時有 100 毫秒的防抖延遲以確保版面配置過渡平滑，且僅支援放大（最小高度為 220px）。

**搜尋：** 藝廊搜尋圖示 —— `camera:sony`, `iso<800`, …（**Full** 完整版還支援 `sunset on beach` 這樣的自然語言搜尋）。
**分享：** 底部的 **Share / Open (分享 / 開啟)** 按鈕，或直接將藝廊/底片縮圖拖曳出來。

搜尋語法請參考 → [進階說明](#進階說明-繁體中文)。

---

## 輕量版 (Lite) vs 完整版 (Full)

這兩個版本都擁有相同的檢視器、篩選工具、書籤和中繼資料 (Metadata) 搜尋功能。**完整版 (Full)** 額外增加了離線 AI 搜尋和人臉篩選器。

| 功能 | 輕量版 (Lite) | 完整版 (Full) |
|---|:--:|:--:|
| 藝廊、底片縮圖、縮放、直方圖、書籤、相片篩選 | ✅ | ✅ |
| 中繼資料搜尋 (`camera:`, `iso:`, `date:`, …) | ✅ | ✅ |
| 自然語言 (AI) 搜尋 | — | ✅ |
| 人臉篩選器 (`has:face`, …) | — | ✅ |

如果您想要更小的安裝體積和純手動/肉眼篩選流程，請選擇 **Lite 輕量版**。如果您希望使用日常語言搜尋相片，請選擇 **Full 完整版**（依然是 100% 離線運行）。

---

## GPS 地圖與地理標記 —— [LocateIt](https://github.com/markyip/LocateIt)

我們曾在 RAWviewer 內部開發了 GPS 叢集地圖原型，但在大型資料夾上的效能表現不夠理想，因此未將其納入主程式。**[LocateIt](https://github.com/markyip/LocateIt)** 是一個專門處理此任務的獨立工具：開啟相簿、在互動式地圖上查看拍攝地點，並透過將相片拖曳到地圖上並儲存回原始檔案（JPEG 和 RAW）來**為缺少 GPS 的相片標記地理位置**。

---

## 下載與安裝

### Windows

1. 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 下載 **`RAWviewer_Setup.exe`**。
2. 在安裝精靈中選擇 **Full (CUDA)**、**Full (DirectML)** 或 **Lite**。**Full** 版會同時下載 AI 模型（約 600 MB）。
3. 啟動 **`RAWviewer.exe`** 或桌面捷徑（不要再次開啟安裝程式）。

註冊常見相片格式的「開啟檔案」關聯。解除安裝：Windows 設定 → 應用程式，或執行 `%LOCALAPPDATA%\RAWviewer` 資料夾中的 **`uninstall.bat`**。

### macOS (13+)

1. 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 下載 **`RAWviewer-v2.5-macOS.zip`** (Full) 或 **`RAWviewer-v2.5-macOS-Lite.zip`** (Lite) 並解壓縮。
2. 開啟 **終端機 (Terminal)**，移至解壓縮後的資料夾（輸入 `cd ` 然後將資料夾拖曳到終端機視窗中），並執行：

```bash
bash install_macos_app.sh
```

3. 在彈出的對話框中按一下 **Install**，然後按 **Open**。RAWviewer 將被複製到 **應用程式 (Applications)** 資料夾中。

**Full 完整版：** 第一次使用藝廊**搜尋**時，RAWviewer 可能會提示從 [Hugging Face](https://huggingface.co/) 下載離線 AI 模型（在 macOS 上約 150 MB，一次性下載，需要網路連線）。若沒有 Hugging Face 帳號，下載可能需要較長時間。在提示時按一下 **Download** —— 進度會顯示在搜尋列中（如 `Downloading... N%`）。Windows 安裝程式則會在安裝過程中自動下載這些模型。

解除安裝：執行解壓縮資料夾中的 **`uninstall_macos_app.sh`** 或 **`Uninstall RAWviewer.command`**（這會清除快取檔案；單純將 App 拖到垃圾桶無法清除快取）。

### 系統需求

Windows 10+ · macOS 13+ · 8 GB RAM（建議 16 GB+ 用於 **Full** 版 + 大型資料夾）· 約 500 MB 磁碟空間（**Lite**）或 1.5 GB+（帶模型的 **Full**）

如需僅清除快取縮圖：執行 **`scripts\Launch\bat\clear_cache.bat`** (Windows) · **`scripts/Launch/shell/clear_cache.sh`** (Mac)

---

## 支援格式

**RAW:** CR2, CR3, NEF, ARW, DNG, ORF, RW2, RAF，以及其他 LibRaw 支援的類型 · **標準格式:** JPEG, TIFF, HEIF

---

## 疑難排解

### Windows

| 問題 | 解決方法 |
|---------|------------|
| SmartScreen 警告 | 按一下「其他資訊」 → 「仍要執行」 |
| AI 搜尋速度慢 (**Full**) | 大多數電腦建議選擇 **DirectML**；僅在使用 NVIDIA 顯示卡 + CUDA 時使用 **CUDA** |
| 安裝程式卡在「正在下載模型」 (**Full**) | AI 模型（約 600 MB）下載可能需要幾分鐘。如果下載失敗，請檢查防火牆、VPN 或代理伺服器 —— 下載失敗仍可正常瀏覽相片；之後可以開啟藝廊**搜尋**來重試 |
| 開啟了安裝程式而不是應用程式本身 | 請啟動 **`RAWviewer.exe`** 或桌面捷徑 —— 而不是 **`RAWviewer_Setup.exe`** |
| 安裝後找不到 AI 搜尋功能 (**Full**) | 開啟藝廊**搜尋** → 接受下載提示 |
| RAWviewer 沒有出現在「開啟檔案」選單中 | 重新執行安裝程式（選擇修復），或重新安裝 |
| 解除安裝後殘留快取 | 再次執行 **`uninstall.bat`**，或手動刪除 `%USERPROFILE%\.rawviewer_cache` |
| AI 索引時記憶體不足 | 請參閱 [自動記憶體調整](#自動記憶體調整-繁體中文)；在 8 GB RAM 的電腦上使用 **Lite** 輕量版，或設置 `RAWVIEWER_MEMORY_TIER_AUTO=0` 並手動降低背景工作執行緒數量 |
| 重新開啟上次的資料夾時 App 變慢或退出 | **v2.4.1+** 版本在還原工作階段時會交錯執行完整解碼 and 預載。在 8 GB 電載上若仍感吃力，請改用 **Lite**，或設定 `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| 應用程式當機 | 設定 `RAWVIEWER_FILE_LOG=1` 啟用檔案記錄，然後查看安裝目錄下的日誌檔案 |

### macOS

| 問題 | 解決方法 |
|---------|------------|
| macOS 阻擋 App（顯示「已損毀」/ 無法開啟） | 在解壓縮後的資料夾中，執行 `bash install_macos_app.sh`（參見上述安裝步驟） |
| `bash: command not found` | 輸入 `cd `，將解壓縮後的資料夾拖曳到終端機中，按 Enter，然後重新輸入指令 |
| 無法讀取桌面 (Desktop) 或文件 (Documents) | 系統設定 → 隱私權與安全性 → **全權數位存取 (Full Disk Access)** → 新增並啟用 RAWviewer |
| 搜尋顯示缺少模型 (**Full**) | 開啟藝廊搜尋，並在出現提示時按一下 **Download**（僅需一次網路連線） |
| 下載失敗（SSL / 憑證錯誤） | 請更新至 **v2.5** 或更新版本（已隨附 certifi）。若使用公司 VPN 或代理，請將組織的根憑證新增至** keychain 存取**並設定為「永遠信任」，然後重試 |
| 需要完全解除安裝 | 請使用 release zip 中的 **`uninstall_macos_app.sh`** 或 **`Uninstall RAWviewer.command`** —— 不要只把 App 拖到垃圾桶 |
| 解除安裝腳本遺失 | 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 重新下載發布的 zip，腳本就在解壓縮後的資料夾內 |
| macOS 「記憶體不足」 / 索引時大量使用 Swap 虛擬記憶體 | 參閱 [自動記憶體調整](#自動記憶體調整-繁體中文)。在 8 GB Mac 上，建議使用 **Lite** 版，或等索引完成後再開啟大型資料夾的藝廊 |
| 重新啟動時 App 被終止（終端機中顯示 `Killed: 9` / exit 137） | **v2.4.1+** 修復了大部分工作階段還原時的記憶體突發問題。請嘗試使用 **Lite**，或設定 `RAWVIEWER_DISABLE_SESSION_RESTORE=1`、`RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| 在極大資料夾中藝廊滾動卡頓 | 更新到 **v2.5**。若問題持續，執行 **`clear_cache.sh`** 並重新開啟該資料夾 |
| 直式相片縮圖顯示為橫向或顛倒 | 更新到 **v2.5**。若有在修復前快取的舊縮圖，請執行一次 **`clear_cache.sh`** 清除舊快取 |

更多細節：[`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## 進階說明 (繁體中文)

*選用資訊 —— 供搜尋功能進階用戶與疑難排解使用。*

### 藝廊搜尋語法

使用空格分隔多個關鍵字。可以使用 `key:value` 篩選器：

| 篩選類型 | 範例 |
|------|---------|
| 自由文字 + 篩選 | `jet takeoff camera:sony iso<800` *(Full版：自由文字會使用 AI 語意)* |
| 相機 / 鏡頭 | `camera:canon` · `lens:70-200` |
| ISO / 年份 | `iso<=800` · `year>=2024` |
| 地點 | `city:tokyo` · `country:jp` |
| 檔案名稱 | `filename:_dsc` |
| 格式 | `format:raw` · `format:jpeg` · `format:cr3` |
| 日期 | `date:2024-05` |
| GPS / 人臉 | `has:gps` · `has:face` · `no:face` *(人臉篩選器僅限 Full 完整版)* |

**人臉 vs 語意搜尋：** `face`、`people`、`person` 等關鍵字使用的是儲存的人臉計數 (`has:face`)，而非神經網路語意搜尋。

**索引：** 在 **Full** 完整版中，語意搜尋和人臉計數會在背景對大型資料夾進行索引（先中繼資料 + AI，後人臉）。藝廊在人臉標記完成前即可進行搜尋。

### 自動記憶體調整

每次啟動時，RAWviewer 都會讀取**系統安裝的實體記憶體大小**（而非當時的可用記憶體），並在**您尚未自行設定對應環境變數**時，自動對載入並行度、預覽快取、預載半徑和索引線程數套用保守的預設值。

| 等級 | 實體記憶體 | 常見 Mac 機型 | 調整細節 (摘要) |
|------|---------------|-------------|-------------------------|
| **low** | &lt; 10 GB | 8 GB MacBook Air | 索引時關閉人臉掃描；減少並行工作執行緒；縮小預覽快取；減少閒置預載 |
| **medium** | 10–14 GB | 12 GB 統一記憶體 | 適度限制執行緒數量與快取 |
| **balanced** | 14–20 GB | 16 GB | 輕度調整（多數筆電的預設值） |
| **high** | 20–28 GB | 24 GB | 稍微放寬快取和並行工作執行緒上限 |
| **ultra** | ≥ 28 GB | 32 GB+ Studio 等級機種 | 套用 App 原始預設值（不進行覆蓋） |

**您可能會注意到的現象**

- 啟動日誌（開發模式 / 終端機）：`[PROFILE] memory tier=balanced (16.0 GB RAM)`
- 記錄檔案：`~/.rawviewer_cache/memory_tier.json`（包含等級、RAM、已套用的預設值數量）
- **Lite** 輕量版仍會先使用 Lite 設定檔預設值；記憶體等級僅補充空白設定
- **Full** 在 8 GB Mac 上：語意 AI 搜尋仍可運行，但人臉索引會自動停用以減輕記憶體壓力
- **還原工作階段 (v2.4.1+)：** 還原上次的資料夾時，會延遲交錯載入完整解碼與鄰近預載，防止 OOM。低 RAM 機型若在此時畫面暫時呈現模糊預覽，屬於正常 staggered 載入現象

**停用自動記憶體調整**（僅使用您自己的環境變數或腳本設定）：

```bash
export RAWVIEWER_MEMORY_TIER_AUTO=0
```

**強制特定的覆蓋值**（優先於自動記憶體調整 —— 以下為低 RAM 設備的推薦設定）：

```bash
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_SEMANTIC_PREP_WORKERS=2
export RAWVIEWER_MEMORY_PREVIEW_MAX=1280
export RAWVIEWER_IDLE_DISPLAY_PREFETCH=0
```

AI 搜尋索引的 MobileCLIP 批次大小會在首次索引時**獨立進行自動調整**（macOS 上為 Core ML，Windows 上為 ONNX），調整結果會快取於 `~/.rawviewer_cache/semantic_batch_tuning.json` 中。

### MobileCLIP 模型 (Full —— AI 搜尋)

| 平台 | 何時下載 | 變更模型版本 (Windows) |
|----------|-----------------|--------------------------|
| **Windows Full** | 安裝過程中 (~600 MB) | 設定 `RAWVIEWER_MOBILECLIP_VARIANT` 為 `s0`、`s2`、`b` 或 `l14` |
| **macOS Full** | 首次藝廊搜尋時 (~150 MB) | 開發輔助：`python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml` |

**Lite 輕量版**不使用 MobileCLIP 模型。

### 對焦框覆蓋 (`F`) 品牌支援度

| 品牌 | 支援情況 |
|-------|---------|
| Canon CR2/CR3, Nikon NEF, Sony ARW, Olympus ORF, Panasonic RW2 | 支援 (製造商 AF 資訊) |
| JPEG / TIFF / HEIF | 部分支援 (EXIF SubjectArea) |
| Fujifilm RAF, Hasselblad 3FR, Pentax PEF, Samsung SRW, Sigma X3F | 不支援 |
| 一般 Adobe DNG | 通常不支援 |

對焦框需要 **pyexiv2** 讀取 RAW 檔中的製造商註記 (MakerNote) AF 數據。

### 開發者說明

指令稿與建置矩陣：[`scripts/Launch/README.md`](scripts/Launch/README.md)

### 快速開始

```bash
pixi install
pixi run start          # 完整版設定檔 (預設)
```

建置輸出：

- **Windows 統一安裝程式:** `dist/RAWviewer_Setup.exe` (包含 Full 及 Lite 安裝選項)
- **macOS Full 完整版:** `dist/RAWviewer-v2.5-macOS.zip`
- **macOS Lite 輕量版:** `dist/RAWviewer-v2.5-macOS-Lite.zip`

### 授權條款

MIT —— 詳見 [LICENSE](LICENSE)。

**盡情享受您的照片吧。** 📸

