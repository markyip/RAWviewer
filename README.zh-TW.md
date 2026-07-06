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

**語言：** [English](README.md) · **繁體中文**

**RAWviewer** 是一款適用於 **Windows 與 macOS** 的快速相片檢視器。瀏覽 RAW 與 JPEG 資料夾、檢查銳利度、篩選淘汰、搜尋圖庫——**全部在本機完成，無需上傳雲端。**

下載：**[GitHub Releases](https://github.com/markyip/RAWviewer/releases/latest)**

---

## 使用 RAWviewer

開啟資料夾（選單、拖放，或雙擊相片）。在**圖庫**中捲動；點擊縮圖進入全螢幕檢視。

在**大型資料夾**（數千張相片）中，待 EXIF 拍攝時間排序完成後才會顯示 **Gallery** 按鈕，進入圖庫時縮圖即為拍攝順序。若中繼資料已快取，排序可瞬間完成。

在**圖庫檢視**中，拖曳底部列的**大小滑桿**可調整縮圖尺寸。列會以齊行網格（滿寬）重新排版；拖曳時捲動位置會錨定在左上角可見相片。

| 按鍵 | 動作 |
|-----|--------|
| **空白鍵** / **雙擊** | 切換「符合視窗」/ 100% 縮放 |
| **捏合** / **Ctrl+捲動** | 放大 / 縮小 |
| **←** / **→** | 上一張 / 下一張 |
| **滑鼠滾輪** | 上一張 / 下一張（單張檢視、符合模式） |
| **↑** | 加入 / 取消書籤（單張檢視底部**星號**亦可） |
| **↓** | 移至 Discard 資料夾 |
| **Delete** | 刪除影像 |
| **Esc** | 圖庫：清除選取 → 離開書籤篩選 · 單張：返回圖庫 |
| **Ctrl/Cmd+點擊** | 圖庫：切換選取 |
| **Shift+點擊** | 圖庫：範圍選取（可見順序） |
| **C** | 切換比較模式（需選取多張） |
| **G** | 切換構圖輔助線 |
| **H** | 顯示 / 隱藏直方圖（啟動時預設隱藏） |
| **J** | 切換高光 / 陰影裁切疊圖（RAW 單張檢視） |
| **P** | 切換 RAW 復原預覽——半解析度陰影 / 高光復原（RAW/DNG，僅本工作階段；僅符合模式） |
| **F** | 顯示 / 隱藏對焦疊圖（支援的檔案） |
| **M** | 顯示 / 隱藏 GPS 地圖疊圖（單張檢視、含 GPS 相片；啟動時預設隱藏） |

**比較模式快捷鍵：**
* **← / →** — 上一張 / 下一張候選
* **↑** — 將候選（右側）提升為已選（左側）
* **↓** — 拒絕候選並移至 Discard（Shift+↓ 拒絕選取範圍）
* **Delete** — 將候選刪至資源回收筒 / 垃圾桶（Shift+Delete 刪除選取範圍）
* **空白鍵** — 同步切換兩側縮放（開啟 **F** 對焦框後：左右各自縮放至該張對焦點）
* **F** — 顯示 / 隱藏兩側對焦疊圖（比較模式）
* **J** — 兩側曝光裁切疊圖
* **G** — 兩側構圖格線
* **C** / **Esc** — 離開比較模式

**圖庫書籤：** 點擊空心**星號**（未選取時）僅顯示已加書籤相片；金色星號 = 篩選開啟。已選相片時，**↑** 或星號可切換書籤。

**搜尋：** 圖庫搜尋圖示——`camera:sony`、`iso<800` 等（**完整版**另支援 `sunset on beach` 等自然語句）。**分享：** 底部 **Share / Open**，或將圖庫 / 底片列縮圖拖出。

搜尋語法 → [進階參考](#進階參考)。

---

## Lite 與 Full

兩個版本共用相同的檢視器、篩選工具、書籤與中繼資料搜尋。**Full（完整版）** 另含離線 AI 搜尋與人臉篩選。

| | Lite（精簡版） | Full（完整版） |
|---|:--:|:--:|
| 圖庫、底片列、縮放、直方圖、書籤、篩選 | ✅ | ✅ |
| 中繼資料搜尋（`camera:`、`iso:`、`date:` 等） | ✅ | ✅ |
| 地點搜尋（`city:`、`country:` 等，需 GPS） | ✅ | ✅ |
| 自然語句搜尋 | — | ✅ |
| 人臉篩選（`has:face` 等） | — | ✅ |

選 **Lite** 可獲較小安裝包，以目視瀏覽為主。選 **Full** 可用日常用語搜尋——仍為 **100% 離線**。

---

## GPS 地圖疊圖與地理標記

在**單張檢視**按 **M** 可切換互動式地圖卡片。卡片會立即開啟並顯示 **Loading map…**，待圖磚載入（無 GPS 的相片不會彈出）。地圖上的**座標徽章**顯示經緯度；點擊可在瀏覽器開啟 **Google Maps**。

內建離線資料庫（`cities500.csv.gz`、`landmarks.csv.gz`，逾 10 萬筆地點）在**背景索引**時將 GPS 解析為城市、地區、國家，供**圖庫搜尋**使用——例如 `city:tokyo`、`country:jp`，無需網路。

若需要整本相簿的**叢集地圖**或為**缺少 GPS 的相片標記位置**，請參閱 **[LocateIt](https://github.com/markyip/LocateIt)**：開啟資料夾、在地圖上檢視拍攝位置、拖放指定座標，並寫回 JPEG 或 RAW。

---

## 下載與安裝

### Windows

1. 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 下載 **`RAWviewer_Setup.exe`**。
2. 在安裝精靈選擇 **Full (CUDA)**、**Full (DirectML)** 或 **Lite**。**Full** 會另下載 AI 模型（約 600 MB）。
3. 啟動 **`RAWviewer.exe`** 或桌面捷徑（勿再次執行 Setup）。

> **v2.5 新功能：** 圖庫縮放滑桿、捲動錨定、GPS 地圖疊圖（**M**）、HDR 靜態與 **RAW（高品質工作流程）** 的 **macOS HDR/EDR**、RAW 的 **P** 復原預覽 / **J** 裁切疊圖、GIF/WebP 動畫播放、啟動時隱藏直方圖 / 地圖、切換相簿時更可靠的圖庫版面、進入圖庫或跳捲時的直向 / 橫向縮圖比例修正、更乾淨的背景索引取消、連拍分組、以及 **C** 鍵切換的並排比較檢視。

會註冊常見格式的**開啟方式**。解除安裝：設定 → 應用程式，或 `%LOCALAPPDATA%\RAWviewer` 內的 **`uninstall.bat`**。

### macOS（13+）

1. 從 **[Releases](https://github.com/markyip/RAWviewer/releases/latest)** 下載 **`RAWviewer-v2.5-macOS.zip`**（Full）或 **`RAWviewer-v2.5-macOS-Lite.zip`**（Lite）並解壓。
2. 開啟**終端機**，進入解壓資料夾（`cd ` 後將資料夾拖入終端機），執行：

```bash
bash install_macos_app.sh
```

3. 在對話框點擊 **Install**，再點 **Open**。RAWviewer 會複製到**應用程式**資料夾。

**完整版：** 首次使用圖庫**搜尋**時，可能提示從 [Hugging Face](https://huggingface.co/) 下載離線 AI 模型（macOS 約 150 MB，一次性，需網路）。無 Hugging Face 帳號時可能較慢。出現提示時點 **Download**——進度顯示於搜尋列 `Downloading... N%`。Windows 安裝程式會自動下載相同模型。

解除安裝：**`uninstall_macos_app.sh`** 或壓縮檔內的 **`Uninstall RAWviewer.command`**（會清除快取；僅丟到垃圾桶不會）。

### 系統需求

Windows 10+ · macOS 13+ · 8 GB RAM（**Full** + 大型資料夾建議 16 GB+）· 約 500 MB 磁碟（**Lite**）或 1.5 GB+（**Full** 含模型）

僅清除縮圖快取：**`scripts\Launch\bat\clear_cache.bat`**（Windows）· **`scripts/Launch/shell/clear_cache.sh`**（Mac）

---

## 支援格式

**RAW：** CR2、CR3、NEF、ARW、DNG、ORF、RW2、RAF 及其他 LibRaw 類型 · **標準：** JPEG、TIFF、HEIF、**GIF**（動畫）、**WebP**（動畫）

在 **macOS** 上，HDR **HEIC / HEIF / AVIF** 與 16 位元 HDR **TIFF** 在啟用 EDR 時可於單張檢視以延伸動態範圍顯示（預設開啟）。使用 **RAW（高品質）** 工作流程時，**RAW / DNG** 亦走 EDR 路徑（線性 16 位元解碼）。**內嵌 JPEG 工作流程** 使用相機內嵌預覽（SDR）。其他平台將 HDR 靜態 tone-map 至 SDR。

**工作流程切換**（單張檢視）：在 **內嵌 JPEG（快速）** 與 **RAW（高品質）** 間切換。EDR 僅在 RAW 工作流程下套用。

**復原預覽（P）：** 半解析度陰影 / 高光復原，用於判斷極端對比——僅本工作階段，不取代全解析度檢視。

---

## macOS EDR 速查

| 顯示內容 | 意義 |
|----------|------|
| 中繼資料中的 `EDR · RAW` | RAW 以延伸動態範圍顯示 |
| `EDR · HDR` | HDR 靜態（HEIC/TIFF）走 EDR |
| `EDR ready · embedded JPEG workflow` | EDR 視埠已啟用；顯示內嵌 JPEG（SDR） |
| 啟動：`RAW EDR active` | RAW EDR 已啟用且選擇 RAW 工作流程 |

| 變數 | 預設 | 效果 |
|------|------|------|
| `RAWVIEWER_RAW_EDR` | `1` | macOS：**RAW 工作流程** 時 RAW 走 EDR；設 `0` 關閉。App 內底部工具列 **EDR** 按鈕可切換；每次切換**進入** RAW 工作流程時會重設為關閉，避免進入 RAW 模式後瀏覽默默變慢 |
| `RAWVIEWER_DISABLE_EDR` | 關 | 關閉所有 macOS EDR（HDR 靜態 + RAW） |

---

## 疑難排解

### 所有平台

| 問題 | 處理方式 |
|------|----------|
| GPS 地圖不顯示 | 單張檢視按 **M**；僅含 GPS 的相片會顯示地圖 |
| HDR HEIC/TIFF 偏平或偏暗 | **Windows：** HDR 靜態圖預設 tone-map 至 SDR。**macOS：** 需 EDR 螢幕（GPU 單張視埠預設開啟）；`RAWVIEWER_DISABLE_EDR=1` 強制 SDR |
| **P** / **J** 無效 | **P**/**J** 僅 RAW/DNG 單張；**P** 僅符合模式半解析度預覽。**P** 復原另需 scipy + rawpy——失敗時查日誌 |

### Windows

| 問題 | 處理方式 |
|------|----------|
| SmartScreen 警告 | 詳細資訊 → 仍要執行 |
| AI 搜尋慢（**Full**） | 多數 PC 建議 **DirectML**；僅 NVIDIA + CUDA 時用 **CUDA** |
| 安裝卡在「Downloading models」（**Full**） | AI 模型約 600 MB，可能需數分鐘。失敗時檢查防火牆、VPN 或 Proxy——瀏覽仍可用；稍後開圖庫 **Search** 重試 |
| 又開啟 Setup 而非程式 | 啟動 **`RAWviewer.exe`** 或桌面捷徑——不是 **`RAWviewer_Setup.exe`** |
| 安裝後無 AI 搜尋（**Full**） | 開圖庫 **Search** → 接受下載提示 |
| 開啟方式沒有 RAWviewer | 重新執行安裝（修復）或重裝 |
| 解除安裝後殘留快取 | 再執行 **`uninstall.bat`**，或手動刪除 `%USERPROFILE%\.rawviewer_cache` |
| AI 索引時記憶體不足 | 見[自動記憶體調校](#自動記憶體調校)；8 GB 請用 **Lite** 或設 `RAWVIEWER_MEMORY_TIER_AUTO=0` 並手動降低 worker |
| 重開上次資料夾後變慢或退出 | 8 GB 機器請用 **Lite** 或設 `RAWVIEWER_DISABLE_SESSION_RESTORE=1` |
| RAW 總是顯示 demosaic 而非內嵌 JPEG | 切換至 **內嵌 JPEG 工作流程**；RAW EDR 會用 LibRaw 重解碼並覆蓋內嵌預覽 |
| 當機 | 設 `RAWVIEWER_FILE_LOG=1` 啟用檔案日誌，再查安裝目錄 |

### macOS

| 問題 | 處理方式 |
|------|----------|
| 系統阻擋（「損毀」/ 無法開啟） | 在解壓資料夾執行 `bash install_macos_app.sh`（見上方安裝步驟） |
| `bash: command not found` | 輸入 `cd `，將解壓資料夾拖入終端機，按 Return，再執行指令 |
| 無法讀取桌面 / 文件 | 系統設定 → 隱私權 → **完整磁碟存取** → 加入 RAWviewer |
| 搜尋提示缺少模型（**Full**） | 開圖庫搜尋，出現提示時點 **Download**（需網路一次） |
| 下載失敗（SSL / 憑證） | 企業 VPN / Proxy 請將根憑證加入**鑰匙圈**並設為**永遠信任** |
| 需完整解除安裝 | 使用壓縮檔內 **`uninstall_macos_app.sh`** 或 **`Uninstall RAWviewer.command`**——勿只丟垃圾桶 |
| 找不到解除安裝腳本 | 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 重新下載；腳本在解壓資料夾內 |
| 索引時記憶體不足 / 大量 swap | 見[自動記憶體調校](#自動記憶體調校)。8 GB Mac 建議 **Lite** 或待索引完成再開大型圖庫 |
| 重開被終止（`Killed: 9` / exit 137） | 可試 **Lite**、`RAWVIEWER_DISABLE_SESSION_RESTORE=1` 或 `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| 大型資料夾圖庫仍卡頓 | 執行 **`clear_cache.sh`** 後重開資料夾 |
| 大型資料夾首次開 Gallery 較慢 | 正常——等待 EXIF 拍攝時間排序以確保順序；中繼資料已快取時可瞬間完成 |
| 想要內嵌 JPEG 而非 RAW EDR | 使用 **內嵌 JPEG 工作流程**，或 `RAWVIEWER_RAW_EDR=0` |

更多細節：[`scripts/Launch/README.md`](scripts/Launch/README.md)

---

## 進階參考

*選讀——供進階搜尋與疑難排解。*

> **縮圖快取說明：** 為加快圖庫載入，RAWviewer 會在本機建立縮圖快取。**絕不會上傳或分享**——僅存於你的電腦。快取檔在 **30 天**未使用後自動刪除。

### 圖庫搜尋語法

以空格分隔關鍵字。使用 `key:value` 篩選：

| 類型 | 範例 |
|------|------|
| 自由文字 + 篩選 | `jet takeoff camera:sony iso<800` *（Full：自由文字走 AI）* |
| 相機 / 鏡頭 | `camera:canon` · `lens:70-200` |
| ISO / 年份 | `iso<=800` · `year>=2024` |
| 地點 | `city:tokyo` · `country:jp` |
| 檔名 | `filename:_dsc` |
| 格式 | `format:raw` · `format:jpeg` · `format:cr3` |
| 日期 | `date:2024-05` |
| GPS / 人臉 | `has:gps` · `has:face` · `no:face` *（人臉篩選：僅 Full）* |

**人臉與語意搜尋：** `face`、`people`、`person` 等使用已儲存人臉計數（`has:face`），非神經網路語意搜尋。

**索引：** **Full** 版在大型資料夾背景執行語意搜尋與人臉計數（先中繼資料 + AI，再人臉）。**搜尋欄在索引完成前為唯讀**（**Lite：** 中繼資料；**Full：** 中繼資料、嵌入向量，以及啟用時的人臉掃描）。**切換資料夾**時會取消上一資料夾的索引與預載（**v2.5.0**）。

### 自動記憶體調校

每次啟動時，RAWviewer 讀取**已安裝的系統 RAM**（非當下可用記憶體），並套用載入並行度、預覽快取、預載與索引的保守預設——**僅在你尚未自行設定相同環境變數時**。

| 層級 | 已安裝 RAM | 典型 Mac | 摘要 |
|------|------------|----------|------|
| **low** | &lt; 10 GB | 8 GB MacBook Air | 索引時關閉人臉掃描；較少平行 worker；較小預覽快取；較少閒置預載 |
| **medium** | 10–14 GB | 12 GB 統一記憶體 | 適度限制 worker 與快取 |
| **balanced** | 14–20 GB | 16 GB | 輕度調校（多數筆電預設） |
| **high** | 20–28 GB | 24 GB | 略提高快取 / worker 上限 |
| **ultra** | ≥ 28 GB | 32 GB+ 工作站 | 應用程式預設（不覆寫） |

**你可能會看到**

- 啟動日誌（開發 / 終端機）：`[PROFILE] memory tier=balanced (16.0 GB RAM)`
- 備註檔：`~/.rawviewer_cache/memory_tier.json`（層級、RAM、套用了多少預設）
- **Lite** 仍先套用 Lite 設定檔；RAM 層級僅填補未設定的項目
- 8 GB Mac 上的 **Full**：語意 AI 仍可運行，但會自動關閉人臉索引以減輕記憶體壓力
- **重新啟動（v2.4.1+）：** 工作階段還原會錯開全解析度解碼與預載，降低 OOM；若符合視圖數秒內仍偏軟屬正常

**關閉自動調校**（僅使用自己的環境變數或腳本）：

```bash
export RAWVIEWER_MEMORY_TIER_AUTO=0
```

**強制覆寫**（優先於自動調校——低記憶體範例）：

```bash
export RAWVIEWER_ENABLE_FACE_SCAN=0
export RAWVIEWER_SEMANTIC_PREP_WORKERS=2
export RAWVIEWER_MEMORY_PREVIEW_MAX=1280
export RAWVIEWER_IDLE_DISPLAY_PREFETCH=0
```

AI 索引的語意 batch / chunk 大小會在**首次索引**時**另行自動調校**（macOS Core ML、Windows ONNX）；結果快取於 `~/.rawviewer_cache/semantic_batch_tuning.json`。

### MobileCLIP 模型（Full——AI 搜尋）

| 平台 | 下載時機 | 變更型號（Windows） |
|------|----------|---------------------|
| **Windows Full** | 安裝時（約 600 MB） | 設 `RAWVIEWER_MOBILECLIP_VARIANT` 為 `s0`、`s2`、`b` 或 `l14` |
| **macOS Full** | 首次圖庫搜尋（約 150 MB） | 開發：`python scripts/download_mobileclip_coreml.py --out-dir models/mobileclip2_coreml` |

**Lite** 不使用 MobileCLIP 模型。

### 對焦疊圖（`F`）依品牌

| 品牌 | 支援 |
|------|------|
| Canon CR2/CR3、Nikon NEF、Sony ARW、Olympus ORF、Panasonic RW2 | 是（製造商 AF） |
| JPEG / TIFF / HEIF | 有時（EXIF SubjectArea） |
| Fujifilm RAF、Hasselblad 3FR、Pentax PEF、Samsung SRW、Sigma X3F | 否 |
| 常見 Adobe DNG | 通常否 |

RAW 製造商 AF 需 **pyexiv2**。

### 環境變數

<details>
<summary><strong>展開——開發 / 調校旗標</strong></summary>

| 變數 | 效果 |
|------|------|
| `RAWVIEWER_MEMORY_TIER_AUTO=1` | **預設。** 依已安裝 RAM 調校 worker、快取、預載 |
| `RAWVIEWER_MEMORY_TIER_AUTO=0` | 關閉 RAM 層級預設；僅套用明確設定的環境變數 |
| `RAWVIEWER_MOBILECLIP_VARIANT` | Windows ONNX 模型：`b`（預設）、`s0`、`s2`、`l14` |
| `RAWVIEWER_GPU_VIEW=1` | GPU 單張視埠（OpenGL 縮放 / 平移；正式版預設開啟） |
| `RAWVIEWER_GPU_VIEW=0` | 強制傳統捲動區單張視圖 |
| `RAWVIEWER_DISABLE_EDR=1` | macOS：關閉 EDR 視埠與 HDR/RAW 16 位元顯示；改用 SDR tone-map |
| `RAWVIEWER_RAW_EDR=1` | **預設。** macOS：選 **RAW（高品質）** 時 RAW 走 EDR；`0` 硬性關閉。App 內：底部工具列 **EDR** 按鈕可由使用者切換，但每次切換進入 RAW 工作流程都會重設為關閉。EDR 解碼採閒置延遲：快速瀏覽時立即顯示 SDR 快速緩衝，只有在該張影像暫留後才升級為 EDR，因此瀏覽速度不受影響 |
| `RAWVIEWER_LIBRAW_CONSISTENT_PREVIEW=1` | RAW 符合與 100% 縮放同色票流程（預設開啟） |
| `RAWVIEWER_FAST_RAW_DECODE=0` | 停用快速 RAW 解碼路徑（cv2 像素運算，與 LibRaw 色彩完全一致，half/full tier 共用 unpack；預設開啟，感光元件不支援時自動回退 rawpy） |
| `RAWVIEWER_EXIF_BACKEND=auto` | `auto`、`pyexiv2` 或 `exifread` |
| `RAWVIEWER_SHARE_MENU=1` | macOS：Qt 分享選單（建議） |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | macOS：優先嘗試原生分享表 |
| `RAWVIEWER_INDEX_DEFER_FACE_SCAN=1` | 語意索引完成後再做人臉掃描（預設） |
| `RAWVIEWER_SEMANTIC_PREP_WORKERS` | AI 編碼前平行 CPU worker（RAM 層級可能設定） |
| `RAWVIEWER_SEMANTIC_BATCH_AUTO=1` | 索引時自動調校 AI batch/chunk（預設） |
| `RAWVIEWER_SEMANTIC_BATCH_CANDIDATES` | 自動調校候選 batch（預設 `8,16,32,64,128`） |
| `RAWVIEWER_PREVIEW_CACHE_ITEMS` | 記憶體中預覽 LRU 上限 |
| `RAWVIEWER_FULL_IMAGE_CACHE_ITEMS` | 全解析度緩衝 LRU 上限（預設 8，上限 32；調高可讓縮放後 A/B 來回切換瞬間顯示，每格約 100–200 MB） |
| `RAWVIEWER_MEMORY_PREVIEW_MAX` | 記憶體中 RAW/JPEG 預覽長邊上限（像素） |
| `RAWVIEWER_IDLE_DISPLAY_PREFETCH=0` | 關閉單張檢視閒置鄰張預載 |
| `RAWVIEWER_SESSION_RESTORE_DEFER_PRELOAD=1` | **預設。** 重新啟動後延遲全解析度解碼與鄰張預載（見 v2.4.1 發行說明） |
| `RAWVIEWER_SESSION_RESTORE_FULL_DECODE_DELAY_MS` | 首次繪製後至全解析度解碼的延遲毫秒（預設 `2500`） |
| `RAWVIEWER_DISABLE_SESSION_RESTORE=1` | 啟動時不還原上次資料夾 / 檔案 |

完整清單與開發預設：[`scripts/Launch/README.md`](scripts/Launch/README.md)、[`docs/macos-sharing-v21-v22.md`](docs/macos-sharing-v21-v22.md)。

</details>

### macOS 版本支援

| 你的 Mac | 官方 `.zip` | 從原始碼建置 |
|----------|-------------|--------------|
| macOS 13 Ventura（Intel） | ✅ | `build_macos_full.sh` 或 Pixi |
| macOS 13 Ventura（Apple Silicon） | ✅ | 請用 **`build_macos_full.sh`**（Pixi 需 14+） |
| macOS 14 Sonoma+ | ✅ | Pixi 或 `build_macos.sh` |
| macOS 12 Monterey 或更舊 | ❌ | ❌ |

### 開發中（development 分支）

尚未發行——在獨立 development 分支追蹤。

**Windows HDR / EDR**——v2.5 已為 HDR 靜態與 RAW（高品質工作流程）加入 macOS EDR。Windows 目前將 HDR HEIC/TIFF 與 RAW tone-map 至 SDR。未來 Windows 路徑將利用 HDR 螢幕（10 位元 / scRGB 或 Qt QRhi HDR10），使高光可使用延伸動態範圍，類似 macOS。

**連拍分組**——在圖庫自動分組短時間內連拍序列。開啟連拍群組一併檢視；**比較**模式可並排檢視候選以挑選最佳張。

**快速 RAW 解碼**——已解決（2026-07）：曾阻擋 GPU 解碼的色彩一致性問題已解決，結論也改變了做法。fit-view（半尺寸）與感光元件解析度（全尺寸）兩個 tier 現在都改用 LibRaw unpack + SIMD 像素運算（cv2/numpy），與舊 rawpy 管線**色彩完全一致**（Sony ARW + 21 個 Canon CR3 黃金樣本驗證於 ±1 8-bit LSB 內——見 `scripts/fast_raw_decode_parity_gate.py`）。兩個 tier 共用同一次 LibRaw unpack：fit-view 解碼會暫存已 unpack 的感光元件資料，之後延遲觸發或縮放觸發的全尺寸解碼直接沿用，不必重新開檔、重新 unpack——省下每張使用者暫留或縮放檢視影像的重複 unpack（100–900ms）。全尺寸解碼比舊 rawpy LINEAR 路徑快 1.4–1.7 倍，demosaic 品質也更佳。GPU offload 本身經實測後否決：在 Apple Silicon 統一記憶體架構下，cv2 的 CPU demosaic 已達到記憶體頻寬上限（OpenCL 未顯示增益），GPU 後端只會增加相依性重量而無實質好處。可用 `RAWVIEWER_FAST_RAW_DECODE=0` 停用（兩個 tier 皆回退 rawpy）。

**多執行緒 LibRaw（macOS 開發環境）**——PyPI 的 rawpy wheel 在 macOS/Linux 上內附單執行緒 LibRaw（Windows wheel 已內建 OpenMP）。`scripts/build_libraw_openmp.sh` 會以 OpenMP 重新編譯 LibRaw 並替換進 Pixi 環境（self-contained dylib、輸出 byte-identical、併發解碼下已驗證具決定性），CR3/RAF/pana8 unpack 及 AHD/DHT/X-Trans demosaic 約快 1.5–2 倍。非執行 RAWviewer 的必要條件，僅為本機開發環境最佳化，`pixi install` 重建環境後需重新執行。可用 `scripts/check_libraw_parallelism.py <raw 檔案>` 驗證核心使用率。

此與 GPU **視埠**（解碼後像素的 OpenGL 縮放 / 平移，正式版預設開啟；`RAWVIEWER_GPU_VIEW=0` 關閉）不同。

---

## 開發者

腳本與建置矩陣：[`scripts/Launch/README.md`](scripts/Launch/README.md)

### 快速開始

```bash
pixi install
pixi run start          # full profile（預設）
```

**Windows**

| 任務 | 腳本 |
|------|------|
| 執行（full） | `scripts\Launch\bat\launch_dev_full.bat` |
| 執行（lite） | `scripts\Launch\bat\launch_dev_lite.bat` |
| 建置 Full 安裝程式 | `scripts\Launch\bat\build_windows_full.bat`（CUDA）或 `build_windows_full.bat directml` |
| 建置 Lite 安裝程式 | `scripts\Launch\bat\build_windows_lite.bat` |
| 建置兩種 Full 後端 | `scripts\Launch\bat\build_windows_all.bat` |

**macOS**

| 任務 | 腳本 |
|------|------|
| 執行（full） | `./scripts/Launch/shell/launch_dev_full.sh` |
| 執行（lite） | `./scripts/Launch/shell/launch_dev_lite.sh` |
| 建置 Full | `./scripts/Launch/shell/build_macos_full.sh` → `dist/RAWviewer.app` |
| 建置 Lite | `./scripts/Launch/shell/build_macos_lite.sh` → `dist/RAWviewer_Lite.app` |

建置產物：

| 設定檔 | Windows | macOS |
|--------|---------|-------|
| **Full / Unified** | `dist/RAWviewer_Setup.exe`（含 Full 與 Lite 選項） | `dist/RAWviewer-v2.5-macOS.zip` |
| **Lite** | （在 `RAWviewer_Setup.exe` 選 Lite） | `dist/RAWviewer-v2.5-macOS-Lite.zip` |

相依套件見 `pixi.toml`。封裝腳本建置正式版時使用本機 `rawviewer_env/` 虛擬環境。

<details>
<summary><strong>從原始碼建置（指令）</strong></summary>

**Windows**
```batch
scripts\Launch\bat\build_windows_full.bat
scripts\Launch\bat\build_windows_lite.bat
```

**macOS**
```bash
./scripts/Launch/shell/build_macos_full.sh
./scripts/Launch/shell/build_macos_lite.sh
# 或：pixi install && pixi run python build.py --profile full
```

</details>

### 架構（簡述）

- **ImageLoadManager** — 執行緒載入佇列；切換資料夾會取消進行中任務（**v2.5.0**）
- **UnifiedImageProcessor** — RAW/JPEG/TIFF 統一路徑
- **Cache** — 記憶體優先；可選磁碟快取；啟動時 **RAM 層級預設**（`rawviewer_profile.py`）
- **Semantic index** — SQLite + 本機嵌入（macOS Core ML、Windows ONNX；僅 Full）；切換資料夾範圍時中止背景 pass（**v2.5.0**）
- **Gallery（JustifiedGallery）** — 齊行網格與縮放滑桿（重排 + 左上捲動錨點）；版面快取綁定資料夾世代；EXIF 排序後以拍攝時間順序開圖庫；鎖定齊行幾何前以容器 EXIF  reconcile 解碼縮圖比例（**v2.5.0**）
- **HDR / EDR（macOS）** — GPU 視埠 EDR 層 + 16 位元 HDR 靜態解碼；RAW 工作流程啟用時經線性 LibRaw 走 RAW EDR（**v2.5.0**）
- **RAW 復原預覽** — **P** 鍵，半解析度線性解碼 + 區域 tone 復原（`raw_tone_recovery.py`；**v2.5.0**）
- **裁切疊圖** — 單張檢視 **J** 鍵（`exposure_clipping.py`；**v2.5.0**）

---

## 授權

MIT——見 [LICENSE](LICENSE)。

## 貢獻

歡迎在 [GitHub](https://github.com/markyip/RAWviewer) 提交 Pull Request。

## 支援

1. 先查上方[疑難排解](#疑難排解)  
2. 搜尋[既有議題](https://github.com/markyip/RAWviewer/issues)  
3. 若仍無解，請開新議題並附上作業系統版本、步驟與日誌  

## ☕ Buy Me a Coffee

若 RAWviewer 對你的工作流程有幫助，歡迎 [請我喝杯咖啡](https://www.buymeacoffee.com/markyip) ☕

---

**享受你的相片。** 📸
