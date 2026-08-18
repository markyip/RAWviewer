<p align="center">
  <img src="icons/appicon-master.png" alt="RAWviewer" width="256">
</p>

<h1 align="center">RAWviewer</h1>

<p align="center"><strong>瀏覽、篩選、評分數千張 RAW 相片——快速、私密、完全在你的電腦上完成。</strong></p>

<p align="center">
  <img src="docs/images/hero-gallery.jpg" alt="RAWviewer 圖庫檢視——RAW 資料夾以齊行網格顯示" width="900">
</p>

<p align="center">
  <a href="https://github.com/markyip/RAWviewer/releases/latest"><strong>⬇&nbsp; 下載 Windows 版</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/markyip/RAWviewer/releases/latest"><strong>⬇&nbsp; 下載 macOS 版</strong></a>
</p>

<p align="center">免費 · 開放原始碼（MIT）· 離線運作——相片絕不離開你的電腦</p>

<p align="center">
  <img src="https://img.shields.io/github/v/release/markyip/RAWviewer?style=flat-square&labelColor=2b2925&color=96897a&label=版本" alt="最新版本">
  <img src="https://img.shields.io/github/downloads/markyip/RAWviewer/total?style=flat-square&labelColor=2b2925&color=d9691e&label=下載次數" alt="總下載次數">
  <img src="https://img.shields.io/badge/授權-MIT-96897a?style=flat-square&labelColor=2b2925" alt="授權：MIT">
  <a href="https://www.buymeacoffee.com/markyip">
    <img src="https://img.shields.io/badge/請我喝咖啡-☕-d9a441?style=flat-square&labelColor=2b2925" alt="Buy Me a Coffee">
  </a>
</p>

**Language / 語言：** [English](README.md) · [繁體中文](README.zh-TW.md)

---

## 它能做什麼

- **以拍攝的速度瀏覽。** 開啟一個 RAW 資料夾，用方向鍵全螢幕翻閱——不必匯入、沒有目錄庫、無需等待。
- **用手指篩片，不用滑鼠。** **1–5** 給留下的照片打星、**↓** 移至淘汰資料夾、**0** 清除星級、**C** 讓相似的幾張並排比較並同步縮放。
- **找到任何一張相片。** 在圖庫搜尋輸入地點（`tokyo`）、相機（`sony`）或年份（`2024`）。完整版還聽得懂日常描述，例如 `sunset on beach`——全部離線完成。
- **顯影而不動原始檔。** 按 **E** 開啟 Adjust 面板：曝光、白平衡、裁切、Dodge & Burn、修復筆刷、LUT，以及局部遮罩——可手塗或由 App 找出（Plus）。所有編輯寫入 XMP sidecar，RAW 檔永遠保持原樣。

<p align="center">
  <img src="docs/images/single-view.jpg" alt="全螢幕單張檢視，含底片列與星級評分" width="900">
  <br><em>全螢幕瀏覽：底片列與星級評分</em>
</p>

<p align="center">
  <img src="docs/images/adjust.jpg" alt="Adjust 面板與前後對比分割檢視——直方圖、色調滑桿與曲線在相片旁" width="900">
  <br><em>Adjust 面板（前後對比分割）：非破壞顯影，寫入 XMP</em>
</p>

## 3.0 新功能

- **明顯快過 2.5** — Windows 冷啟動套件（每次清快取）：圖庫就緒約 **2.7–2.9×**（**8.6 秒 → 約 3 秒**）；RAW 全解析度導航中位數約 **1.6–1.7×**（**0.95 秒 → 約 0.6 秒**）。Standard 與 Plus 共用 3.0 管線，篩片關鍵路徑皆勝過 2.5 — 見 [`RELEASE_NOTES.zh-TW.md`](RELEASE_NOTES.zh-TW.md)
- **Adjust 面板** — 非破壞顯影：色調、白平衡預設、裁切、Dodge & Burn、修復筆刷、暈影／去霧、Creative LUT 與可儲存的預設；匯出 JPEG／WebP／16 位元 TIFF
- **星級評分** — 用數字鍵 1–5 評分；圖庫可依最低星級篩選
- **Nikon 高效率（HE/HE\*）檔案**現可瀏覽與篩片
- **更精簡的 Standard 版**，適合在意安裝大小的機器

完整變更說明：[`RELEASE_NOTES.zh-TW.md`](RELEASE_NOTES.zh-TW.md)

## 未發行的 3.1.0（相對 3.0.2 的穩定度＋效能）

GitHub Releases 目前仍是 **3.0.x**。**3.1.0 是穩定度與效能補丁**，不是功能車次，**尚未打 tag**。測試：[`testplan/TEST_PLAN_3.1.md`](testplan/TEST_PLAN_3.1.md)。說明：[`RELEASE_NOTES.zh-TW.md`](RELEASE_NOTES.zh-TW.md)。

3.1.0 **相對 3.0.2 宣稱**的內容：

- **RAM 分級 Adjust 上限** — 較慢／≤16 GB 機器用較小工作底圖 settle；匯出仍全解析度。沒有記憶體分級選單。
- **圖庫捲動與載入** — 佇列對準可視範圍、消失的縮圖會回來、觸控板次像素、切資料夾不清縮圖、取消載入只收尾一次。
- **快取** — 完整影像改位元組預算；開啟 Adjust 未編輯離開不清所有層；sidecar 解析共用。
- **匯出可靠度** — 分帶接縫、分段路徑靜默跳過 AI 降噪、OpenMP／BLAS 不占滿核心。
- **當機／執行緒／取消** — 工作執行緒不建 `QPixmap`、配額重複釋放、`id()` 登錄表、CUDA-GL 執行緒、預載錯誤風暴。
- **HE/HE\* NEF** — Adjust 走內嵌 JPEG 而不拒檔；無法解碼的 RAW 保留內嵌預覽（不是灰格）。

*（未提交、針對此補丁 — 尚未進 git：）* 先前只活在 RAM 的 EXIF 欄位寫進磁碟 blob；捲軸日期氣泡不再每 tick 閃爍／解碼。

遮罩、成對、MCP、Share to AI，以及工作區其餘功能車次**不是** 3.1.0 出貨——見[展望](#展望)（**3.2+**）。生成式編輯仍暫緩。

---

## 開始使用

### Windows

1. 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 下載 **`RAWviewer_Setup.exe`**。
2. 在安裝精靈選擇 **Plus**（含相片描述搜尋與 AI 降噪，另下載約 700 MB 模型）或 **Standard**（較小）。不確定？見下方[選 Standard 還是 Plus？](#選-standard-還是-plus)。
3. **從舊版升級？** 可選擇勾選安裝精靈中的 **Clear existing cache**。可啟用較快的搜尋／索引預設，**不會**刪除相片或 XMP。
4. 從桌面捷徑啟動 **RAWviewer**。

常見相片格式會自動加入**開啟方式**。隨時可從「設定 → 應用程式」解除安裝。之後若要清快取，可執行安裝目錄內與 `RAWviewer.exe` 同層的 **`clear_cache.bat`**。

### macOS（13 或更新）

1. 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 下載 macOS zip（**Plus** 或 **Standard**），雙擊解壓。
2. 開啟**終端機**（Spotlight 搜尋「終端機」），輸入 `cd `（含一個空格），把解壓出的資料夾拖進終端機視窗，按 Return。
3. 執行安裝腳本，依序點擊兩個對話框（**Install**、**Open**）：

```bash
bash install_macos_app.sh
```

4. **升級？** 建議清一次快取，讓搜尋／索引使用較新、較快的預設（不會刪除相片或 XMP）：
   - 雙擊 **Clear Cache.command**（若被阻擋：右鍵 → 打開 → 打開），或
   - 在終端機執行：`bash clear_macos_cache.sh`

RAWviewer 會複製到「應用程式」資料夾。這個一次性的終端機步驟是讓 macOS 信任這個 App——之後就和一般 App 一樣直接開啟。要移除時請使用 zip 內的 **`Uninstall RAWviewer.command`**（只丟垃圾桶會殘留快取檔）。完整步驟見壓縮檔內的 **Start Here.txt**。

**Plus：** 首次使用圖庫**搜尋**時，會提示一次性下載 AI 模型（約 150 MB，僅此一次需要網路）。點 **Download** 後，搜尋即完全離線。

### 上手的前十分鐘

開啟任何相片資料夾（拖進視窗，或雙擊一張相片），然後：

| 按鍵 | 動作 |
|-----|--------|
| **← / →** | 上一張 / 下一張 |
| **空白鍵** 或雙擊 | 符合視窗 ↔ 100% 縮放 |
| **0–5** | 星級評分（0 清除；底部星星亦可） |
| **6–9** | 彩色標籤（紅／黃／綠／藍；**Shift+6** 紫；再按一次清除） |
| **K** | 編輯關鍵字（XMP `dc:subject`） |
| **↓** | 移至淘汰（Discard）資料夾 |
| **C** | 並排比較選取的相片 |
| **E** | 開啟 Adjust（顯影）面板 |
| **Esc** | 返回圖庫 |

篩片流程就是這樣。以下內容都是選讀。

<details>
<summary><strong>所有瀏覽與圖庫快捷鍵</strong></summary>

開啟資料夾（選單、拖放，或雙擊相片）。在**圖庫**中捲動；點擊縮圖進入全螢幕檢視。**大型資料夾**（數千張）會待拍攝時間排序完成後才顯示 **Gallery** 按鈕，確保縮圖為拍攝順序；中繼資料已快取時排序瞬間完成。圖庫底部的**大小滑桿**可調整縮圖尺寸。

| 按鍵 | 動作 |
|-----|--------|
| **空白鍵** / **雙擊** | 切換「符合視窗」/ 100% 縮放 |
| **捏合** / **Ctrl+捲動** | 放大 / 縮小 |
| **←** / **→** | 上一張 / 下一張 |
| **滑鼠滾輪** | 上一張 / 下一張（單張檢視、符合模式） |
| **0–5** | 星級評分（**0** 清除；單張檢視底部星星亦可） |
| **6–9** | 彩色標籤（紅／黃／綠／藍；**Shift+6** 紫；再按一次清除） |
| **K** | 編輯關鍵字 |
| **↓** | 移至 Discard 資料夾 |
| **Delete** | 刪除影像 |
| **Esc** | 圖庫：清除選取 → 離開篩選 · 單張：返回圖庫 |
| **Ctrl/Cmd+點擊** | 圖庫：切換選取 |
| **Shift+點擊** | 圖庫：範圍選取（可見順序） |
| **↑** | 圖庫：向上捲動 · 比較模式：提升候選 |
| **C** | 切換比較模式（需選取多張） |
| **E** | 顯示 / 隱藏 **Adjust** 面板 |
| **G** | 切換構圖輔助線 |
| **H** | 顯示 / 隱藏直方圖 |
| **J** | 切換高光 / 陰影裁切疊圖 |
| **P** | 切換 RAW 復原預覽——LibRaw Reconstruct + 局部修飾（RAW/DNG，僅本工作階段；僅符合模式）。不是 Adjust 的 Highlights／Shadows 滑桿。 |
| **F** | 顯示 / 隱藏對焦疊圖（支援的檔案） |
| **M** | 顯示 / 隱藏 GPS 地圖疊圖（單張檢視、含 GPS 相片） |

**圖庫星級篩選：** 用底部**星星**只顯示 ≥ N★ 的相片。有彩色標籤時，縮圖上會出現色點。

**RAW+JPEG 成對**（預設開啟）將同檔名 RAW 與 JPEG 疊成一格 `R+J`。**連拍分組**為選用且**預設關閉**（至少 **3** 張；封面由你指定）。成對與連拍是兩個開關——兩檔 RAW+JPEG 不會被當成連拍。

**單張檢視工作流程切換：** 在**內嵌 JPEG（快速）**與 **RAW（高品質）**間切換。**復原預覽（P）**使用 Reconstruct + 局部修飾來判斷極端對比，**不是** Highlights／Shadows 滑桿（滑桿在場景線性空間壓縮／提升亮度，無法把完全過曝的 (1,1,1) 高光色相救回來）。

**分享：** 底部 **Share / Open** 按鈕，或將圖庫 / 底片列縮圖拖出。Mac 選單含 AirDrop（原檔）與 WhatsApp／Signal（RAW 先轉 JPEG）。**檔案 → Share to AI agent** 會把預覽 JPEG 與 MCP 工單交給 agent（風格說明不是生成式提示詞）。見 [`docs/macos-sharing.md`](docs/macos-sharing.md) 與 [`docs/MCP.md`](docs/MCP.md)。

</details>

<details>
<summary><strong>比較模式快捷鍵</strong></summary>

* **← / →** — 上一張 / 下一張候選
* **↑** — 將候選（右側）提升為已選（左側）
* **↓** — 拒絕候選並移至 Discard（Shift+↓ 拒絕選取範圍）
* **Delete** — 將候選刪至資源回收筒 / 垃圾桶（Shift+Delete 刪除選取範圍）
* **空白鍵** — 同步切換兩側縮放（開啟 **F** 對焦框後：左右各自縮放至該張對焦點）
* **F** — 顯示 / 隱藏兩側對焦疊圖
* **J** — 兩側曝光裁切疊圖
* **G** — 兩側構圖格線
* **C** / **Esc** — 離開比較模式

</details>

<details>
<summary><strong>Adjust（顯影）快捷鍵</strong></summary>

Adjust 面板開啟時（**E**）：

| 按鍵 | 動作 |
|-----|--------|
| **E** / **Esc** | 關閉 Adjust（返回瀏覽模式） |
| **D** / **B** / **X** / **H** | 按住並掃過即可塗 **Dodge**／**Burn**／**Eraser**／**Heal**。放開即收起工具 |
| **P** | **遮罩分頁：** 塗入選取遮罩（**X** 為擦除）。瀏覽（關閉 Adjust）時 **P** 仍是 Reconstruct *預覽*——Auto 不會把它烤進去 |
| **O** / **M** | 遮罩疊色（遮罩分頁一次只顯示一個；**I** 反轉） |
| **雙指捲動** | 筆刷啟用時調整 **Brush Size**（**Ctrl+捲動**仍為縮放） |
| **←** / **→** | 微調目前焦點滑桿（無焦點時則上一張／下一張） |
| **Ctrl/Cmd+Z** | 復原上一步編輯 |
| **空白鍵**／**雙擊** | 符合視窗／100% 縮放 |
| **J**／**G**／**F** | 裁切疊圖／構圖輔助線／對焦疊圖（與瀏覽相同） |

說明：**Effect Strength** 僅套用於 Dodge/Burn；Heal 使用 **Size**／**Flow**，inpaint 一律滿強度。瀏覽專用鍵（**M**、**P**、直方圖 **H**）在 Adjust 開啟時不適用——此時 **H** 改為啟用 Heal。預設編輯結果在 Adjust 面板內呈現、瀏覽顯示原始像素；所有編輯寫入 RAW 旁的 XMP sidecar。

</details>

<details>
<summary><strong>搜尋——如何找到相片</strong></summary>

開啟圖庫搜尋，以空格分隔關鍵字。**多數情況不需要特殊語法**——地點、相機、鏡頭、檔名，或 `2024`／`2024-05` 這類日期直接輸入即可。需要指定欄位或比較數值時再用 `key:value`。

| 類型 | 範例 |
|------|------|
| 地點 | `tokyo` · `Taipei` · `hong kong` · `city:tokyo` · `country:jp` |
| 相機 / 鏡頭 | `sony` · `canon` · `70-200` · `camera:canon` · `lens:70-200` |
| 檔名 | `_dsc` · `IMG_1234` · `filename:_dsc` |
| 日期 | `2024` · `2024-05` · `date:2024-05` |
| ISO / 年份（比較） | `iso<=800` · `iso under 800` · `year>=2024` |
| 格式 | `format:raw` · `format:jpeg` · `format:cr3` |
| GPS / 人臉 | `has:gps` · `has:face` · `no:face` *（人臉篩選：僅 Plus）* |
| 自由文字 + 篩選 | `jet takeoff camera:sony iso<800` *（Plus：未匹配的自由文字走 AI）* |

地名搜尋可離線使用：內建逾 10 萬筆城市與地標資料庫，會在背景索引時把相片的 GPS 解析成可搜尋的地名。

搜尋欄在目前資料夾索引完成前為唯讀。完整版的語意與人臉索引在背景執行；切換資料夾時會取消舊資料夾的索引。

</details>

---

## 選 Standard 還是 Plus？

兩個版本都有完整的檢視器：圖庫、篩片、比較、星級、書籤、GPS 地圖、中繼資料搜尋，以及 Adjust 顯影面板與匯出（JPEG／WebP／16 位元 TIFF）。

| | Standard | Plus (DirectML) | Plus (CUDA) |
|---|:--:|:--:|:--:|
| 上述全部——瀏覽、篩片、評分、比較、顯影、匯出 | ✅ | ✅ | ✅ |
| 用描述找相片（`sunset on beach`） | — | ✅ | ✅ |
| 找出有人物的相片（`has:face`） | — | ✅ | ✅ |
| AI 降噪（Adjust 面板匯出） | 色度／亮度雜訊抑制 | **SCUNet AI 降噪** | **SCUNet AI 降噪** |
| RAW demosaic | CPU Fast RAW | CPU Fast RAW | **CuPy** GPU（NVIDIA） |
| 安裝大小（約） | 約 0.8–1 GB | 約 1.5–2 GB + 約 700 MB 模型 | 約 1.5–2 GB + 約 700 MB 模型 |
| 建議記憶體 | 8 GB | 16 GB | 16 GB + NVIDIA |

**DirectML 負責 AI 搜尋與降噪，不做 demosaic。** Windows Plus 的相片描述搜尋與 AI 降噪在兩個 Plus 版本都走 ONNX Runtime＋DirectML。Plus（DirectML）與 Standard 的 RAW 解碼是 **CPU Fast RAW**；只有 Plus（CUDA）才加 **CuPy** GPU demosaic。

**AI 降噪**透過 ONNX Runtime 執行真實影像降噪模型（SCUNet）——不需要 PyTorch。隨 Plus 安裝的其他 AI 模型一併下載（約 100 MB），在 Adjust 面板開啟降噪時，匯出／完整品質算圖會自動套用；Standard 維持既有的色度／亮度雜訊抑制。

**CuPy 一次只處理一張**（序列化），但單張 RAW HQ demosaic 在現代 NVIDIA 上通常仍比 CPU Fast RAW 快約 **1.25–1.4×**——方向鍵高品質瀏覽時感受得到，但不是倍數級加速。

**選 Standard**：安裝精簡、目視篩片。**選 Plus（DirectML）**：要 AI 搜尋（多數 PC 的建議預設）。**選 Plus（CUDA）**：有 NVIDIA，且想要上述略快的 RAW HQ demosaic（不必下載數 GB 的 PyTorch）。

Windows 安裝程式提供 **Standard**、**Plus（DirectML）** 與 **Plus（CUDA）**。macOS Plus 搜尋用 Core ML，demosaic 仍為 CPU Fast RAW（無 CuPy 路徑）。

**語意搜尋**不需要 PyTorch。


## 相機與格式

**各大品牌 RAW：** Canon（CR2/CR3）、Nikon（NEF）、Sony（ARW）、Fujifilm（RAF）、Olympus（ORF）、Panasonic（RW2）、Adobe DNG 及多數其他相機。**另支援：** JPEG、TIFF、HEIF，與動畫 GIF／WebP。

**Nikon 高效率（HE/HE\*）NEF** 以相機內建預覽開啟——瀏覽、篩片、評分皆正常。這類檔案暫時無法在 Adjust 顯影；標準與無損 NEF 顯影不受影響。

**HDR 靜態影像**（HEIC／HEIF／AVIF／HDR TIFF）會轉換為標準亮度顯示，以維持瀏覽速度。

<details>
<summary><strong>哪些相機會顯示對焦點（F）</strong></summary>

| 品牌 | 支援 |
|------|------|
| Canon CR2/CR3、Nikon NEF、Sony ARW、Olympus ORF、Panasonic RW2 | 是 |
| JPEG / TIFF / HEIF | 有時（相機有記錄主體區域時） |
| Fujifilm RAF、Hasselblad 3FR、Pentax PEF、Samsung SRW、Sigma X3F | 否 |
| 常見 Adobe DNG | 通常否 |

</details>

## 地圖與地點

在任何含 GPS 的相片上按 **M**，即可在互動地圖上看到拍攝位置；點擊座標徽章可開啟 Google Maps。地名解析完全離線，沒有網路也能搜尋 `tokyo` 或 `city:taipei`。

想一次瀏覽整本相簿的拍攝地圖，或為沒有 GPS 的相片補上位置，請參考姊妹作 **[LocateIt](https://github.com/markyip/LocateIt)**。

## 你的相片只屬於你

RAWviewer 不上傳任何東西。搜尋、地圖、AI 功能全部在你的電腦上執行。唯二需要網路的情況都是可選的：完整版的一次性 AI 模型下載，以及開啟地圖時載入圖磚。本機縮圖快取用於加速圖庫，僅存於你的電腦，且 30 天未使用會自動清除。

### 從舊版升級

多數快取升級後仍可繼續使用。過期的縮圖與 EXIF 會在**重新開啟資料夾時自動重建**（第一次可能較慢）。

**搜尋／索引速度**則不同：若本機已有升級前的 `~/.rawviewer_cache`，RAWviewer 會維持較舊、較保守的效能模式，讓升級更可預期。若要啟用較新、較快的預設：

1. 關閉 RAWviewer。
2. **Windows：** 重裝／升級時在 Setup 勾選 **Clear existing cache**，或執行安裝目錄內與 `RAWviewer.exe` 同層的 **`clear_cache.bat`**。
   **macOS：** 雙擊發行壓縮檔內的 **Clear Cache.command**（或 `bash clear_macos_cache.sh`）。
3. 重新開啟資料夾——首次索引會重建快取，之後即可享受較佳效能。

這只清除本機快取與工作階段，**不會刪除相片或 XMP**。進階用法也可設 `RAWVIEWER_PERF_V2=1` 而不清快取。

---

## 疑難排解

**系統需求：** Windows 10+ · macOS 13+ · 8 GB RAM（Plus + 大型資料夾建議 16 GB+）。

<details>
<summary><strong>所有平台</strong></summary>

| 問題 | 處理方式 |
|------|----------|
| GPS 地圖不顯示 | 單張檢視按 **M**；僅含 GPS 的相片會顯示地圖 |
| HDR HEIC/TIFF 偏平或偏暗 | v3.0 設計上會將 HDR 靜態影像轉為標準亮度 |
| **P** / **J** 無效 | **P**/**J** 僅 RAW/DNG 單張；**P** 是符合模式的 Reconstruct 復原預覽，不是 Highlights／Shadows 滑桿 |
| 大型資料夾首次開圖庫較慢 | 正常——等待拍攝時間排序以確保順序；中繼資料已快取時瞬間完成 |
| 升級後搜尋／圖庫仍偏慢 | 執行一次 **`clear_cache`**（見[從舊版升級](#從舊版升級)），再重開資料夾 |

清除快取：**`scripts\Launch\windows\clear_cache.bat`**（Windows）· **`scripts/Launch/macos/clear_cache.sh`**（Mac）

</details>

<details>
<summary><strong>Windows</strong></summary>

| 問題 | 處理方式 |
|------|----------|
| SmartScreen 警告 | 詳細資訊 → 仍要執行 |
| AI 搜尋慢（**Plus**） | 兩個 Plus 版本的 AI 搜尋都走 DirectML——換版本不會讓搜尋變快。若要在 NVIDIA 上略加快 RAW HQ 瀏覽，可試 **Plus（CUDA）**（demosaic 約比 CPU 快 1.25–1.4×） |
| 安裝卡在「Downloading models」（**Plus**） | 模型約 700 MB，可能需數分鐘。失敗時檢查防火牆、VPN 或 Proxy——瀏覽仍可用；稍後開圖庫 **Search** 重試 |
| 又開啟 Setup 而非程式 | 從桌面捷徑啟動 **RAWviewer**——不是 **`RAWviewer_Setup.exe`** |
| 安裝後無 AI 搜尋（**Plus**） | 開圖庫 **Search** → 接受下載提示 |
| 開啟方式沒有 RAWviewer | 重新執行安裝（修復）或重裝 |
| 解除安裝後殘留快取 | 再執行 **`uninstall.bat`**，或手動刪除 `%USERPROFILE%\.rawviewer_cache` |
| AI 索引時記憶體不足 | 8 GB 機器請用 **Standard**，或見[記憶體調校](docs/DEVELOPING.md#automatic-memory-tuning) |
| 低記憶體機在重啟還原上次資料夾時 OOM／退出 | **還原上次資料夾／檔案是預期功能**。8 GB 機器壓力大時請用 **Standard**，或設 `RAWVIEWER_DISABLE_SESSION_RESTORE=1` 關閉還原 |
| RAW 總是顯示 demosaic 而非內嵌 JPEG | 切換至**內嵌 JPEG 工作流程** |
| 當機 | 一般執行**不會**寫入工作階段日誌。設 `RAWVIEWER_FILE_LOG=1` 或用 **`run_with_debug_log.bat`** 啟用後，再查 `%LOCALAPPDATA%\RAWviewer\logs\` |
| 需要從**安裝版**抓凍結／圖庫 log | 雙擊安裝目錄內 **`run_with_debug_log.bat`**（或 repo 根目錄同名腳本）。重現問題後關閉 App，把 `%LOCALAPPDATA%\RAWviewer\logs\rawviewer_latest.log` 傳出即可 |

</details>

<details>
<summary><strong>macOS</strong></summary>

| 問題 | 處理方式 |
|------|----------|
| 系統阻擋（「損毀」/ 無法開啟） | 在解壓資料夾執行 `bash install_macos_app.sh`（見上方安裝步驟） |
| `bash: command not found` | 輸入 `cd `，將解壓資料夾拖入終端機，按 Return，再執行指令 |
| 無法讀取桌面 / 文件 | 系統設定 → 隱私權 → **完整磁碟存取** → 加入 RAWviewer |
| 搜尋提示缺少模型（**Plus**） | 開圖庫搜尋，出現提示時點 **Download**（需網路一次） |
| 匯出選單沒有 **AI denoise** | 預期行為——匯出時 SCUNet 降噪已於 3.0.1 移除；請用 Adjust 色度降噪 + JPEG／WebP／TIFF 匯出。 |
| 下載失敗（SSL / 憑證） | 企業 VPN / Proxy 請將根憑證加入**鑰匙圈**並設為**永遠信任** |
| 需完整解除安裝 | 使用壓縮檔內 **`Uninstall RAWviewer.command`**——勿只丟垃圾桶 |
| 找不到解除安裝腳本 | 從 [Releases](https://github.com/markyip/RAWviewer/releases/latest) 重新下載；腳本在解壓資料夾內 |
| 索引時記憶體不足 / 大量 swap | 8 GB Mac 建議 **Standard** 或待索引完成；見[記憶體調校](docs/DEVELOPING.md#automatic-memory-tuning) |
| 重開被終止（終端機顯示 `Killed: 9`） | **還原上次資料夾是預期功能**。8 GB Mac 記憶體壓力大時可試 **Standard**、`RAWVIEWER_DISABLE_SESSION_RESTORE=1`（關閉還原）或 `RAWVIEWER_ENABLE_SEMANTIC_SEARCH=0` |
| 大型資料夾圖庫仍卡頓 | 執行 **`clear_cache.sh`** 後重開資料夾 |

</details>

---

## 展望

**3.1.0** 是[相對 3.0.2 的穩定度／效能補丁](#未發行的-310相對-302-的穩定度效能)。下列為**後續車次（3.2+）**。其中許多**已在 `development` 工作區**（有些只在髒工作區）。程式碼**保留**；**不是** 3.1.0 宣稱要出的內容。

原則：**只要 Plus 版能交付就算可行**，即使 Standard 因大小／無 ML 而必須省略。

### 後續車次（3.2+）— 在工作區，不是 3.1.0

| 範圍 | 位置 | 說明 |
|------|------|------|
| **遮罩分頁**（筆刷、亮度、色彩、反轉、群組）＋ Plus AI 遮罩／景深／Blur background | `development` | 不是 3.1.0 產品宣稱 |
| **Auto Tone**（白平衡＋光線＋拉直；不烤 P／不做 VLM） | `development` | |
| **合成** — HDR、全景、全景 HDR、對焦疊加 | `development` | |
| **點陣顯影**（JPEG／TIFF／WebP／PNG／BMP + XMP） | `development` | HE-NEF *拒檔*修正是 3.1.0 例外 |
| **色卡**、手動畸變、變形鏡頭反壓平 | `development` | DPReview 樣本按鈕在髒工作區 |
| **分享** WhatsApp／Signal（RAW 轉 JPEG）與 AirDrop | `development` | |
| **AI 2× 放大** | `development` | Plus 降噪*可靠度*屬 3.1.0 |
| **彩色標籤／關鍵字** | 髒工作區 | 標籤**快取持久化**是 3.1.0 針對的硬化，不是目錄 UI |
| **RAW+JPEG 成對**、Adjust 左右停駐、LUT 縮圖、splash | 髒工作區 | |
| **匯出樣板／去 GPS／浮水印／佇列** | 髒工作區 | 該車次出貨時佇列不可丟工作；浮水印用**角落小圖**（非整頁疊圖）省記憶體 |
| **Headless MCP**、Share to AI agent | 髒工作區 | GUI **不會**即時重載 MCP sidecar |
| **App 內介面語言** | 髒工作區（`i18n/`） | 文案在磁碟；接上 `QTranslator` 前介面仍為英文 |

### 尚未完成

| 排序 | 項目 | 可行性 | 工作量 | 說明 |
|------|------|--------|--------|------|
| 1 | **GUI 即時重載 MCP sidecar** | **高** | M | Headless MCP 已能寫 XMP 與預覽。Adjust／圖庫不會監看那些 sidecar——需重開相片（或 Adjust）才看得到 agent 的編輯 |
| 2 | **App 內介面語言** | **高** | M | `i18n/catalog.json` 已備妥英／繁中／日文案。尚未接上 `QTranslator`，介面仍為英文 |
| 3 | **冷資料夾已編輯縮圖重生** | **高** | M | Adjust 儲存已烘焙對齊編輯器的縮圖；從未開過 Adjust 的資料夾仍可能顯示內嵌 JPEG |
| 4 | **使用者可調的放大強度** | **高** | S | Real-ESRGAN 全強度偏銳；混合比例目前是環境變數，不是滑桿 |
| 5 | **AI 選取的負向點擊** | **中偏高** | M | 模型支援「不要這個」的點，目前只送正向點擊。選過頭請用筆刷修 |
| 6 | **具名人物／熟面孔分群** | **中** | L | Plus 可篩 `has:face`。沒有人物名冊，也不會把「同一個人」聚在一起 |
| 7 | **透過本機伺服器進行生成式編輯** | **高** | S | **App 內已暫緩。** 後端停在 `generative-editing`。日後走本機 OpenAI 相容 `/v1/images/edits`，不是內嵌 6–7 GB 模型 |
| 8 | **DNG 匯出／往返** | **中** | L | Writer 已於 2026-07 移除；需要真正的 DNG 路徑 |
| 9 | **Windows HDR 顯示路徑** | **中** | L | macOS EDR 為 Fast RAW 速度而移除；Windows 仍為 SDR 色調映射 |
| 10 | **在 Fast RAW 下恢復 macOS EDR** | **低偏中** | L | 先前與快速載入管線衝突 |
| 11 | **VLM 輔助 Auto** | **低偏中** | L | 產品 Auto 只有白平衡＋光線＋拉直，沒有像素／VLM 風格。視覺模型 Auto 不在範圍內 |
| 12 | **Google Drive 瀏覽／編輯／XMP 同步** | **中** | L | 本機快取同步＋OAuth；尚未開始 |
| 13 | **以感光元件資料編輯 Nikon HE/HE\* NEF** | **低** | L+ | 目前走內嵌 JPEG；真正 RAW 仍待 HE 馬賽克解碼器 |

**目前限制（不是願望清單）：**
- 從未在 Adjust 開啟過的已編輯相片，**冷圖庫縮圖**可能仍顯示內嵌 JPEG。
- **Nikon HE-NEF** 一旦允許 Adjust（3.1.0 拒檔修正），以內嵌 JPEG（8-bit）顯影。
- **沒有記憶體分級選單**——RAM 上限是自動的。
- **僅 3.2+：** AI 遮罩／放大權重首次使用才下載；智慧物件只回傳一個遮罩；浮水印僅 JPEG／WebP；Apply to Burst Group 複製色調／色彩／降噪，不複製裁切、Dodge/Heal 或局部遮罩。

---

## 開發者

建置腳本、環境變數、記憶體調校與架構說明：**[docs/DEVELOPING.md](docs/DEVELOPING.md)**（英文）。啟動／打包腳本見 **[`scripts/Launch/`](scripts/Launch/README.md)**（`windows/` · `macos/` · `macos/release/`）。歡迎提交 Pull Request。

### 從原始碼建置（摘要）

| 平台 | 指令 | 產出 |
|------|------|------|
| **Windows** | `scripts\Launch\windows\build_windows.bat` | `dist/RAWviewer_Setup.exe`（精靈可選 Standard／Plus DirectML／Plus CUDA） |
| **macOS Plus** | `./scripts/Launch/macos/build_macos_full.sh` | `dist/RAWviewer.app` + 發行 zip |
| **macOS Standard** | `./scripts/Launch/macos/build_macos_lite.sh` | `dist/RAWviewer_Lite.app` + 發行 zip |

日常開發不必打包：`pixi install && pixi run start`，或 `scripts\Launch\windows\run_debug.bat menu`／`./scripts/Launch/macos/launch_dev_full.sh`。

**Plus（CUDA）** 安裝 `cupy-cuda12x` 做 GPU demosaic（序列化；單張通常比 CPU Fast RAW 快約 1.25–1.4×；不再捆綁數 GB 的 PyTorch）。**Plus（DirectML）** 與 **Standard** 使用 CPU Fast RAW。Windows 兩個 Plus 版本的 AI 搜尋都走 ONNX DirectML；macOS Plus 用 Core ML。

## 支援

1. 先查上方[疑難排解](#疑難排解)
2. 搜尋[既有議題](https://github.com/markyip/RAWviewer/issues)
3. 若仍無解，請開新議題並附上作業系統版本、步驟與日誌

## 致謝

RAWviewer 建立在出色的開源專案之上，包括：

- **[LibRaw](https://www.libraw.org/)** / **[rawpy](https://github.com/letmaik/rawpy)** —— RAW 解碼
- **[MobileCLIP](https://github.com/apple/ml-mobileclip)**（Apple）—— 裝置端相片描述搜尋（完整版）
- **[Qt / PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** —— 應用程式框架
- **[CuPy](https://cupy.dev/)** —— Windows Plus（CUDA）可選的 NVIDIA GPU demosaic
- **[onnxruntime](https://onnxruntime.ai/)** —— MobileCLIP ONNX 推論（Windows Plus）

## 授權

MIT——見 [LICENSE](LICENSE)。

## ☕ 請我喝杯咖啡

若 RAWviewer 對你的工作流程有幫助，歡迎[請我喝杯咖啡](https://www.buymeacoffee.com/markyip)。

---

**享受你的相片。** 📸
