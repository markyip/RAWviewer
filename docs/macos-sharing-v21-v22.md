# macOS 分享：v2.1 與 v2.2 行為差異調查

> 調查期間：2026-06-04  
> 環境：macOS 25.5.0，PyQt6，RAWviewer v2.2（`/Volumes/Development/Development/RAWviewer`）  
> 對照：v2.1（`/Users/markyipkk/Developments/RAWviewer`，commit `7ed0536` 一帶）

## 摘要

| 項目 | 結論 |
|------|------|
| 檔案與系統能力 | **正常**。`NSSharingService.sharingServicesForItems_` 對同一 JPG 可回傳 **8** 個服務（含 AirDrop）。 |
| v2.1 單張分享 | **可用**：約 15 行 `NSSharingServicePicker`，錨在分享按鈕 `winId()`。 |
| v2.2 原生 picker（未改前） | **常轉圈、無選項**（狀況 B）；log 仍可能顯示 `show` 成功。 |
| 根因 | **非**「抄錯 API」，而是 **Qt/AppKit 整合**（觸發時機、event filter、OpenGL 疊層、delegate 等）。 |
| v2.2 目前 dev 預設 | **Qt 選單**（`RAWVIEWER_SHARE_MENU=1`）；原生 picker 為 opt-in（`RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1`）。分享期間可暫停 event filter、隱藏 gpu_view；AirDrop 預設不在選單內。 |

POC（極簡 `QMainWindow` + 延遲呼叫 picker）可正常顯示完整分享面板，證明 **AppKit 路徑在機器上沒壞**，差異在 **完整應用殼層**。

---

## v2.1 與 v2.2 實作對照

### 相同（意圖上）

- 分享按鈕 → `_share_current_image_os` → macOS 上 `_share_macos`
- v2.1 `_share_macos` 核心（約略）：

```python
url = NSURL.fileURLWithPath_(path)
btn = self.share_bottom_button
view = objc.objc_object(c_void_p=int(btn.winId()))
rect = NSMakeRect(0, 0, w, h)
picker = NSSharingServicePicker.alloc().initWithItems_([url])
picker.showRelativeToRect_ofView_preferredEdge_(rect, view, 3)  # NSMaxYEdge
```

### v2.2 曾額外引入、後來部分還原的差異

| 項目 | v2.1 | v2.2（調查期間曾出現） |
|------|------|------------------------|
| 分享按鈕訊號 | `clicked`（mouseUp） | 曾改 `clicked` + 實驗路徑；**應使用 `pressed` + 延遲 `QTimer`** |
| Status bar event filter | 無遞迴 filter | 曾對整條 status bar 遞迴 `installEventFilter` |
| `centralWidget` filter | 無（非 frameless 為主） | 有（frameless 邊緣 resize） |
| `single_view_container` filter | 無 | 曾有 |
| GPU 單張檢視 | 無 | `RAWVIEWER_GPU_VIEW`（OpenGL `QGraphicsView`） |
| `_activate_macos_foreground_app` | 無 | 多處 `activateIgnoringOtherApps` |
| 分享 UI 退路 | 僅 picker | 曾改 **QMenu + `performWithItems_`**（Mail 等可用，AirDrop 常怪） |
| 啟動 splash | dev 也顯示 | v2.2 dev 可關閉（frozen 才強制） |

---

## 為何 POC 正常、v2.2 主程式失敗

### 1. `clicked` = mouseUp（macOS 13+ share sheet）

Apple / 社群回報：`NSSharingServicePicker` **不應在 mouseUp 觸發**。  
Qt `QPushButton.clicked` 在**放開滑鼠**時才觸發 → 分享 sheet 可能**一直轉圈、無選項**。Console 可能出現：

`should not be called on mouseUp Please configure the sender with sendActionOn:NSLeftMouseDownMask`

**POC** 用 `QTimer.singleShot` 延遲開 picker，**不是**模擬按鈕 `clicked` 路徑，故 POC 不能證明「按鈕接法相同就會成功」。

**修正方向：** `pressed` + `QTimer.singleShot(50, …)` 再開 picker。

### 2. Event filter 干擾

v2.2 `MainWindow.eventFilter` 裝在 scroll / image / GPU viewport、`centralWidget`、title bar、拖放目標、gallery 等（遠多於 v2.1 的 scroll + image）。

- 曾在 `show` 後 **500ms 恢復 filter**，picker **非同步載入**服務時又被 Qt 攔事件 → 可能持續轉圈。
- **修正：** 分享期間暫停 filter，在 picker delegate `didChoose` 或長逾時後再恢復；status bar 不要遞迴 filter。

### 3. OpenGL（GPU view）疊在原生 UI 上

`RAWVIEWER_GPU_VIEW=1` 時 `gpu_view.raise_()`，OpenGL 表面常畫在 Cocoa **之上**。  
`NSSharingServicePicker` 掛在 `NSWindow.contentView` 時，面板可能：

- 被擋在圖後面（像「沒出來」），或  
- 整合異常（轉圈）

**單張模式時 gallery 已 `hide()`**，故通常**不是**「被 gallery 縮圖蓋住」，而是 **GPU/整張圖層**。

**驗證：** `RAWVIEWER_GPU_VIEW=0 RAWVIEWER_AIRDROP_PICKER=1` 再測 in-app picker。

### 4. 錨點與 `preferredEdge`

- 巢狀 status bar 內按鈕的 `winId()` 在完整 app 內不如 POC 穩。
- 改 **主視窗 `contentView` + 按鈕座標** 較可靠。
- 錯用 **`NSMinYEdge`**（面板往視窗**下方**開）在 status bar（`rect` 近 `y≈0`）可能開到螢幕外 → 像轉圈。v2.1 / POC 用 **`edge=3`（`NSMaxYEdge`，在按鈕上方）**。

### 5. `performWithItems` / delegate 陷阱

- Log 寫 **`performWithItems OK`** ≠ 使用者看到 AirDrop 面板（Qt 整合問題）。
- Delegate 回傳 **`(NSWindow, scope)` 元組** → PyObjC 變成 `OC_BuiltinPythonArray`，系統當 `NSWindow` 用 → `styleMask` 崩潰。
- `sharingContentScope` 的 `^q` 指標 → `ObjCPointerWarning`（宜只回傳 `NSWindow` 或正確處理 out 參）。

### 6. AirDrop 專屬

- 選單裡列表的 `NSSharingService` 物件對 AirDrop **`performWithItems_` 常不可靠**；應 `sharingServiceNamed_("com.apple.share.AirDrop.send")` + **`NSSharingServiceDelegate` 提供 `sourceWindow`**。
- 檔案在 **`/Volumes/...`** 時可先 **複製到暫存** 再分享。
- 在 Qt app 內仍可能無 UI → **預設改 Finder `open -R`**（右鍵 → 分享 → AirDrop），與系統 Finder 行為一致。

---

## 常見失敗原因與 v2.2 防護（2026-06-04）

| 原因 | 症狀 | 程式防護 |
|------|------|----------|
| **缺少 NSWindow / NSView** | Picker 轉圈、無選項 | `_macos_share_ensure_appkit_context`：`raise_` / `activateWindow` / `setActivationPolicy_(Regular)` / `makeKeyAndOrderFront`；picker 要求 `contentView` 錨點；`NSSharingServiceDelegate` 提供 `sourceWindow` |
| **非主執行緒呼叫 AppKit** | 不穩定或無 UI | `_macos_share_defer_to_main` + `_macos_share_on_main_thread`；分享入口經 `QTimer`（按鈕 `pressed` + 50ms） |
| **主執行緒長時間阻塞** | Sheet 卡住 | `/Volumes/` 複製會 log 耗時（ms）；大檔仍會短暫阻塞——避免在背景 thread 直接呼叫 AppKit |
| **Sandbox / Entitlements** | 正式包分享失敗、無服務 | **dev**（`python src/main.py`）通常**無** App Sandbox；**frozen** `.app` 需 `NSRemovableVolumesUsageDescription` 等（見 `build.py` `update_macos_plist`）。log：`runtime: frozen=...` |
| **空或非法 items** | `sharingServicesForItems` = 0 或轉圈 | `_macos_share_validate_file`（存在、可讀、非 0 byte）；`NSURL` 必須 `isFileURL`；`NSArray.arrayWithArray_([url])` 且 `count >= 1` |

---

## 狀況分類（調查過程）

| 代號 | 現象 | 含義 |
|------|------|------|
| A | v2.1 picker 正常 | 基線 |
| B | `sharingServicesForItems_` 有 8 個，picker **轉圈** | 路徑 OK，**UI/事件層**壞 |
| C | QMenu 有項目，AirDrop 無效 | `perform` / delegate / 服務物件問題 |
| D | log `shown OK` / `perform OK`，**看不到面板** | 可能被 GPU 擋或開在螢幕外 |

---

## v2.2 目前 dev 行為（以 `launch_dev.sh` 與 `main` 為準）

### `scripts/Launch/shell/launch_dev.sh` 預設

| 變數 | 預設 | 說明 |
|------|------|------|
| `RAWVIEWER_GPU_VIEW` | `1` | GPU 單張檢視 on |
| `RAWVIEWER_SHARE_MENU` | `1` | **Qt 選單**（v2.2 預設）；`0` = 僅原生 picker 路徑 |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER` | off | `1` = 先試 picker，約 900ms 後 fallback 選單 |
| `RAWVIEWER_SHARE_SHOW_AIRDROP` | off | `1` = 選單顯示 AirDrop |
| `RAWVIEWER_SHARE_KEEP_GPU_VISIBLE` | off | `1` = 分享時不隱藏 `gpu_view`（除錯 picker 與 OpenGL 衝突） |
| `RAWVIEWER_MACOS_FORCE_FOREGROUND` | off | 僅 opt-in 時 `activateIgnoringOtherApps` |

### 執行權限

`launch_dev.sh` 需可執行：`chmod +x scripts/Launch/shell/launch_dev.sh`  
若 `permission denied`，改用：

```bash
RAWVIEWER_GPU_VIEW=0 RAWVIEWER_AIRDROP_PICKER=1 bash /Volumes/Development/Development/RAWviewer/scripts/Launch/shell/launch_dev.sh
```

### 環境變數一覽

| 變數 | 用途 |
|------|------|
| `RAWVIEWER_SHARE_MENU=1` | **Qt 選單**（**v2.2 預設**；Mail 等可靠） |
| `RAWVIEWER_SHARE_MENU=0` | 僅走 **NSSharingServicePicker**（常轉圈） |
| `RAWVIEWER_SHARE_TRY_NATIVE_PICKER=1` | 先 picker，失敗/逾時 fallback 選單 |
| `RAWVIEWER_SHARE_KEEP_GPU_VISIBLE=1` | 分享時不隱藏 OpenGL 層（A/B 測試） |
| `RAWVIEWER_AIRDROP_PICKER=1` | AirDrop 改試 **僅 AirDrop 的 picker** |
| `RAWVIEWER_AIRDROP_PERFORM=1` | AirDrop 改試 **`performWithItems`**（常 log OK、無 UI） |
| （預設，無上述 AirDrop 變數） | AirDrop → **Finder 顯示暫存/原檔** |
| `RAWVIEWER_GPU_VIEW=0` | 關 OpenGL，測是否 picker 恢復 |
| `RAWVIEWER_MACOS_FORCE_FOREGROUND=1` | Terminal 啟動時強制前景（檔案對話框用；分享預設不開） |
| `RAWVIEWER_SHARE_DEBUG=1` | 額外 share 診斷 |

### Log 關鍵字

```
[SHARE] share requested:
[SHARE] sharingServicesForItems count=
[SHARE] picker show anchor=ns.contentView edge=
[SHARE] AirDrop temp copy:
[SHARE] AirDrop picker shown
[SHARE] AirDrop Finder fallback: open -R
```

---

## 建議後續（暫停實作前）

1. **產品預設（v2.2）**：**Qt 選單** + AirDrop 預設隱藏（Finder 路徑）；原生 picker 僅 opt-in。  
2. **若要 v2.1 體驗**：`pressed` + 延遲 picker + `contentView` + `edge=3` + 分享期間暫停 filter + 預設 `GPU_VIEW=0` 或分享時暫時降 OpenGL。  
3. **不要**僅複製 v2.1 的 `_share_macos` 15 行而忽略殼層差異。  
4. **Bisect**：同一檔案、同一機器，v2.1 `launch_dev.sh` vs v2.2（`GPU_VIEW` / `SHARE_MENU` 矩陣）並行對照。

---

## 相關檔案

| 檔案 | 內容 |
|------|------|
| `src/main.py` | `_share_current_image_os`、`_share_macos`、`_share_macos_services_menu`、`_perform_macos_share_airdrop`、event filter 安裝處 |
| `scripts/Launch/shell/launch_dev.sh` | dev 預設 env |
| v2.1 參考 | `~/Developments/RAWviewer/src/main.py` 約 `_share_macos`、status bar 初始化 |

---

## 參考

- [NSSharingServicePicker — show(relativeTo:of:preferredEdge:)](https://developer.apple.com/documentation/appkit/nssharingservicepicker)  
- Stack Overflow：picker 應在 main queue **非 mouseUp** 觸發；`sendActionOn:NSLeftMouseDownMask`  
- Flutter share_plus #1177 / #1223：macOS picker 需非同步於主執行緒  

---

*本文件為調查筆記，不代表最終產品規格；程式以工作區 `main.py` / `launch_dev.sh` 為準。*
