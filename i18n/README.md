# UI copy: English, Traditional Chinese, Japanese

Not wired to `QTranslator` yet (`build.py` still strips Qt’s own `.qm` files). This folder is the wording source for when language becomes a setting.

## Files

| File | Role |
|------|------|
| [`catalog.json`](catalog.json) | Every prepared string: `id` → `{en, zh_TW, ja, ref}` |
| This README | How terms were chosen |

`ref` names the convention we followed. Brand names (WhatsApp, Signal, AirDrop, XMP, LUT, RAW) stay untranslated.

## Sources

- **Lightroom Classic** UI (zh-Hant and ja): Develop sliders, Library/cull verbs, masks.
- **Capture One** (ja/zh): “不採用／淘汰” style reject, not LR’s flag-only vocabulary.
- **darktable / RawTherapee** manuals: histogram, clipping, filmstrip.
- **Apple HIG** (zh-Hant / ja): File, Open, Share, Reveal in Finder.

Labels are short nouns (slider row). Tooltips may be a sentence. Japanese uses Lightroom’s 体言止め on labels (`露光量`, not `露光量を調整`).

## Product names we keep

| English | zh-TW | ja | Why |
|---------|-------|-----|-----|
| Adjust | Adjust | Adjust | Panel title in the app; LR’s 顯影／現像 is the *module*, we are not LR. Section copy may say 調整／補正. |
| Discard | Discard | Discard | Folder name on disk; UI can say 淘汰／不採用 next to it. |
| Plus / Standard | Plus／Standard | Plus／Standard | Edition names. |

## Applying later

1. Wrap user-visible literals in `self.tr("…")` using the **English** `en` value as the source key (Qt idiom).
2. Convert `catalog.json` → `rawviewer_zh_TW.ts` / `rawviewer_ja.ts` (`pylupdate` / `lrelease`).
3. Load `QTranslator` from the chosen language; default **follow the OS**, restart to apply.
4. Stop pruning app `.qm` in `build.py` (Qt’s own widget translations can stay pruned).
