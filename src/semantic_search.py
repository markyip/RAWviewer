"""
Lightweight local semantic image search (text -> image) with EXIF-aware metadata.

MVP goals:
- 100% local index and search (SQLite + CLIP embeddings)
- Incremental indexing by file mtime/size
- Optional metadata filters for quick narrowing
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import hashlib
import sys
import gzip
import urllib.request
import tempfile
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

import exifread

try:
    import pycountry
except Exception:
    pycountry = None


ProgressCallback = Optional[Callable[[int, int, str], None]]


def _load_index_source_image(file_path: str, max_size: int = 1024) -> Image.Image:
    """Load a small RGB image suitable for indexing/detection, preferring app caches."""
    try:
        from image_cache import get_image_cache
        cache = get_image_cache()
        for getter_name in ("get_thumbnail", "get_preview"):
            try:
                arr = getattr(cache, getter_name)(file_path)
                if arr is not None:
                    im = Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("RGB")
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im
            except Exception:
                continue
    except Exception:
        pass

    try:
        with Image.open(file_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
            return im.copy()
    except Exception:
        pass

    try:
        import rawpy  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
        ) from exc

    with rawpy.imread(file_path) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb is not None:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(BytesIO(thumb.data))
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im.copy()
                if thumb.format == rawpy.ThumbFormat.BITMAP:
                    im = Image.fromarray(thumb.data, mode="RGB")
                    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
                    return im
        except Exception:
            pass

        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            half_size=True,
            output_bps=8,
        )
    im = Image.fromarray(rgb, mode="RGB")
    im.thumbnail((max_size, max_size), Image.Resampling.BICUBIC)
    return im


@dataclass
class SearchHit:
    file_path: str
    score: float
    file_name: str = ""
    capture_time: str = ""
    camera_model: str = ""
    lens_model: str = ""
    iso: int = 0
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    city: str = ""
    admin1: str = ""
    country: str = ""
    country_code: str = ""
    face_count: int = 0


class MobileCLIPCoreMLBackend:
    """Optional macOS Core ML backend for MobileCLIP.

    Model assets are expected in:
    - RAWVIEWER_MOBILECLIP_MODEL_DIR, or
    - ~/.rawviewer_cache/mobileclip_coreml

    Expected Apple filenames:
    - mobileclip_s2_image.mlpackage
    - mobileclip_s2_text.mlpackage

    Note: text encoding also needs a tokenizer compatible with the Core ML text
    encoder. Until a tokenizer asset is present, the backend reports unavailable
    rather than silently falling back to metadata-only results.

    Exported MobileCLIP2 models (see ``scripts/export_mobileclip2_coreml.py``)
    expose the image encoder as FLOAT32 MultiArray ``[1,3,256,256]`` in NCHW
    pixel scale ``[0,1]``. Apple-shipped MobileCLIP S2 bundles use MLFeatureTypeImage;
    ``encode_image`` supports both via model introspection.
    """

    MODEL_ID = "mobileclip-coreml-s2"
    HUB_REPO_ID = "apple/coreml-mobileclip"
    IMAGE_MODEL_FILE = "mobileclip_s2_image.mlpackage"
    TEXT_MODEL_FILE = "mobileclip_s2_text.mlpackage"
    TOKENIZER_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    SUPPORTS_HUB_DOWNLOAD = True

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = self._default_model_dir()
        self.model_dir = model_dir
        self.image_model_path = os.path.join(model_dir, self.IMAGE_MODEL_FILE)
        self.text_model_path = os.path.join(model_dir, self.TEXT_MODEL_FILE)
        self.tokenizer_path = os.path.join(model_dir, "bpe_simple_vocab_16e6.txt.gz")
        self._image_model = None
        self._text_model = None
        self._tokenizer = None
        self._CoreML = None
        self._Foundation = None
        self._Quartz = None

    @staticmethod
    def _candidate_model_dirs() -> List[str]:
        dirs: List[str] = []
        env_dir = os.environ.get("RAWVIEWER_MOBILECLIP_MODEL_DIR")
        if env_dir:
            dirs.append(env_dir)
        dirs.append(os.path.expanduser("~/.rawviewer_cache/mobileclip_coreml"))
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(sys.executable)
            dirs.extend(
                [
                    os.path.join(exe_dir, "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Resources", "mobileclip_coreml"),
                    os.path.join(exe_dir, "..", "Frameworks", "mobileclip_coreml"),
                ]
            )
        module_dir = os.path.dirname(os.path.abspath(__file__))
        dirs.extend(
            [
                os.path.join(module_dir, "..", "models", "mobileclip2_coreml"),
                os.path.join(module_dir, "..", "models", "mobileclip_coreml"),
                os.path.join(module_dir, "..", "mobileclip_coreml"),
            ]
        )
        out: List[str] = []
        for d in dirs:
            full = os.path.realpath(os.path.abspath(os.path.expanduser(d)))
            if full not in out:
                out.append(full)
        return out

    @classmethod
    def _default_model_dir(cls) -> str:
        for d in cls._candidate_model_dirs():
            if (
                os.path.exists(os.path.join(d, cls.IMAGE_MODEL_FILE))
                and os.path.exists(os.path.join(d, cls.TEXT_MODEL_FILE))
                and os.path.exists(os.path.join(d, "bpe_simple_vocab_16e6.txt.gz"))
            ):
                return d
        return cls._candidate_model_dirs()[0]

    def availability_error(self) -> str:
        if sys.platform != "darwin":
            return "MobileCLIP Core ML is only available on macOS"
        if not self._mlpackage_complete(self.image_model_path):
            return f"Missing MobileCLIP image model in {self.model_dir}"
        if not self._mlpackage_complete(self.text_model_path):
            return f"Missing MobileCLIP text model in {self.model_dir}"
        if not os.path.exists(self.tokenizer_path):
            return f"Missing MobileCLIP tokenizer in {self.model_dir}"
        try:
            import CoreML  # noqa: F401
            import Foundation  # noqa: F401
            import Quartz  # noqa: F401
        except Exception as exc:
            return f"Missing native Core ML runtime: {exc}"
        return ""

    def available(self) -> bool:
        return self.availability_error() == ""

    @staticmethod
    def _mlpackage_complete(path: str) -> bool:
        return (
            os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "Manifest.json"))
            and os.path.isfile(os.path.join(path, "Data", "com.apple.CoreML", "model.mlmodel"))
            and os.path.isfile(os.path.join(path, "Data", "com.apple.CoreML", "weights", "weight.bin"))
        )

    def download_assets(self, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        """Download MobileCLIP S2 Core ML assets into the backend model directory."""
        if sys.platform != "darwin":
            raise RuntimeError("MobileCLIP Core ML download is only supported on macOS")

        def _progress(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        os.makedirs(self.model_dir, exist_ok=True)
        _progress("Downloading MobileCLIP Core ML models...")
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "MobileCLIP auto-download requires 'huggingface_hub'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        snapshot_download(
            repo_id=self.HUB_REPO_ID,
            allow_patterns=[
                f"{self.IMAGE_MODEL_FILE}/**",
                f"{self.TEXT_MODEL_FILE}/**",
            ],
            local_dir=self.model_dir,
        )

        if not os.path.exists(self.tokenizer_path):
            _progress("Downloading MobileCLIP tokenizer...")
            urllib.request.urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)

        err = self.availability_error()
        if err:
            raise RuntimeError(err)
        _progress("MobileCLIP assets ready")
        return self.model_dir

    def _load_models(self):
        if self._image_model is not None and self._text_model is not None:
            return
        import CoreML
        import Foundation
        import Quartz

        self._CoreML = CoreML
        self._Foundation = Foundation
        self._Quartz = Quartz

        def _load_one(path: str):
            url = Foundation.NSURL.fileURLWithPath_(path)
            compiled_url, compile_error = CoreML.MLModel.compileModelAtURL_error_(url, None)
            if compile_error is not None or compiled_url is None:
                raise RuntimeError(f"Failed to compile Core ML model: {compile_error}")
            model, load_error = CoreML.MLModel.modelWithContentsOfURL_error_(compiled_url, None)
            if load_error is not None or model is None:
                raise RuntimeError(f"Failed to load Core ML model: {load_error}")
            return model

        self._image_model = _load_one(self.image_model_path)
        self._text_model = _load_one(self.text_model_path)

    @staticmethod
    def _native_feature_name(model, direction: str) -> str:
        desc = model.modelDescription()
        features = (
            desc.inputDescriptionsByName()
            if direction == "input"
            else desc.outputDescriptionsByName()
        )
        names = list(features.keys())
        if not names:
            raise RuntimeError(f"Core ML model has no {direction} features")
        return str(names[0])

    @staticmethod
    def _normalize(vec) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _ClipBPETokenizer(self.tokenizer_path)
        return self._tokenizer

    def encode_text(self, text: str) -> np.ndarray:
        self._load_models()
        CoreML = self._CoreML
        tokenizer = self._ensure_tokenizer()
        tokens = np.asarray([tokenizer.encode_for_clip(text)], dtype=np.int32)
        input_name = self._native_feature_name(self._text_model, "input")
        output_name = self._native_feature_name(self._text_model, "output")
        multi_array = self._int32_multi_array(tokens.reshape(-1))
        feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi_array)
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {input_name: feature}, None
        )
        if err is not None or provider is None:
            raise RuntimeError(f"Failed to create Core ML text input: {err}")
        out, err = self._text_model.predictionFromFeatures_error_(provider, None)
        if err is not None or out is None:
            raise RuntimeError(f"MobileCLIP text prediction failed: {err}")
        return self._normalize(self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue()))

    def _float32_multi_array_nchw(self, tensor_nchw: np.ndarray):
        CoreML = self._CoreML
        t = np.asarray(tensor_nchw, dtype=np.float32).reshape(1, 3, 256, 256)
        flat = np.ascontiguousarray(t).ravel(order="C")
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, 3, 256, 256], CoreML.MLMultiArrayDataTypeFloat32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML image tensor: {err}")
        for i, value in enumerate(flat):
            arr.setObject_atIndexedSubscript_(float(value), i)
        return arr

    def encode_image(self, file_path: str) -> np.ndarray:
        self._load_models()
        CoreML = self._CoreML
        im = _load_index_source_image(file_path, max_size=512).resize(
            (256, 256), Image.Resampling.BICUBIC
        )
        desc = self._image_model.modelDescription()
        input_name = self._native_feature_name(self._image_model, "input")
        output_name = self._native_feature_name(self._image_model, "output")
        by_name = desc.inputDescriptionsByName()
        feature_desc = by_name.objectForKey_(input_name) if by_name is not None else None
        if feature_desc is None and by_name is not None:
            for key in by_name:
                if str(key) == str(input_name):
                    feature_desc = by_name.objectForKey_(key)
                    break
        if feature_desc is None:
            raise RuntimeError("Core ML image model has no input description")
        in_type = int(feature_desc.type())

        feature = None
        if in_type == CoreML.MLFeatureTypeMultiArray:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
            multi = self._float32_multi_array_nchw(nchw)
            feature = CoreML.MLFeatureValue.featureValueWithMultiArray_(multi)
        elif in_type == CoreML.MLFeatureTypeImage:
            pixel_buffer = self._image_to_pixel_buffer(im)
            feature = CoreML.MLFeatureValue.alloc().initWithValue_type_(
                pixel_buffer, CoreML.MLFeatureTypeImage
            )
        else:
            raise RuntimeError(f"Unsupported Core ML image encoder input type: {in_type}")

        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            {input_name: feature}, None
        )
        if err is not None or provider is None:
            raise RuntimeError(f"Failed to create Core ML image input: {err}")
        out, err = self._image_model.predictionFromFeatures_error_(provider, None)
        if err is not None or out is None:
            raise RuntimeError(f"MobileCLIP image prediction failed: {err}")
        return self._normalize(self._multi_array_to_numpy(out.featureValueForName_(output_name).multiArrayValue()))

    def _int32_multi_array(self, values: np.ndarray):
        CoreML = self._CoreML
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(
            [1, int(values.size)], CoreML.MLMultiArrayDataTypeInt32, None
        )
        if err is not None or arr is None:
            raise RuntimeError(f"Failed to allocate Core ML token array: {err}")
        flat = np.asarray(values, dtype=np.int32).reshape(-1)
        for i, value in enumerate(flat):
            arr.setObject_atIndexedSubscript_(int(value), i)
        return arr

    @staticmethod
    def _multi_array_to_numpy(multi_array) -> np.ndarray:
        values = multi_array.numberArray()
        return np.asarray([float(values[i]) for i in range(len(values))], dtype=np.float32)

    def _image_to_pixel_buffer(self, im: Image.Image):
        Quartz = self._Quartz
        im = im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        status, pixel_buffer = Quartz.CVPixelBufferCreate(
            None,
            256,
            256,
            Quartz.kCVPixelFormatType_32BGRA,
            {
                Quartz.kCVPixelBufferCGImageCompatibilityKey: True,
                Quartz.kCVPixelBufferCGBitmapContextCompatibilityKey: True,
            },
            None,
        )
        if status != 0 or pixel_buffer is None:
            raise RuntimeError(f"Failed to create CVPixelBuffer: status {status}")
        rgba = np.asarray(im, dtype=np.uint8)
        bgra = np.empty((256, 256, 4), dtype=np.uint8)
        bgra[..., 0] = rgba[..., 2]
        bgra[..., 1] = rgba[..., 1]
        bgra[..., 2] = rgba[..., 0]
        bgra[..., 3] = 255
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 0)
        try:
            bytes_per_row = int(Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer))
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            buf = base.as_buffer(bytes_per_row * 256)
            row_bytes = 256 * 4
            for y in range(256):
                start = y * bytes_per_row
                buf[start:start + row_bytes] = bgra[y].tobytes()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
        return pixel_buffer


class MobileCLIP2CoreMLBackend(MobileCLIPCoreMLBackend):
    """Core ML encoders from ``scripts/export_mobileclip2_coreml.py --for-app``."""

    MODEL_ID = "mobileclip-coreml-2-s0"
    HUB_REPO_ID = ""
    IMAGE_MODEL_FILE = "mobileclip2_s0_image.mlpackage"
    TEXT_MODEL_FILE = "mobileclip2_s0_text.mlpackage"
    SUPPORTS_HUB_DOWNLOAD = False

    def download_assets(self, progress_callback: Optional[Callable[[str], None]] = None) -> str:
        if self._mlpackage_complete(self.image_model_path) and self._mlpackage_complete(self.text_model_path):
            if not os.path.exists(self.tokenizer_path):
                os.makedirs(self.model_dir, exist_ok=True)
                if progress_callback:
                    progress_callback("Downloading CLIP BPE tokenizer...")
                urllib.request.urlretrieve(self.TOKENIZER_URL, self.tokenizer_path)
            err = self.availability_error()
            if err:
                raise RuntimeError(err)
            return self.model_dir
        raise RuntimeError(
            "MobileCLIP2 Core ML models are not bundled. Build with "
            "`python scripts/export_mobileclip2_coreml.py --for-app` and copy outputs into "
            "models/mobileclip2_coreml/ (sources) or set RAWVIEWER_MOBILECLIP_MODEL_DIR, "
            "or install Apple MobileCLIP S2 with RAWVIEWER_MOBILECLIP_VARIANT=s2."
        )


def resolve_mobileclip_coreml_backend() -> MobileCLIPCoreMLBackend:
    """Prefer MobileCLIP2 on disk; fall back to downloadable Apple MobileCLIP S2."""
    variant = (os.environ.get("RAWVIEWER_MOBILECLIP_VARIANT") or "").strip().lower()
    if variant in ("s2", "legacy", "mobileclip-s2", "mobileclip1"):
        order: List[type] = [MobileCLIPCoreMLBackend, MobileCLIP2CoreMLBackend]
    elif variant in ("2", "mobileclip2", "mc2", "s0"):
        order = [MobileCLIP2CoreMLBackend, MobileCLIPCoreMLBackend]
    else:
        order = [MobileCLIP2CoreMLBackend, MobileCLIPCoreMLBackend]

    for cls in order:
        inst = cls()
        if inst.available():
            return inst
    return order[0]()


def _bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class _ClipBPETokenizer:
    """Minimal OpenAI CLIP BPE tokenizer for MobileCLIP Core ML text encoder."""

    def __init__(self, bpe_path: str):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges if merge]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    @lru_cache(maxsize=8192)
    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        for token in re.findall(r"<\|startoftext\|>|<\|endoftext\|>|[\w']+|[^\s\w]", text.lower()):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def encode_for_clip(self, text: str, context_length: int = 77) -> List[int]:
        tokens = [self.sot] + self.encode(text)[: context_length - 2] + [self.eot]
        return tokens + [0] * (context_length - len(tokens))


class SemanticImageIndex:
    """SQLite-backed CLIP embedding index for local semantic search."""

    def __init__(self, db_path: Optional[str] = None, model_name: Optional[str] = None):
        if db_path is None:
            cache_dir = os.path.expanduser("~/.rawviewer_cache")
            os.makedirs(cache_dir, exist_ok=True)
            db_path = os.path.join(cache_dir, "semantic_index.db")
        self.db_path = db_path
        if sys.platform == "darwin":
            self._mobileclip_backend = resolve_mobileclip_coreml_backend()
        else:
            self._mobileclip_backend = None
        if model_name is None:
            model_name = (
                getattr(type(self._mobileclip_backend), "MODEL_ID", MobileCLIPCoreMLBackend.MODEL_ID)
                if sys.platform == "darwin"
                else "clip-ViT-B-32"
            )
        self.model_name = model_name
        self._model = None
        self._reverse_geocoder = None
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_index (
                file_path TEXT PRIMARY KEY,
                file_name TEXT,
                file_signature TEXT,
                file_size INTEGER NOT NULL,
                file_mtime REAL NOT NULL,
                mtime_ns INTEGER,
                model_name TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                capture_time TEXT,
                camera_model TEXT,
                lens_model TEXT,
                iso INTEGER,
                width INTEGER,
                height INTEGER,
                gps_lat REAL,
                gps_lon REAL,
                gps_raw TEXT,
                city TEXT,
                admin1 TEXT,
                country TEXT,
                country_code TEXT,
                face_count INTEGER,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_capture_time ON semantic_index(capture_time)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_camera_model ON semantic_index(camera_model)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_gps_latlon ON semantic_index(gps_lat, gps_lon)"
        )
        # Backward-compatible migration for existing DBs created before GPS fields.
        cols = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(semantic_index)").fetchall()
        }
        if "file_name" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_name TEXT")
        if "file_signature" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN file_signature TEXT")
        if "mtime_ns" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN mtime_ns INTEGER")
        if "gps_lat" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lat REAL")
        if "gps_lon" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_lon REAL")
        if "gps_raw" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN gps_raw TEXT")
        if "city" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN city TEXT")
        if "admin1" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN admin1 TEXT")
        if "country" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country TEXT")
        if "country_code" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN country_code TEXT")
        if "face_count" not in cols:
            self._conn.execute("ALTER TABLE semantic_index ADD COLUMN face_count INTEGER")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_semantic_signature_model ON semantic_index(file_signature, model_name)"
        )
        self._conn.commit()
        self._backfill_file_signatures()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search requires 'sentence-transformers'. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def semantic_backend_available() -> bool:
        if sys.platform == "darwin":
            return resolve_mobileclip_coreml_backend().available()
        try:
            import sentence_transformers  # noqa: F401
            return True
        except Exception:
            return False

    def semantic_backend_error(self) -> str:
        if sys.platform == "darwin" or self.model_name.startswith("mobileclip-coreml"):
            backend = self._mobileclip_backend or resolve_mobileclip_coreml_backend()
            return backend.availability_error()
        try:
            self._ensure_model()
            return ""
        except Exception as exc:
            return str(exc)

    def mobileclip_supports_hub_download(self) -> bool:
        return bool(
            self.model_name.startswith("mobileclip-coreml")
            and self._mobileclip_backend is not None
            and getattr(self._mobileclip_backend, "SUPPORTS_HUB_DOWNLOAD", False)
        )

    def download_semantic_backend_assets(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        if sys.platform == "darwin" or self.model_name.startswith("mobileclip-coreml"):
            backend = self._mobileclip_backend or resolve_mobileclip_coreml_backend()
            path = backend.download_assets(progress_callback=progress_callback)
            self._mobileclip_backend = backend
            return path
        raise RuntimeError("This semantic backend has no downloadable assets")

    def _ensure_reverse_geocoder(self):
        if self._reverse_geocoder is not None:
            return self._reverse_geocoder
        try:
            import reverse_geocoder as rg  # type: ignore
        except ImportError:
            self._reverse_geocoder = False
            return False
        self._reverse_geocoder = rg
        return rg

    @staticmethod
    def _country_name_from_code(code: str) -> str:
        cc = (code or "").strip().upper()
        if not cc:
            return ""
        if cc == "TW":
            return "Taiwan"
        if pycountry is not None:
            try:
                c = pycountry.countries.get(alpha_2=cc)
                if c is not None:
                    name = str(getattr(c, "name", "") or "")
                    # Keep UI-friendly labels for search/filter display.
                    if name == "Taiwan, Province of China":
                        return "Taiwan"
                    return name
            except Exception:
                pass
        return cc

    @staticmethod
    def _to_blob(vec: np.ndarray) -> bytes:
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _from_blob(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0 and arr.size != dim:
            arr = arr[:dim]
        return arr

    @staticmethod
    def _safe_int(v) -> int:
        try:
            values = getattr(v, "values", None)
            if values:
                return SemanticImageIndex._safe_int(values[0])
            ratio_value = SemanticImageIndex._ratio_to_float(v)
            if ratio_value is not None:
                return int(round(ratio_value))
            text = str(v or "").strip()
            if not text:
                return 0
            # exifread may stringify numeric tags as "[800]" or "800 800".
            m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
            if not m:
                return 0
            token = m.group(0)
            if "/" in token:
                num, den = token.split("/", 1)
                den_f = float(den) if float(den) != 0 else 1.0
                return int(round(float(num) / den_f))
            return int(round(float(token)))
        except Exception:
            return 0

    @staticmethod
    def _tag_text(tags: Dict[str, object], *names: str) -> str:
        for name in names:
            value = tags.get(name)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @classmethod
    def _tag_int(cls, tags: Dict[str, object], *names: str) -> int:
        for name in names:
            if name not in tags:
                continue
            value = cls._safe_int(tags.get(name))
            if value:
                return value
        return 0

    @staticmethod
    def _pil_dimensions(file_path: str) -> tuple[int, int]:
        try:
            with Image.open(file_path) as im:
                return int(im.width), int(im.height)
        except Exception:
            pass
        try:
            import rawpy  # type: ignore
            with rawpy.imread(file_path) as raw:
                return int(raw.sizes.width), int(raw.sizes.height)
        except Exception:
            pass
        return 0, 0

    @staticmethod
    def _clean_join(*parts: str) -> str:
        out: List[str] = []
        for part in parts:
            p = (part or "").strip()
            if p and p not in out:
                out.append(p)
        return " ".join(out).strip()

    @staticmethod
    def _legacy_int(v) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    @staticmethod
    def _mtime_matches(stored_mtime: float, st: os.stat_result) -> bool:
        """
        Compare mtime with tolerance to avoid cross-session float precision drift.
        """
        try:
            current = float(st.st_mtime)
            return abs(float(stored_mtime) - current) <= 1e-3
        except Exception:
            return False

    @staticmethod
    def _canonical_path(file_path: str) -> str:
        try:
            return os.path.realpath(os.path.abspath(file_path))
        except Exception:
            return file_path

    @staticmethod
    def _path_aliases(file_path: str) -> List[str]:
        aliases: List[str] = []
        for p in (
            file_path,
            os.path.abspath(file_path),
            os.path.realpath(os.path.abspath(file_path)),
        ):
            if p and p not in aliases:
                aliases.append(p)
        return aliases

    @staticmethod
    def _mtime_ns_from_stat(st: os.stat_result) -> int:
        try:
            return int(st.st_mtime_ns)
        except Exception:
            return int(float(st.st_mtime) * 1_000_000_000)

    @classmethod
    def _file_signature_from_stat(cls, file_path: str, st: os.stat_result) -> str:
        """
        Stable per-file identity that survives path alias changes.
        """
        name = os.path.basename(file_path)
        raw = f"{name}\0{int(st.st_size)}\0{cls._mtime_ns_from_stat(st)}"
        return hashlib.sha1(raw.encode("utf-8", errors="surrogateescape")).hexdigest()

    def _backfill_file_signatures(self) -> None:
        """
        Populate signature columns for existing databases created before stable identity.
        """
        try:
            rows = self._conn.execute(
                """
                SELECT file_path
                FROM semantic_index
                WHERE file_signature IS NULL OR file_signature = '' OR mtime_ns IS NULL OR file_name IS NULL
                """
            ).fetchall()
            for (fp,) in rows:
                try:
                    canonical = self._canonical_path(str(fp))
                    if not os.path.isfile(canonical):
                        continue
                    st = os.stat(canonical)
                    self._conn.execute(
                        """
                        UPDATE semantic_index
                        SET file_name = ?, file_signature = ?, mtime_ns = ?, file_path = ?
                        WHERE file_path = ?
                        """,
                        (
                            os.path.basename(canonical),
                            self._file_signature_from_stat(canonical, st),
                            self._mtime_ns_from_stat(st),
                            canonical,
                            fp,
                        ),
                    )
                except Exception:
                    continue
            self._conn.commit()
        except Exception:
            pass

    @staticmethod
    def _ratio_to_float(v) -> Optional[float]:
        try:
            # exifread.utils.Ratio supports num/den
            if hasattr(v, "num") and hasattr(v, "den"):
                den = float(v.den) if float(v.den) != 0 else 1.0
                return float(v.num) / den
            return float(v)
        except Exception:
            return None

    def _gps_to_decimal(self, gps_vals, ref: str) -> Optional[float]:
        try:
            vals = getattr(gps_vals, "values", gps_vals)
            if not vals or len(vals) < 3:
                return None
            d = self._ratio_to_float(vals[0])
            m = self._ratio_to_float(vals[1])
            s = self._ratio_to_float(vals[2])
            if d is None or m is None or s is None:
                return None
            dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
            if (ref or "").strip().upper() in ("S", "W"):
                dec = -dec
            return dec
        except Exception:
            return None

    def _extract_exif_brief(self, file_path: str, include_face: bool = False) -> Dict[str, object]:
        result = {
            "capture_time": "",
            "camera_model": "",
            "lens_model": "",
            "iso": 0,
            "width": 0,
            "height": 0,
            "gps_lat": None,
            "gps_lon": None,
            "gps_raw": "",
            "city": "",
            "admin1": "",
            "country": "",
            "country_code": "",
            "face_count": None,
        }
        try:
            with open(file_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
            result["capture_time"] = self._tag_text(
                tags,
                "EXIF DateTimeOriginal",
                "EXIF DateTimeDigitized",
                "Image DateTime",
                "EXIF DateTime",
            )
            make = self._tag_text(tags, "Image Make")
            model = self._tag_text(tags, "Image Model", "EXIF BodySerialNumber")
            result["camera_model"] = self._clean_join(make, model)
            result["lens_model"] = self._tag_text(
                tags,
                "EXIF LensModel",
                "EXIF LensMake",
                "MakerNote LensType",
                "MakerNote Lens",
                "Image LensModel",
                "Composite LensID",
            )
            result["iso"] = self._tag_int(
                tags,
                "EXIF ISOSpeedRatings",
                "EXIF PhotographicSensitivity",
                "EXIF RecommendedExposureIndex",
                "EXIF ISO",
                "MakerNote ISO",
            )
            result["width"] = self._tag_int(
                tags,
                "EXIF ExifImageWidth",
                "Image ImageWidth",
                "Image Width",
                "EXIF PixelXDimension",
            )
            result["height"] = self._tag_int(
                tags,
                "EXIF ExifImageLength",
                "Image ImageLength",
                "Image Height",
                "Image Length",
                "EXIF PixelYDimension",
            )
            if not result["width"] or not result["height"]:
                w, h = self._pil_dimensions(file_path)
                result["width"] = int(result["width"] or w)
                result["height"] = int(result["height"] or h)
            lat = tags.get("GPS GPSLatitude")
            lon = tags.get("GPS GPSLongitude")
            lat_ref = str(tags.get("GPS GPSLatitudeRef", "")).strip()
            lon_ref = str(tags.get("GPS GPSLongitudeRef", "")).strip()
            result["gps_lat"] = self._gps_to_decimal(lat, lat_ref) if lat else None
            result["gps_lon"] = self._gps_to_decimal(lon, lon_ref) if lon else None
            if lat and lon:
                result["gps_raw"] = f"{lat_ref} {lat} | {lon_ref} {lon}"
            if result["gps_lat"] is not None and result["gps_lon"] is not None:
                geo = self._ensure_reverse_geocoder()
                if geo:
                    try:
                        recs = geo.search(
                            [(float(result["gps_lat"]), float(result["gps_lon"]))],
                            mode=1,
                        )
                        if recs:
                            rec = recs[0] or {}
                            result["city"] = str(rec.get("name", "") or "")
                            result["admin1"] = str(rec.get("admin1", "") or "")
                            cc = str(rec.get("cc", "") or "").upper()
                            result["country_code"] = cc
                            result["country"] = self._country_name_from_code(cc)
                    except Exception:
                        pass
            if include_face:
                result["face_count"] = self._detect_face_count(file_path)
        except Exception:
            pass
        return result

    @staticmethod
    def _detect_face_count(file_path: str) -> int:
        if sys.platform != "darwin":
            return 0
        try:
            import Foundation
            import Vision
        except Exception:
            return 0

        def _run_vision(path: str) -> Optional[int]:
            try:
                url = Foundation.NSURL.fileURLWithPath_(path)
                request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
                handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
                ok, err = handler.performRequests_error_([request], None)
                if not ok or err is not None:
                    return None
                return len(request.results() or [])
            except Exception:
                return None

        tmp_path = ""
        try:
            im = _load_index_source_image(file_path, max_size=1280)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            im.save(tmp_path, "JPEG", quality=90)
            fallback = _run_vision(tmp_path)
            return int(fallback or 0)
        except Exception:
            return 0
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _lookup_index_rows(self, file_path: str, st: Optional[os.stat_result] = None) -> List[sqlite3.Row]:
        canonical = self._canonical_path(file_path)
        if st is None:
            st = os.stat(canonical)
        signature = self._file_signature_from_stat(canonical, st)
        aliases = self._path_aliases(canonical)
        placeholders = ",".join(["?"] * len(aliases))
        self._conn.row_factory = sqlite3.Row
        return self._conn.execute(
            f"""
            SELECT file_path, file_name, file_signature, file_size, file_mtime, mtime_ns, model_name, face_count
            FROM semantic_index
            WHERE (file_signature = ? AND model_name = ?)
               OR file_path IN ({placeholders})
            """,
            [signature, self.model_name, *aliases],
        ).fetchall()

    def _row_matches_file(self, row: sqlite3.Row, st: os.stat_result) -> bool:
        if str(row["model_name"]) != self.model_name:
            return False
        if int(row["file_size"]) != int(st.st_size):
            return False
        row_mtime_ns = row["mtime_ns"] if "mtime_ns" in row.keys() else None
        if row_mtime_ns is not None:
            try:
                return int(row_mtime_ns) == self._mtime_ns_from_stat(st)
            except Exception:
                pass
        return self._mtime_matches(float(row["file_mtime"]), st)

    def _needs_reindex(self, file_path: str, st: os.stat_result) -> bool:
        rows = self._lookup_index_rows(file_path, st)
        if not rows:
            return True
        for row in rows:
            if self._row_matches_file(row, st):
                return False
        return True

    def _encode_image(self, file_path: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-coreml"):
            backend = self._mobileclip_backend or resolve_mobileclip_coreml_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP Core ML backend unavailable: {err}")
            return backend.encode_image(file_path)
        model = self._ensure_model()
        try:
            im = _load_index_source_image(file_path, max_size=1024)
            emb = model.encode(im, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decode image for semantic index: {os.path.basename(file_path)}"
            ) from exc

    def _encode_text(self, text: str) -> np.ndarray:
        if self.model_name.startswith("mobileclip-coreml"):
            backend = self._mobileclip_backend or resolve_mobileclip_coreml_backend()
            err = backend.availability_error()
            if err:
                raise RuntimeError(f"MobileCLIP Core ML backend unavailable: {err}")
            return backend.encode_text(text)
        model = self._ensure_model()
        return np.asarray(model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def build_index(self, file_paths: Sequence[str], progress_callback: ProgressCallback = None) -> Dict[str, int]:
        total = len(file_paths)
        indexed = 0
        skipped = 0
        failed = 0
        for i, fp in enumerate(file_paths, start=1):
            if progress_callback:
                progress_callback(i, total, fp)
            if not fp or not os.path.isfile(fp):
                failed += 1
                continue
            try:
                canonical_fp = self._canonical_path(fp)
                st = os.stat(canonical_fp)
                if not self._needs_reindex(canonical_fp, st):
                    skipped += 1
                    continue
                file_name = os.path.basename(canonical_fp)
                file_signature = self._file_signature_from_stat(canonical_fp, st)
                mtime_ns = self._mtime_ns_from_stat(st)
                vec = self._encode_image(canonical_fp)
                # Face presence is metadata: compute it once during indexing so
                # face/no-face filters do not need to scan images repeatedly.
                meta = self._extract_exif_brief(canonical_fp, include_face=True)
                self._conn.execute(
                    """
                    INSERT INTO semantic_index (
                        file_path, file_name, file_signature, file_size, file_mtime, mtime_ns,
                        model_name, dim, embedding,
                        capture_time, camera_model, lens_model, iso, width, height,
                        gps_lat, gps_lon, gps_raw, city, admin1, country, country_code, face_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(file_path) DO UPDATE SET
                        file_name=excluded.file_name,
                        file_signature=excluded.file_signature,
                        file_size=excluded.file_size,
                        file_mtime=excluded.file_mtime,
                        mtime_ns=excluded.mtime_ns,
                        model_name=excluded.model_name,
                        dim=excluded.dim,
                        embedding=excluded.embedding,
                        capture_time=excluded.capture_time,
                        camera_model=excluded.camera_model,
                        lens_model=excluded.lens_model,
                        iso=excluded.iso,
                        width=excluded.width,
                        height=excluded.height,
                        gps_lat=excluded.gps_lat,
                        gps_lon=excluded.gps_lon,
                        gps_raw=excluded.gps_raw,
                        city=excluded.city,
                        admin1=excluded.admin1,
                        country=excluded.country,
                        country_code=excluded.country_code,
                        face_count=excluded.face_count,
                        updated_at=excluded.updated_at
                    """,
                    (
                        canonical_fp,
                        file_name,
                        file_signature,
                        int(st.st_size),
                        float(st.st_mtime),
                        mtime_ns,
                        self.model_name,
                        int(vec.size),
                        self._to_blob(vec),
                        str(meta["capture_time"]),
                        str(meta["camera_model"]),
                        str(meta["lens_model"]),
                        int(meta["iso"]),
                        int(meta["width"]),
                        int(meta["height"]),
                        float(meta["gps_lat"]) if meta["gps_lat"] is not None else None,
                        float(meta["gps_lon"]) if meta["gps_lon"] is not None else None,
                        str(meta["gps_raw"]),
                        str(meta["city"]),
                        str(meta["admin1"]),
                        str(meta["country"]),
                        str(meta["country_code"]),
                        int(meta["face_count"]) if meta.get("face_count") is not None else None,
                        float(time.time()),
                    ),
                )
                indexed += 1
                # Persist each successful file so interrupted sessions can resume accurately.
                self._conn.commit()
            except Exception:
                failed += 1
        self._conn.commit()
        return {"indexed": indexed, "skipped": skipped, "failed": failed, "total": total}

    def get_index_coverage(self, file_paths: Sequence[str]) -> Dict[str, int]:
        """
        Return index coverage for the given file set.

        A file is counted as indexed if an entry exists with matching size/mtime/model.
        """
        valid_paths = [self._canonical_path(p) for p in file_paths if p and os.path.isfile(p)]
        total = len(valid_paths)
        if total == 0:
            return {"total": 0, "indexed": 0, "missing": 0, "ready": 1}

        indexed = 0
        for fp in valid_paths:
            try:
                st = os.stat(fp)
                rows = self._lookup_index_rows(fp, st)
                if any(self._row_matches_file(r, st) for r in rows):
                    indexed += 1
            except Exception:
                continue

        missing = max(0, total - indexed)
        return {
            "total": total,
            "indexed": indexed,
            "missing": missing,
            "ready": 1 if missing == 0 else 0,
        }

    def get_pending_paths(self, file_paths: Sequence[str]) -> List[str]:
        """
        Return only files that are missing or stale and need reindexing.
        """
        pending: List[str] = []
        for fp in file_paths:
            if not fp or not os.path.isfile(fp):
                continue
            try:
                canonical_fp = self._canonical_path(fp)
                st = os.stat(canonical_fp)
                if self._needs_reindex(canonical_fp, st):
                    pending.append(canonical_fp)
            except Exception:
                pending.append(self._canonical_path(fp))
        return pending

    def _fetch_rows_for_paths(self, paths: Sequence[str]) -> List[sqlite3.Row]:
        if not paths:
            return []
        out = []
        self._conn.row_factory = sqlite3.Row
        seen = set()
        for fp in paths:
            if not fp or not os.path.isfile(fp):
                continue
            try:
                canonical = self._canonical_path(fp)
                st = os.stat(canonical)
                signature = self._file_signature_from_stat(canonical, st)
                aliases = self._path_aliases(canonical)
                placeholders = ",".join(["?"] * len(aliases))
                rows = self._conn.execute(
                    f"""
                    SELECT file_path, file_name, file_signature, dim, embedding, capture_time, camera_model, lens_model, iso
                         , gps_lat, gps_lon, width, height, city, admin1, country, country_code, face_count
                         , file_size, file_mtime, mtime_ns, model_name
                    FROM semantic_index
                    WHERE (file_signature = ? AND model_name = ?)
                       OR file_path IN ({placeholders})
                    """,
                    [signature, self.model_name, *aliases],
                ).fetchall()
                for row in rows:
                    if not self._row_matches_file(row, st):
                        continue
                    key = str(row["file_signature"] or row["file_path"])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(row)
                    break
            except Exception:
                continue
        return out

    @staticmethod
    def _parse_capture_year(capture_time: str) -> int:
        # EXIF style: "YYYY:MM:DD HH:MM:SS"
        try:
            return int((capture_time or "")[:4])
        except Exception:
            return 0

    @staticmethod
    def _parse_capture_month(capture_time: str) -> int:
        try:
            return int((capture_time or "")[5:7])
        except Exception:
            return 0

    @staticmethod
    def _row_value(row, key: str, default=""):
        try:
            return row[key]
        except Exception:
            try:
                return row.get(key, default)
            except Exception:
                return default

    @staticmethod
    def _normalize_date_value(value: str) -> str:
        return (value or "").strip().replace("-", ":").replace("/", ":")

    @classmethod
    def _capture_time_matches_date(cls, capture_time: str, value: str) -> bool:
        normalized = cls._normalize_date_value(value)
        if not normalized:
            return False
        capture = str(capture_time or "")
        return capture.startswith(normalized)

    @staticmethod
    def _contains_loose(haystack: str, needle: str) -> bool:
        h = str(haystack or "").lower()
        n = str(needle or "").strip().lower()
        if not n:
            return False
        variants = {n, n.replace("_", " "), n.replace(" ", "_")}
        return any(v and v in h for v in variants)

    def _apply_filters(
        self, rows: Sequence[sqlite3.Row], query_text: str
    ) -> tuple[List[sqlite3.Row], str]:
        """
        Parse a mixed query and filter rows.

        Supported filter tokens:
        - camera:<text>
        - lens:<text>
        - city:<text>
        - admin:<text>
        - country:<text>          (country name or code)
        - date:<prefix>          (e.g. date:2026:05, date:2026-05, date:2026-05-01)
        - filename:<text> / file:<text> / name:<text>
        - iso<800 / iso<=800 / iso=400 / iso>=200 / iso>100
        - year=2026 / year>=2024 / year<2020
        - month=5 / month>=6 / month<3
        - width>=3000 / height<2000
        - has:gps / no:gps
        - has:face / no:face
        """
        raw = (query_text or "").strip()
        if not raw:
            return list(rows), ""

        # Normalize numeric filters so both styles work:
        # - iso<800
        # - iso < 800
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)\b",
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}",
            raw,
            flags=re.I,
        )
        raw = re.sub(
            r"\b(iso|year|month|width|height)\s*:\s*(\d+)\b",
            lambda m: f"{m.group(1)}={m.group(2)}",
            raw,
            flags=re.I,
        )
        raw = self._normalize_loose_metadata_query(raw)

        parts = [p for p in raw.split() if p.strip()]
        semantic_terms: List[str] = []
        filtered = list(rows)

        num_pat = re.compile(r"^(iso|year|month|width|height)\s*(<=|>=|=|<|>)\s*(\d+)$", re.I)
        date_like_pat = re.compile(r"^\d{4}(?:[:-]\d{1,2}){0,2}$")

        for token in parts:
            t = token.strip()
            low = t.lower()
            matched = False

            if low.startswith("camera:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["camera_model"] or ""), needle)]
                continue

            if low.startswith("lens:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["lens_model"] or ""), needle)]
                continue

            if low.startswith("date:"):
                matched = True
                pref = t.split(":", 1)[1].strip()
                if pref:
                    filtered = [
                        r
                        for r in filtered
                        if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), pref)
                    ]
                continue

            if low.startswith(("filename:", "file:", "name:")):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r
                        for r in filtered
                        if self._contains_loose(
                            str(self._row_value(r, "file_name") or os.path.basename(str(self._row_value(r, "file_path") or ""))),
                            needle,
                        )
                    ]
                continue

            if low.startswith("city:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["city"] or ""), needle)]
                continue

            if low.startswith("admin:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [r for r in filtered if self._contains_loose(str(r["admin1"] or ""), needle)]
                continue

            if low.startswith("country:"):
                matched = True
                needle = t.split(":", 1)[1].strip().lower()
                if needle:
                    filtered = [
                        r
                        for r in filtered
                        if (
                            self._contains_loose(str(r["country"] or ""), needle)
                            or self._contains_loose(str(r["country_code"] or ""), needle)
                        )
                    ]
                continue

            if low == "has:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is not None and r["gps_lon"] is not None
                ]
                continue

            if low == "no:gps":
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if r["gps_lat"] is None or r["gps_lon"] is None
                ]
                continue

            if low in ("has:face", "has:faces", "face", "faces"):
                matched = True
                filtered = [r for r in filtered if int(r["face_count"] or 0) > 0]
                continue

            if low in ("no:face", "no:faces"):
                matched = True
                filtered = [r for r in filtered if int(r["face_count"] or 0) <= 0]
                continue

            m = num_pat.match(low)
            if m:
                matched = True
                key, op, val_str = m.group(1), m.group(2), m.group(3)
                val = int(val_str)

                def _ok(x: int) -> bool:
                    if op == "<":
                        return x < val
                    if op == "<=":
                        return x <= val
                    if op == "=":
                        return x == val
                    if op == ">=":
                        return x >= val
                    return x > val

                if key == "iso":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "iso", 0) or 0))]
                elif key == "year":
                    filtered = [r for r in filtered if _ok(self._parse_capture_year(str(self._row_value(r, "capture_time") or "")))]
                elif key == "month":
                    filtered = [r for r in filtered if _ok(self._parse_capture_month(str(self._row_value(r, "capture_time") or "")))]
                elif key == "width":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "width", 0) or 0))]
                elif key == "height":
                    filtered = [r for r in filtered if _ok(int(self._row_value(r, "height", 0) or 0))]
                continue

            if date_like_pat.match(low):
                matched = True
                filtered = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), low)
                ]
                continue

            if not matched:
                semantic_terms.append(t)

        filtered, semantic_terms = self._auto_match_metadata_keywords(filtered, semantic_terms)
        metadata_stopwords = {
            "a",
            "an",
            "the",
            "of",
            "with",
            "for",
            "photo",
            "photos",
            "image",
            "images",
            "picture",
            "pictures",
            "shot",
            "shots",
            "taken",
            "exif",
            "metadata",
        }
        semantic_terms = [
            t for t in semantic_terms if (t or "").strip().lower() not in metadata_stopwords
        ]
        return filtered, " ".join(semantic_terms).strip()

    def _normalize_loose_metadata_query(self, raw: str) -> str:
        """
        Accept light natural-language metadata filters without an LLM.
        Examples:
        - iso under 800 -> iso<800
        - in tokyo -> city:tokyo
        - from japan -> country:japan
        """
        text = f" {raw.strip()} "

        text = re.sub(
            r"\b(iso|year|month|width|height)\s+(\d+)\b",
            r"\1=\2",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(camera|lens|city|admin|country|date|filename|file|name)\s*=\s*([^\s]+)\b",
            r"\1:\2",
            text,
            flags=re.I,
        )

        numeric_phrases = [
            (r"\b(iso|year|month|width|height)\s+(?:under|below|less\s+than|smaller\s+than)\s+(\d+)\b", r"\1<\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+most|up\s+to|no\s+more\s+than|less\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1<=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:over|above|more\s+than|greater\s+than|larger\s+than)\s+(\d+)\b", r"\1>\2"),
            (r"\b(iso|year|month|width|height)\s+(?:at\s+least|no\s+less\s+than|greater\s+than\s+or\s+equal\s+to)\s+(\d+)\b", r"\1>=\2"),
            (r"\b(iso|year|month|width|height)\s+(?:is|equals?|equal\s+to)\s+(\d+)\b", r"\1=\2"),
        ]
        for pattern, replacement in numeric_phrases:
            text = re.sub(pattern, replacement, text, flags=re.I)

        text = re.sub(
            r"\b(?:in|at|near)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            lambda m: f"city:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:from|country)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,2})\b",
            lambda m: f"country:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bcamera\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"camera:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\blens\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,3})\b",
            lambda m: f"lens:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:filename|file\s+name|file|name)\s+([A-Za-z0-9][\w\-.]*(?:\s+[A-Za-z0-9][\w\-.]*){0,2})\b",
            lambda m: f"filename:{'_'.join(m.group(1).split())}",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\bdate\s+(\d{4}(?:[-:/]\d{1,2}){0,2})\b",
            r"date:\1",
            text,
            flags=re.I,
        )
        return text.strip()

    def _auto_match_metadata_keywords(
        self, rows: Sequence[sqlite3.Row], semantic_terms: Sequence[str]
    ) -> tuple[List[sqlite3.Row], List[str]]:
        """
        If a free-text token appears in metadata fields, use it as a metadata filter
        and remove it from the semantic text query.
        """
        filtered = list(rows)
        remaining: List[str] = []
        metadata_fields = ("city", "admin1", "country", "country_code", "camera_model", "lens_model", "file_name")

        for term in semantic_terms:
            needle = (term or "").strip().lower()
            if len(needle) < 3:
                remaining.append(term)
                continue
            if re.fullmatch(r"\d{4}(?:[:-]\d{1,2}){0,2}", needle):
                matched_rows = [
                    r
                    for r in filtered
                    if self._capture_time_matches_date(str(self._row_value(r, "capture_time") or ""), needle)
                ]
                if matched_rows:
                    filtered = matched_rows
                    continue
            matched_rows = [
                r
                for r in filtered
                if any(self._contains_loose(str(self._row_value(r, field) or ""), needle) for field in metadata_fields)
            ]
            # Any metadata hit satisfies this token. Requiring the match to narrow
            # the set breaks folders where every image shares the same camera/city.
            if matched_rows:
                filtered = matched_rows
            else:
                remaining.append(term)

        return filtered, remaining

    def search_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 200, min_score: float = 0.0
    ) -> List[SearchHit]:
        raw_query = (query or "").strip()
        if not raw_query:
            return []
        canonical_to_original = {}
        signature_to_original = {}
        for p in candidate_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                canonical = self._canonical_path(p)
                st = os.stat(canonical)
                canonical_to_original[canonical] = p
                signature_to_original[self._file_signature_from_stat(canonical, st)] = p
            except Exception:
                canonical_to_original[self._canonical_path(p)] = p
        rows = self._fetch_rows_for_paths(candidate_paths)
        if not rows:
            return []
        rows, semantic_query = self._apply_filters(rows, raw_query)
        if not rows:
            return []

        # Filter-only query: return deterministic list without semantic ranking.
        if not semantic_query:
            hits = [
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=1.0,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=int(r["iso"] or 0),
                    gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                    gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                    city=str(r["city"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=int(r["face_count"] or 0),
                )
                for r in rows
            ]
            hits.sort(key=lambda h: h.capture_time, reverse=True)
            return hits[: max(1, int(top_k))]

        query_vec = self._encode_text(semantic_query)

        scores: List[SearchHit] = []
        for r in rows:
            vec = self._from_blob(r["embedding"], int(r["dim"]))
            if vec.size == 0:
                continue
            score = float(np.dot(query_vec, vec))
            # Semantic query should filter out clearly non-matching images.
            # Keep strict filtering (not just ranking) by removing sub-threshold scores.
            if score <= float(min_score):
                continue
            scores.append(
                SearchHit(
                    file_path=str(
                        signature_to_original.get(str(r["file_signature"] or ""))
                        or canonical_to_original.get(self._canonical_path(str(r["file_path"])))
                        or str(r["file_path"])
                    ),
                    score=score,
                    file_name=str(r["file_name"] or os.path.basename(str(r["file_path"]))),
                    capture_time=str(r["capture_time"] or ""),
                    camera_model=str(r["camera_model"] or ""),
                    lens_model=str(r["lens_model"] or ""),
                    iso=int(r["iso"] or 0),
                    gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                    gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                    city=str(r["city"] or ""),
                    admin1=str(r["admin1"] or ""),
                    country=str(r["country"] or ""),
                    country_code=str(r["country_code"] or ""),
                    face_count=int(r["face_count"] or 0),
                )
            )
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[: max(1, int(top_k))]

    def search_metadata_text(
        self, query: str, candidate_paths: Sequence[str], top_k: int = 500
    ) -> tuple[List[SearchHit], str]:
        """
        Metadata/EXIF-only search that works without any embedding backend.

        Returns (hits, remaining_semantic_query). If remaining_semantic_query is empty,
        the query was fully satisfied by metadata filters/auto-matches.
        """
        raw_query = (query or "").strip()
        if not raw_query:
            return [], ""

        needs_face = self._query_needs_face_detection(raw_query)
        rows = self._metadata_rows_for_search(candidate_paths, needs_face=needs_face)

        filtered, semantic_query = self._apply_filters(rows, raw_query)
        hits = [
            SearchHit(
                file_path=str(r["file_path"]),
                score=1.0,
                file_name=str(r.get("file_name") or os.path.basename(str(r["file_path"]))),
                capture_time=str(r["capture_time"] or ""),
                camera_model=str(r["camera_model"] or ""),
                lens_model=str(r["lens_model"] or ""),
                iso=int(r["iso"] or 0),
                gps_lat=float(r["gps_lat"]) if r["gps_lat"] is not None else None,
                gps_lon=float(r["gps_lon"]) if r["gps_lon"] is not None else None,
                city=str(r["city"] or ""),
                admin1=str(r["admin1"] or ""),
                country=str(r["country"] or ""),
                country_code=str(r["country_code"] or ""),
                face_count=int(r["face_count"] or 0),
            )
            for r in filtered
        ]
        hits.sort(key=lambda h: h.capture_time, reverse=True)
        return hits[: max(1, int(top_k))], semantic_query

    def _metadata_rows_for_search(
        self, candidate_paths: Sequence[str], needs_face: bool = False
    ) -> List[Dict[str, object]]:
        original_by_canonical: Dict[str, str] = {}
        original_by_signature: Dict[str, str] = {}
        valid_paths: List[str] = []
        for p in candidate_paths:
            if not p or not os.path.isfile(p):
                continue
            try:
                canonical = self._canonical_path(p)
                st = os.stat(canonical)
                valid_paths.append(p)
                original_by_canonical[canonical] = p
                original_by_signature[self._file_signature_from_stat(canonical, st)] = p
            except Exception:
                canonical = self._canonical_path(p)
                valid_paths.append(p)
                original_by_canonical[canonical] = p

        rows: List[Dict[str, object]] = []
        seen: set[str] = set()
        for row in self._fetch_rows_for_paths(valid_paths):
            signature = str(row["file_signature"] or "")
            canonical = self._canonical_path(str(row["file_path"]))
            original = original_by_signature.get(signature) or original_by_canonical.get(canonical) or canonical
            face_count = row["face_count"] if "face_count" in row.keys() else None
            if needs_face and face_count is None:
                face_count = self._detect_face_count(original)
                self._store_face_count(original, int(face_count or 0))
            seen.add(self._canonical_path(original))
            rows.append(
                {
                    "file_path": original,
                    "file_name": str(row["file_name"] or os.path.basename(original)),
                    "capture_time": str(row["capture_time"] or ""),
                    "camera_model": str(row["camera_model"] or ""),
                    "lens_model": str(row["lens_model"] or ""),
                    "iso": int(row["iso"] or 0),
                    "width": int(row["width"] or 0),
                    "height": int(row["height"] or 0),
                    "gps_lat": row["gps_lat"],
                    "gps_lon": row["gps_lon"],
                    "city": str(row["city"] or ""),
                    "admin1": str(row["admin1"] or ""),
                    "country": str(row["country"] or ""),
                    "country_code": str(row["country_code"] or ""),
                    "face_count": int(face_count or 0),
                }
            )

        for fp in valid_paths:
            canonical = self._canonical_path(fp)
            if canonical in seen:
                continue
            if not fp or not os.path.isfile(fp):
                continue
            try:
                meta = self._extract_exif_brief(canonical, include_face=needs_face)
                rows.append(
                    {
                        "file_path": fp,
                        "file_name": os.path.basename(fp),
                        "capture_time": str(meta.get("capture_time", "")),
                        "camera_model": str(meta.get("camera_model", "")),
                        "lens_model": str(meta.get("lens_model", "")),
                        "iso": int(meta.get("iso", 0) or 0),
                        "width": int(meta.get("width", 0) or 0),
                        "height": int(meta.get("height", 0) or 0),
                        "gps_lat": meta.get("gps_lat"),
                        "gps_lon": meta.get("gps_lon"),
                        "city": str(meta.get("city", "")),
                        "admin1": str(meta.get("admin1", "")),
                        "country": str(meta.get("country", "")),
                        "country_code": str(meta.get("country_code", "")),
                        "face_count": int(meta.get("face_count") or 0),
                    }
                )
            except Exception:
                continue
        return rows

    def _store_face_count(self, file_path: str, face_count: int) -> None:
        try:
            canonical = self._canonical_path(file_path)
            st = os.stat(canonical)
            signature = self._file_signature_from_stat(canonical, st)
            aliases = self._path_aliases(canonical)
            placeholders = ",".join(["?"] * len(aliases))
            self._conn.execute(
                f"""
                UPDATE semantic_index
                SET face_count = ?
                WHERE (file_signature = ? AND model_name = ?)
                   OR file_path IN ({placeholders})
                """,
                [int(face_count), signature, self.model_name, *aliases],
            )
            self._conn.commit()
        except Exception:
            pass

    @staticmethod
    def _query_needs_face_detection(query: str) -> bool:
        return bool(re.search(r"\b(?:has:faces?|no:faces?|faces?)\b", query or "", flags=re.I))

