"""CLIP byte-pair tokenizer, in pure Python.

The local generative provider needs to turn an instruction into the token
ids the text encoder expects. HuggingFace ``transformers`` would do this in a
line, but pulling it in would add a large dependency tree to an app whose
entire ML stack is ONNX Runtime and numpy. The tokenizer is a few hundred
lines of well-specified BPE, so it is implemented here instead.

Reads the same ``vocab.json`` and ``merges.txt`` that ship beside the ONNX
graphs, so it stays in step with whatever text encoder is downloaded rather
than hard-coding a vocabulary.
"""

from __future__ import annotations

import functools
import gzip
import html
import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


@functools.lru_cache()
def _bytes_to_unicode() -> Dict[int, str]:
    """Reversible byte<->unicode map, as used by GPT-2 and CLIP.

    BPE works on unicode strings, but the input is bytes. Mapping the
    printable ASCII range to itself and the rest to unused codepoints keeps
    every byte representable without ever producing whitespace or control
    characters that would break the merge rules.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word: Tuple[str, ...]):
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def _basic_clean(text: str) -> str:
    text = html.unescape(html.unescape(text))
    return unicodedata.normalize("NFC", text).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class CLIPBPETokenizer:
    """CLIP's 49408-token BPE tokenizer.

    Only encoding is implemented -- the pipeline never needs to turn ids back
    into text -- plus ``decode`` for diagnostics.
    """

    PAT = re.compile(
        r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
        r"""[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""".replace(r"\p{L}", "a-zA-Z")
        .replace(r"\p{N}", "0-9"),
        re.IGNORECASE,
    )

    def __init__(self, vocab_path: str, merges_path: str, context_length: int = 77):
        self.context_length = int(context_length)

        with open(vocab_path, "r", encoding="utf-8") as fh:
            self.encoder: Dict[str, int] = json.load(fh)
        self.decoder = {v: k for k, v in self.encoder.items()}

        merges = _read_merges(merges_path)
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self._cache: Dict[str, str] = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

        # CLIP pads with the end-of-text token rather than a dedicated pad id.
        self.sot = self.encoder.get("<|startoftext|>")
        self.eot = self.encoder.get("<|endoftext|>")
        if self.sot is None or self.eot is None:
            raise ValueError("vocab.json is missing CLIP's start/end tokens")

    # -- BPE ----------------------------------------------------------

    def _bpe(self, token: str) -> str:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
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

        result = " ".join(word)
        self._cache[token] = result
        return result

    # -- public -------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Token ids for ``text``, without start/end markers or padding."""
        ids: List[int] = []
        cleaned = _whitespace_clean(_basic_clean(text)).lower()
        for token in self.PAT.findall(cleaned):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            for piece in self._bpe(token).split(" "):
                found = self.encoder.get(piece)
                if found is not None:
                    ids.append(found)
        return ids

    def tokenize(self, text: str) -> List[int]:
        """Fixed-length id sequence the text encoder expects.

        Start token, the text, end token, then padding with the end token to
        ``context_length``. Over-long text is truncated with the end token
        forced into the final slot -- an unterminated sequence produces
        garbage embeddings rather than a clear error.
        """
        ids = [self.sot] + self.encode(text) + [self.eot]
        if len(ids) > self.context_length:
            ids = ids[: self.context_length]
            ids[-1] = self.eot
        else:
            ids = ids + [self.eot] * (self.context_length - len(ids))
        return ids

    def decode(self, ids) -> str:
        """Round-trip helper for diagnostics."""
        text = "".join(self.decoder.get(int(i), "") for i in ids)
        raw = bytearray(self.byte_decoder.get(c, 0) for c in text)
        return raw.decode("utf-8", errors="replace").replace("</w>", " ")


def _read_merges(path: str) -> List[Tuple[str, str]]:
    """merges.txt, tolerating the gzipped form some exports ship."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            lines = fh.read().split("\n")
    else:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().split("\n")
    # First line is a version banner in every published copy of this file.
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    merges: List[Tuple[str, str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    return merges


_TOKENIZER_CACHE: Dict[str, CLIPBPETokenizer] = {}


def load_tokenizer(vocab_path: str, merges_path: str) -> Optional[CLIPBPETokenizer]:
    """Cached tokenizer for a vocab/merges pair, or None if unreadable."""
    key = os.path.abspath(vocab_path) + "|" + os.path.abspath(merges_path)
    hit = _TOKENIZER_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        tok = CLIPBPETokenizer(vocab_path, merges_path)
    except Exception:
        return None
    _TOKENIZER_CACHE[key] = tok
    return tok
