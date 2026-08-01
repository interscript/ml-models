"""Rababa Arabic data module.

Source: Tashkeela++ (cleaned, deduplicated). The data module owns:
- Character-level encoding (Arabic + harakat)
- Train/val splitting
- Vocabulary (input = bare Arabic chars; output = chars + harakat)

The encoding scheme mirrors the interscript-ts runtime so models
trained here can run inference identically in the browser.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from framework.data import DataModule, DataSplit, Example, PreparedData
from framework.registry import register_data_module

BASIC_HARAKAT = "ًٌٍَُِّْ"
SHADDA = "ّ"
SUKUN = "ْ"
TATWEEL = "ـ"

INPUT_ALPHABET = (
    "ءآأؤإئابةت"
    "ثجحخدذرزسش"
    "صضطظعغفقكل"
    "منهوىي "
)

OUTPUT_ALPHABET = INPUT_ALPHABET + BASIC_HARAKAT

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


def _build_vocab(alphabet: str) -> dict[str, int]:
    special = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID}
    chars = {c: i + len(special) for i, c in enumerate(alphabet)}
    return {**special, **chars}


INPUT_VOCAB = _build_vocab(INPUT_ALPHABET)
OUTPUT_VOCAB = _build_vocab(OUTPUT_ALPHABET)


def strip_diacritics(text: str) -> str:
    """Remove all harakat from text. Used to derive the bare source."""
    return "".join(c for c in text if c not in BASIC_HARAKAT and c != TATWEEL)


def clean_arabic(text: str) -> str:
    """Collapse whitespace + strip non-Arabic characters."""
    out = []
    for c in text:
        if c in INPUT_ALPHABET or c in BASIC_HARAKAT:
            out.append(c)
        elif c.isspace():
            out.append(" ")
    cleaned = "".join(out)
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned.strip()


@register_data_module("rababa_arabic_data")
class RababaArabicData(DataModule):
    """Concrete data module for rababa_arabic."""

    def prepare_data(self) -> PreparedData:
        if self._prepared is not None:
            return self._prepared
        tsv_path = self.data_root / "raw" / f"{self.config.source}.tsv"
        txt_path = self.data_root / "raw" / f"{self.config.source}.txt"
        if tsv_path.is_file():
            examples = self._read_examples_tsv(tsv_path)
        else:
            examples = self._read_examples(txt_path)
        if not examples:
            examples = self._fallback_examples()
        random.Random(42).shuffle(examples)
        val_n = max(1, min(len(examples) // 10, self.config.max_val_samples or 1000))
        val_pairs = examples[:val_n]
        train_pairs = examples[val_n:]
        if self.config.max_train_samples:
            train_pairs = train_pairs[: self.config.max_train_samples]
        train_split = self._encode_split(train_pairs)
        val_split = self._encode_split(val_pairs)
        all_examples = list(train_split) + list(val_split)
        prepared = PreparedData(
            train=train_split,
            val=val_split,
            vocab_size=len(OUTPUT_VOCAB),
            max_seq_len=max(len(e.source) for e in all_examples) + 1,
        )
        self._prepared = prepared
        return prepared

    def encode_source(self, text: str) -> tuple[int, ...]:
        cleaned = clean_arabic(text)
        return tuple(INPUT_VOCAB.get(c, PAD_ID) for c in cleaned)

    def decode_target(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in OUTPUT_VOCAB.items() if k not in {"<pad>", "<sos>", "<eos>"}}
        return "".join(inv.get(int(i), "") for i in ids if int(i) not in {PAD_ID, SOS_ID, EOS_ID})

    def _read_examples(self, path: Path) -> list[tuple[str, str]]:
        if not path.is_file():
            return []
        out: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            diacritized = clean_arabic(line)
            if not diacritized:
                continue
            bare = strip_diacritics(diacritized)
            if not bare.strip():
                continue
            out.append((bare, diacritized))
        return out

    def _read_examples_tsv(self, path: Path) -> list[tuple[str, str]]:
        """Read paired ``bare<TAB>diacritized`` rows from the fetcher.

        Preferred over the single-text path because the upstream dataset
        carries both columns already paired — no stripping needed, and
        the dataset's canonical letter forms are preserved.
        """
        if not path.is_file():
            return []
        out: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            bare = clean_arabic(parts[0])
            diacritized = clean_arabic(parts[1])
            if not bare or not diacritized:
                continue
            bare_dediac = strip_diacritics(bare)
            if not bare_dediac.strip():
                continue
            out.append((bare_dediac, diacritized))
        return out

    def _fallback_examples(self) -> list[tuple[str, str]]:
        """Tiny built-in corpus so unit tests don't need real data."""
        return [
            ("كتب", "كَتَبَ"),
            ("علم", "عَلِمَ"),
            ("قرا", "قَرَأَ"),
            ("ذهب", "ذَهَبَ"),
            ("سمع", "سَمِعَ"),
            ("فهم", "فَهِمَ"),
            ("كتب كتابا", "كَتَبَ كِتَاباً"),
            ("قرا الكتاب", "قَرَأَ الْكِتَابَ"),
        ]

    def _encode_split(
        self,
        pairs: list[tuple[str, str]],
    ) -> DataSplit:
        examples: list[Example] = []
        for source, target in pairs:
            input_ids = tuple(INPUT_VOCAB.get(c, PAD_ID) for c in source)
            target_ids = (SOS_ID,) + tuple(
                OUTPUT_VOCAB.get(c, PAD_ID) for c in target
            ) + (EOS_ID,)
            examples.append(
                Example(
                    source=source,
                    target=target,
                    input_ids=input_ids,
                    target_ids=target_ids,
                )
            )
        return DataSplit(tuple(examples))


def vocab_stats() -> dict[str, int]:
    """Expose vocab sizes for tests + docs."""
    return {"input": len(INPUT_VOCAB), "output": len(OUTPUT_VOCAB)}
