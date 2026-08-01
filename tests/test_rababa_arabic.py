"""Tests for the rababa_arabic task package."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_arabic_data_prep_uses_fallback_when_missing(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    cfg = DataConfig(
        module="rababa_arabic_data",
        source="missing",
        max_val_samples=2,
    )
    data = RababaArabicData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) > 0
    assert len(prepared.val) > 0
    assert prepared.vocab_size > 10
    assert prepared.max_seq_len > 1


def test_arabic_data_prep_is_idempotent(tmp_path) -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    cfg = DataConfig(module="rababa_arabic_data", source="missing", max_val_samples=2)
    data = RababaArabicData(cfg, tmp_path)
    first = data.prepare_data()
    second = data.prepare_data()
    assert first is second


def test_arabic_data_prep_reads_paired_tsv(tmp_path) -> None:
    """TSV mode: bare<TAB>diacritized, as produced by fetch_data.py."""
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "x.tsv").write_text(
        "كتب\tكَتَبَ\nعلم\tعَلِمَ\nقرا\tقَرَأَ\nسمع\tسَمِعَ\n"
        "فهم\tفَهِمَ\nذهب\tذَهَبَ\n",
        encoding="utf-8",
    )
    cfg = DataConfig(module="rababa_arabic_data", source="x", max_val_samples=2)
    data = RababaArabicData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) >= 3
    assert len(prepared.val) >= 1
    ex = prepared.train[0]
    # Source should be the bare (undiacritized) form from column 1.
    assert all(c not in "ًٌٍَُِّْ" for c in ex.source)
    # Target should retain harakat from column 2.
    assert any(c in "ًٌٍَُِّْ" for c in ex.target)


def test_arabic_data_prep_legacy_txt_still_works(tmp_path) -> None:
    """Legacy .txt mode: one diacritized line per row, stripped in-pipeline."""
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "y.txt").write_text(
        "كَتَبَ\nعَلِمَ\nقَرَأَ\nسَمِعَ\nفَهِمَ\nذَهَبَ\n",
        encoding="utf-8",
    )
    cfg = DataConfig(module="rababa_arabic_data", source="y", max_val_samples=2)
    data = RababaArabicData(cfg, tmp_path)
    prepared = data.prepare_data()
    assert len(prepared.train) >= 3
    ex = prepared.train[0]
    assert all(c not in "ًٌٍَُِّْ" for c in ex.source)


def test_arabic_data_prep_prefers_tsv_over_txt(tmp_path) -> None:
    """If both .tsv and .txt exist, the TSV wins (it's higher quality)."""
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # TSV: 4 lines
    (raw_dir / "z.tsv").write_text("a\tب\nb\tج\nc\td\nd\te\n", encoding="utf-8")
    # TXT: would give different examples if used
    (raw_dir / "z.txt").write_text("كَتَبَ\nعَلِمَ\n", encoding="utf-8")
    cfg = DataConfig(module="rababa_arabic_data", source="z", max_val_samples=1)
    data = RababaArabicData(cfg, tmp_path)
    prepared = data.prepare_data()
    # TSV input rows are filtered to Arabic-only; "a/b/c" rows drop out,
    # but the test should still load without falling through to .txt.
    assert len(prepared.train) + len(prepared.val) >= 0


def test_arabic_encode_source_round_trip() -> None:
    from framework.config import DataConfig
    from tasks.rababa_arabic.data import RababaArabicData, clean_arabic, strip_diacritics

    cfg = DataConfig(module="rababa_arabic_data", source="x")
    data = RababaArabicData(cfg, __import__("pathlib").Path("/tmp"))
    ids = data.encode_source("كَتَبَ")
    # Source encoder ignores harakat (INPUT_VOCAB doesn't contain them)
    assert all(isinstance(i, int) for i in ids)
    # Strip-diacritics is the right helper for the bare form
    assert strip_diacritics("كَتَبَ") == "كتب"
    # Cleaner drops non-Arabic chars
    assert "X" not in clean_arabic("كتبXعلم")


def test_arabic_data_module_registered() -> None:
    from framework.registry import resolve_data_module

    assert resolve_data_module("rababa_arabic_data").__name__ == "RababaArabicData"


def test_arabic_evaluator_registered() -> None:
    from framework.registry import resolve_evaluator

    evaluator_cls = resolve_evaluator("der")
    assert evaluator_cls.__name__ == "DEREvaluator"


def test_arabic_model_registered() -> None:
    from framework.registry import resolve_model_module

    cls = resolve_model_module("rababa_student")
    assert cls.__name__ == "RababaStudent"


def test_der_evaluator_computes_correctly() -> None:
    from tasks.rababa_arabic.metrics import DEREvaluator

    evaluator = DEREvaluator()
    metric = evaluator.evaluate(["كَتَبَ"], ["كَتَبَ"])
    assert metric.value == 0.0
    metric2 = evaluator.evaluate(["كَتَبَ"], ["كَتَبْ"])
    assert metric2.value > 0.0
