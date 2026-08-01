"""Fetch raw corpora from HuggingFace Hub.

Replaces ``scripts/fetch_data.sh``. The shell version required manual
env-var URLs; this one knows the canonical dataset locations and
validates downloads.

For ``rababa_arabic``:
  Default source: ``arbml/tashkeelav2`` — open-access, pre-split, and
  pre-paired (each row carries both the bare and diacritized text).
  No HF_TOKEN, no gating, no acceptance click-through.
  Fallback: ``community-datasets/tashkeela`` — GPLv2, raw book text
  that needs heavier cleaning (handled by the data module).

For ``rababa_hebrew`` and ``secryst_thai_ipa`` the upstream sources are
not yet on HF as datasets — leave the manual env-var path intact in
``fetch_data.sh`` until they are.

Output format
-------------
The fetcher writes one of two file shapes:

  * ``<source>.tsv`` — paired columns ``bare<TAB>diacritized``. Used
    when the upstream dataset provides both columns (arbml/tashkeelav2).
    The data module reads both directly — no stripping needed.
  * ``<source>.txt`` — one diacritized line per row. Used when the
    upstream is unpaired (raw tashkeela). The data module strips
    diacritics itself.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "data" / "raw"

DATASETS = {
    "rababa_arabic": {
        "primary": {
            "repo_id": "arbml/tashkeelav2",
            "repo_type": "dataset",
            "train_files": [
                "data/train-00000-of-00002-74e12fe61707b796.parquet",
                "data/train-00001-of-00002-4f28b1f4b7fd8dd8.parquet",
            ],
            "test_files": ["data/test-00000-of-00001-ffd46a42fa4bfebf.parquet"],
            "bare_column": "text",
            "diacritized_column": "diacratized",
            "out_name": "tashkeela_plus_plus.tsv",
            "note": (
                "Open-access, pre-paired, pre-split. ~116k train + 58k "
                "test rows. Each row provides both bare + diacritized "
                "text — no in-pipeline stripping required."
            ),
        },
        "fallback": {
            "repo_id": "community-datasets/tashkeela",
            "repo_type": "dataset",
            "train_files": None,
            "test_files": [],
            "bare_column": None,
            "diacritized_column": "text",
            "out_name": "tashkeela_plus_plus.txt",
            "note": (
                "Open-access raw corpus (GPLv2). Each row is a full "
                "book; we split on newlines and skip lines >1024 chars. "
                "Bare form is derived by the data module via strip()."
            ),
        },
    },
}


def fetch_task(
    task: str,
    out_dir: Path,
    max_samples: int | None,
    use_fallback: bool,
) -> Path:
    cfg = DATASETS.get(task)
    if cfg is None:
        raise SystemExit(
            f"No fetcher registered for task '{task}'. "
            f"Known: {sorted(DATASETS)}"
        )
    source = cfg["fallback"] if use_fallback else cfg["primary"]

    import importlib.util

    if importlib.util.find_spec("huggingface_hub") is None:
        raise SystemExit(
            "huggingface_hub is required. Install with: "
            "pip install -e '.[publish]'"
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source["out_name"]

    train_files = source["train_files"]
    if train_files is None:
        train_files = _list_repo_files(source["repo_id"], source["repo_type"], token)

    if source["bare_column"] and source["diacritized_column"]:
        count = _stream_paired_tsv(
            files=train_files,
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            token=token,
            bare_column=source["bare_column"],
            diacritized_column=source["diacritized_column"],
            out_path=out_path,
            max_samples=max_samples,
        )
    else:
        count = _stream_single_text(
            files=train_files,
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            token=token,
            text_column=source["diacritized_column"],
            out_path=out_path,
            max_samples=max_samples,
            split_lines=True,
        )

    print(
        f"[{task}] wrote {count:,} lines -> {out_path} "
        f"({out_path.stat().st_size:,} bytes) "
        f"from {source['repo_id']}"
    )
    if count == 0:
        raise SystemExit(
            f"No lines written. Source note: {source.get('note', '')}"
        )
    return out_path


def _list_repo_files(repo_id: str, repo_type: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    return [f for f in files if f.endswith((".parquet", ".json", ".jsonl", ".txt"))]


def _stream_paired_tsv(
    files: list[str],
    repo_id: str,
    repo_type: str,
    token: str | None,
    bare_column: str,
    diacritized_column: str,
    out_path: Path,
    max_samples: int | None,
    max_line_chars: int = 1024,
) -> int:
    """Stream both columns to a TSV: ``bare<TAB>diacritized`` per line."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    written = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    cols = [bare_column, diacritized_column]
    with tmp.open("w", encoding="utf-8") as fp:
        for fpath in files:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=fpath,
                repo_type=repo_type,
                token=token,
            )
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024, columns=cols):
                bare_list = batch.column(bare_column).to_pylist()
                dia_list = batch.column(diacritized_column).to_pylist()
                for bare, dia in zip(bare_list, dia_list, strict=True):
                    if not bare or not dia:
                        continue
                    bare = " ".join(bare.split())
                    dia = " ".join(dia.split())
                    if (
                        not bare
                        or not dia
                        or len(bare) > max_line_chars
                        or len(dia) > max_line_chars
                    ):
                        continue
                    fp.write(bare)
                    fp.write("\t")
                    fp.write(dia)
                    fp.write("\n")
                    written += 1
                    if max_samples is not None and written >= max_samples:
                        tmp.replace(out_path)
                        return written
    tmp.replace(out_path)
    return written


def _stream_single_text(
    files: list[str],
    repo_id: str,
    repo_type: str,
    token: str | None,
    text_column: str,
    out_path: Path,
    max_samples: int | None,
    split_lines: bool = False,
    max_line_chars: int = 1024,
) -> int:
    """Stream one text column to ``out_path`` (one line per row, or per
    embedded line if ``split_lines`` is set for raw book corpora)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    written = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for fpath in files:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=fpath,
                repo_type=repo_type,
                token=token,
            )
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=1024, columns=[text_column]):
                col = batch.column(text_column).to_pylist()
                for blob in col:
                    if not blob:
                        continue
                    chunks = blob.splitlines() if split_lines else [blob]
                    for raw in chunks:
                        line = " ".join(raw.split())
                        if not line or len(line) > max_line_chars:
                            continue
                        fp.write(line)
                        fp.write("\n")
                        written += 1
                        if max_samples is not None and written >= max_samples:
                            tmp.replace(out_path)
                            return written
    tmp.replace(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(DATASETS),
        help="Which task corpus to fetch.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of lines written (dev/CI mode).",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help=(
            "Use the open-access fallback dataset instead of the primary "
            "(useful when the primary is unavailable or you want GPLv2 "
            "raw text)."
        ),
    )
    args = parser.parse_args()

    try:
        fetch_task(
            task=args.task,
            out_dir=args.out_dir,
            max_samples=args.max_samples,
            use_fallback=args.fallback,
        )
    except SystemExit:
        raise
    except Exception as e:
        msg = str(e)
        if "GatedRepoError" in type(e).__name__ or "gated" in msg.lower():
            raise SystemExit(
                f"Gated dataset encountered: {type(e).__name__}\n"
                f"Switch to --fallback to use an open dataset."
            ) from e
        raise SystemExit(f"Fetch failed: {type(e).__name__}: {msg}") from e
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
