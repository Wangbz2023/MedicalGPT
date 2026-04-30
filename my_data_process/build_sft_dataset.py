"""
三段式 SFT 数据构建主入口：向量化 → 召回与溯源 → SHA-256 去重 → 规则清洗 → 产物与 manifest。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 支持 `python my_data_process/build_sft_dataset.py` 与 `python -m my_data_process.build_sft_dataset`
if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from my_data_process.anchors import load_anchors_from_directory
from my_data_process.cleaning import clean_records
from my_data_process.dedup import deduplicate_records
from my_data_process.manifest import build_manifest, write_manifest_json
from my_data_process.retrieval import prepare_corpus_embeddings, retrieve_with_traceability

DEFAULT_HUATUO = "/root/autodl-tmp/data/raw_data/HuatuoGPT2-GPT4-SFT-140K.json"
DEFAULT_CEVAL_DIR = "/root/autodl-tmp/data/ceval"
DEFAULT_PROCESSED = "/root/autodl-tmp/data/processed_data"
DEFAULT_MODEL = "/root/autodl-tmp/models/bge-large-zh-v1.5"
DEFAULT_ANCHOR_SUBJECTS = ("basic_medicine", "clinical_medicine", "physician")
DEFAULT_OUTPUT_NAME = "sft_v1_traceable_clean.json"
INTERNAL_ROW_KEYS = ("_corpus_index",)


@dataclass
class PipelineResult:
    """各阶段统计与输出路径。"""

    anchor_count: int
    retrieved_count: int
    dedup_kept: int
    dedup_dropped: int
    clean_kept: int
    clean_dropped: int
    output_path: Path
    manifest_path: Optional[Path]


def _strip_internal_fields(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in records:
        d = {k: v for k, v in row.items() if k not in INTERNAL_ROW_KEYS}
        out.append(d)
    return out


def run_sft_pipeline(
    huatuo_path: str,
    ceval_dir: str,
    processed_dir: str,
    *,
    anchor_subjects: Optional[Sequence[str]] = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 100,
    batch_size: int = 32,
    max_total_length: int = 2048,
    output_filename: str = DEFAULT_OUTPUT_NAME,
    manifest_filename: str = "manifest.json",
    force_rebuild_emb: bool = False,
    write_manifest: bool = True,
    pretty_json: bool = True,
) -> PipelineResult:
    """
    执行完整漏斗：召回 → 去重 → 清洗 → 写 JSON；可选写 manifest（整文件 SHA-256）。
    """
    subjects = list(anchor_subjects) if anchor_subjects is not None else list(DEFAULT_ANCHOR_SUBJECTS)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    anchors = load_anchors_from_directory(ceval_dir, patterns=subjects)
    rows, _texts, corpus_emb, _cache_dir = prepare_corpus_embeddings(
        huatuo_path,
        model_name=model_name,
        cache_root=processed,
        batch_size=batch_size,
        force_rebuild=force_rebuild_emb,
    )
    retrieved = retrieve_with_traceability(
        rows,
        _texts,
        corpus_emb,
        anchors,
        model_name=model_name,
        top_k=top_k,
        batch_size=batch_size,
    )
    deduped, dedup_dropped = deduplicate_records(retrieved)
    cleaned, clean_dropped = clean_records(deduped, max_total_length=max_total_length)

    out_path = processed / output_filename
    final_rows = _strip_internal_fields(cleaned)
    with open(out_path, "w", encoding="utf-8") as f:
        if pretty_json:
            json.dump(final_rows, f, ensure_ascii=False, indent=2)
        else:
            json.dump(final_rows, f, ensure_ascii=False)

    manifest_path: Optional[Path] = None
    if write_manifest:
        manifest_path = processed / manifest_filename
        manifest = build_manifest(
            {"sft_traceable_clean": str(out_path)},
            extra={
                "pipeline": "sft_three_stage",
                "model_name": model_name,
                "top_k_per_anchor": top_k,
                "anchor_subjects": subjects,
                "anchor_count": len(anchors),
                "retrieved_count": len(retrieved),
                "dedup_dropped": dedup_dropped,
                "clean_dropped": clean_dropped,
                "final_count": len(final_rows),
            },
        )
        write_manifest_json(manifest, manifest_path)

    return PipelineResult(
        anchor_count=len(anchors),
        retrieved_count=len(retrieved),
        dedup_kept=len(deduped),
        dedup_dropped=dedup_dropped,
        clean_kept=len(cleaned),
        clean_dropped=clean_dropped,
        output_path=out_path,
        manifest_path=manifest_path,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="医疗 SFT 三段式数据构建（向量化 / 召回 / 去重 / 清洗 / manifest）")
    p.add_argument("--huatuo-json", default=DEFAULT_HUATUO, help="Huatuo 140K Alpaca JSON 路径")
    p.add_argument("--ceval-dir", default=DEFAULT_CEVAL_DIR, help="C-Eval val CSV 目录（download_ceval_val.py 输出）")
    p.add_argument(
        "--anchor-subjects",
        default=",".join(DEFAULT_ANCHOR_SUBJECTS),
        help="逗号分隔科目名，需与目录下 *_val.csv 对应",
    )
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED, help="加工数据与向量缓存根目录")
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers 模型名")
    p.add_argument("--top-k", type=int, default=100, help="每锚点 Top-K 召回")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-total-length", type=int, default=2048, help="instruction+input+output 总长度上限")
    p.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="最终 JSON 文件名")
    p.add_argument("--manifest-name", default="manifest.json", help="manifest 文件名（位于 processed-dir）")
    p.add_argument("--force-rebuild-emb", action="store_true", help="忽略缓存重建候选库向量")
    p.add_argument("--skip-manifest", action="store_true", help="不写 manifest（仍会写最终 JSON）")
    p.add_argument("--compact-json", action="store_true", help="输出单行 JSON（默认缩进美化）")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    subjects = [s.strip() for s in args.anchor_subjects.split(",") if s.strip()]
    try:
        r = run_sft_pipeline(
            args.huatuo_json,
            args.ceval_dir,
            args.processed_dir,
            anchor_subjects=subjects,
            model_name=args.model,
            top_k=args.top_k,
            batch_size=args.batch_size,
            max_total_length=args.max_total_length,
            output_filename=args.output_name,
            manifest_filename=args.manifest_name,
            force_rebuild_emb=args.force_rebuild_emb,
            write_manifest=not args.skip_manifest,
            pretty_json=not args.compact_json,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"流水线失败: {e}", file=sys.stderr)
        raise

    print(
        f"锚点数={r.anchor_count} | 召回={r.retrieved_count} | "
        f"去重后={r.dedup_kept}（丢弃 {r.dedup_dropped}）| "
        f"清洗后={r.clean_kept}（丢弃 {r.clean_dropped}）"
    )
    print(f"输出: {r.output_path}")
    if r.manifest_path is not None:
        print(f"Manifest: {r.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
