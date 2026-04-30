"""SFT 三段式数据构建：向量化、召回、去重、清洗、manifest（整文件 SHA-256）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from my_data_process.anchors import AnchorItem, load_anchors_from_directory
from my_data_process.cleaning import clean_records
from my_data_process.dedup import deduplicate_records, text_sha256
from my_data_process.embedding import (
    build_or_load_corpus_embeddings,
    encode_query_batch,
    file_quick_fingerprint,
)
from my_data_process.manifest import build_manifest, sha256_file, write_manifest_json
from my_data_process.retrieval import (
    load_huatuo_alpaca_json,
    prepare_corpus_embeddings,
    retrieve_with_traceability,
)

if TYPE_CHECKING:
    from my_data_process.build_sft_dataset import PipelineResult, run_sft_pipeline

__all__ = [
    "AnchorItem",
    "PipelineResult",
    "build_manifest",
    "build_or_load_corpus_embeddings",
    "clean_records",
    "deduplicate_records",
    "encode_query_batch",
    "file_quick_fingerprint",
    "load_anchors_from_directory",
    "load_huatuo_alpaca_json",
    "prepare_corpus_embeddings",
    "retrieve_with_traceability",
    "run_sft_pipeline",
    "sha256_file",
    "text_sha256",
    "write_manifest_json",
]


def __getattr__(name: str) -> Any:
    if name == "PipelineResult" or name == "run_sft_pipeline":
        from my_data_process import build_sft_dataset as _build

        return getattr(_build, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
