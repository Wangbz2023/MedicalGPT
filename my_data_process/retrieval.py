"""
漏斗 1：候选库向量化 + 锚点 Top-K 召回 + 注入 anchor_source。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .anchors import AnchorItem
from .embedding import build_or_load_corpus_embeddings, encode_query_batch


def corpus_text_for_embedding(instruction: str, input_text: str) -> str:
    """与去重键一致的「检索侧」文本：instruction 与 input 拼接。"""
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return f"{inst}\n\n{inp}".strip()
    return inst


def _normalize_role(turn: Dict[str, Any]) -> str:
    r = turn.get("from")
    if r is None:
        r = turn.get("role")
    return str(r or "").strip().lower()


def sharegpt_row_to_alpaca(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    将单条 ShareGPT 样本转为 Alpaca 三字段。每条顶层记录只产出一条样本：
    取首轮「连续 human/user 的拼接」对「紧随其后的首个 gpt/assistant」。
    可选 system 段拼在 instruction 之前。
    """
    conv = row.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None

    system_chunks: List[str] = []
    human_buf: List[str] = []

    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn)
        val = str(turn.get("value") or "").strip()
        if role in ("system",):
            if val:
                system_chunks.append(val)
            continue
        if role in ("human", "user"):
            if val:
                human_buf.append(val)
            continue
        if role in ("gpt", "assistant", "chatglm", "bing", "bard"):
            if not human_buf:
                continue
            instruction = "\n\n".join(human_buf).strip()
            if system_chunks:
                instruction = "\n\n".join(system_chunks) + "\n\n" + instruction
            rec: Dict[str, Any] = {
                "instruction": instruction,
                "input": "",
                "output": val,
            }
            oid = row.get("id")
            if oid is not None:
                rec["id"] = oid
            return rec
    return None


def load_huatuo_alpaca_json(path: str) -> List[Dict[str, Any]]:
    """
    加载 SFT 语料 JSON（数组或带 data 数组的对象）。

    自动识别：
    - ShareGPT：存在 conversations（from/value），在加载时规范为 instruction / input / output；
    - Alpaca：instruction、input、output。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            data = raw["data"]
        else:
            raise ValueError(f"JSON 需为数组或包含 data 数组的对象: {path}")
    else:
        raise ValueError(f"无法解析 JSON 结构: {path}")

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        sample: Dict[str, Any] | None = None
        if isinstance(row.get("conversations"), list):
            sample = sharegpt_row_to_alpaca(row)
        elif "instruction" in row or "output" in row:
            sample = {
                "instruction": row.get("instruction") or "",
                "input": row.get("input") or "",
                "output": row.get("output") or "",
            }
            oid = row.get("id")
            if oid is not None:
                sample["id"] = oid

        if not sample:
            continue
        if not str(sample.get("output") or "").strip():
            continue

        sample = dict(sample)
        sample["_corpus_index"] = i
        out.append(sample)

    if not out:
        raise ValueError(f"{path} 中未找到有效样本（Alpaca 或 ShareGPT）")
    return out


def top_k_indices(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    query_emb: (Nq, D) 已 L2 归一化
    corpus_emb: (Nc, D) 已 L2 归一化
    返回: (Nq, k) 每个 query 的 top-k 行索引（分数降序）
    """
    if corpus_emb.shape[0] == 0:
        return np.zeros((query_emb.shape[0], 0), dtype=np.int64)
    sim = np.matmul(query_emb, corpus_emb.T)
    k = min(k, corpus_emb.shape[0])
    top = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    # 每行按分数排序
    rows = np.arange(sim.shape[0])[:, None]
    scores = sim[rows, top]
    order = np.argsort(-scores, axis=1)
    return top[rows, order]


def build_anchor_source(
    anchor: AnchorItem,
    rank: int,
    score: float,
) -> Dict[str, Any]:
    return {
        "anchor_subject": anchor.subject,
        "anchor_row_index": anchor.row_index,
        "anchor_question_id": anchor.question_id,
        "anchor_text_preview": anchor.text[:200] + ("..." if len(anchor.text) > 200 else ""),
        "rank": rank,
        "similarity_score": float(score),
    }


def retrieve_with_traceability(
    corpus_rows: List[Dict[str, Any]],
    corpus_texts: List[str],
    corpus_emb: np.ndarray,
    anchors: List[AnchorItem],
    model_name: str,
    top_k: int,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    对每个锚点取 Top-K，合并为召回列表（允许跨锚点重复）。
    每条记录为原始样本 dict + anchor_source。
    """
    anchor_texts = [a.text for a in anchors]
    q_emb = encode_query_batch(model_name, anchor_texts, batch_size=batch_size)

    sim_full = np.matmul(q_emb, corpus_emb.T)
    k = min(top_k, corpus_emb.shape[0])
    top_idx = top_k_indices(q_emb, corpus_emb, k)

    retrieved: List[Dict[str, Any]] = []
    for i, anchor in enumerate(anchors):
        for rank, j in enumerate(top_idx[i]):
            j = int(j)
            row = dict(corpus_rows[j])
            score = float(sim_full[i, j])
            row["anchor_source"] = build_anchor_source(anchor, rank=rank + 1, score=score)
            retrieved.append(row)
    return retrieved


def prepare_corpus_embeddings(
    huatuo_path: str,
    model_name: str,
    cache_root: Path,
    batch_size: int,
    force_rebuild: bool,
) -> Tuple[List[Dict[str, Any]], List[str], np.ndarray, Path]:
    """加载语料、构建/读取向量，返回 rows, texts, corpus_emb, cache_dir。"""
    rows = load_huatuo_alpaca_json(huatuo_path)
    texts = [corpus_text_for_embedding(r["instruction"], r["input"]) for r in rows]
    emb, cdir = build_or_load_corpus_embeddings(
        texts,
        source_path=huatuo_path,
        model_name=model_name,
        cache_root=Path(cache_root),
        batch_size=batch_size,
        force_rebuild=force_rebuild,
    )
    return rows, texts, emb, cdir
