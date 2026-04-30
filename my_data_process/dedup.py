"""
漏斗 2：基于 instruction+input 的 SHA-256 绝对去重（不依赖外部 ID）。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple


def concat_for_hash(instruction: str, input_text: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inp:
        return f"{inst}\n\n{inp}".strip()
    return inst


def text_sha256(instruction: str, input_text: str) -> str:
    """与 Huatuo-26M 方案一致：对拼接文本做 SHA-256 十六进制。"""
    s = concat_for_hash(instruction, input_text)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deduplicate_records(
    records: List[Dict[str, Any]],
    text_key_instruction: str = "instruction",
    text_key_input: str = "input",
    hash_field: str = "content_sha256",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    按拼接文本去重，保留首次出现的一条。
    在保留的记录上写入 content_sha256（若不存在）。
    返回 (去重后列表, 丢弃条数)。
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    dropped = 0
    for row in records:
        inst = row.get(text_key_instruction) or ""
        inp = row.get(text_key_input) or ""
        h = text_sha256(inst, inp)
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        row = dict(row)
        row[hash_field] = h
        out.append(row)
    return out, dropped
