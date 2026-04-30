"""
漏斗 3：判空、长度上限、HTML 与黑名单词过滤。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

DEFAULT_BLACKLIST = [
    "包治",
    "加微信",
    "100%有效",
    "点击链接",
]

# 常见 HTML 标签片段（含 <br> 等）
_HTML_PATTERN = re.compile(r"<[^>]+>")


def total_text_length(row: Dict[str, Any]) -> int:
    parts = [
        str(row.get("instruction") or ""),
        str(row.get("input") or ""),
        str(row.get("output") or ""),
    ]
    return sum(len(p) for p in parts)


def has_html_noise(text: str) -> bool:
    return bool(_HTML_PATTERN.search(text or ""))


def hits_blacklist(text: str, blacklist: List[str]) -> bool:
    t = text or ""
    for w in blacklist:
        if w and w in t:
            return True
    return False


def clean_records(
    records: List[Dict[str, Any]],
    max_total_length: int = 2048,
    blacklist: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    剔除：
    - instruction / output 为空（input 允许为空）
    - 总长度超过 max_total_length
    - 任意字段含 HTML 标签
    - 合并全文命中黑名单
    返回 (清洗后列表, 丢弃条数)。
    """
    bl = blacklist if blacklist is not None else DEFAULT_BLACKLIST
    out: List[Dict[str, Any]] = []
    dropped = 0
    for row in records:
        inst = str(row.get("instruction") or "").strip()
        out_text = str(row.get("output") or "").strip()
        inp = str(row.get("input") or "")

        if not inst or not out_text:
            dropped += 1
            continue

        combined_for_scan = f"{inst}\n{inp}\n{out_text}"
        if has_html_noise(combined_for_scan):
            dropped += 1
            continue
        if hits_blacklist(combined_for_scan, bl):
            dropped += 1
            continue

        row = dict(row)
        row["instruction"] = inst
        row["input"] = inp.strip()
        row["output"] = out_text

        if total_text_length(row) > max_total_length:
            dropped += 1
            continue

        out.append(row)
    return out, dropped
