"""
从 C-Eval 医疗子集 CSV 加载锚点题干（默认 3 科目 × 验证集 ≈ 90 题）。
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class AnchorItem:
    """单条锚点：用于检索编码与溯源字段。"""

    subject: str
    row_index: int
    question_id: str
    text: str


def _row_to_text(row: dict) -> Optional[str]:
    q = (row.get("question") or row.get("Question") or "").strip()
    if not q:
        return None
    lines = [q]
    for key in ("A", "B", "C", "D"):
        v = (row.get(key) or "").strip()
        if v:
            lines.append(f"{key}. {v}")
    return "\n".join(lines)


def load_anchors_from_ceval_csv(csv_path: str, subject: str) -> List[AnchorItem]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"锚点 CSV 不存在: {csv_path}")
    items: List[AnchorItem] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = _row_to_text(row)
            if not text:
                continue
            qid = (
                row.get("id")
                or row.get("Id")
                or row.get("ID")
                or f"{subject}_{i}"
            )
            items.append(
                AnchorItem(
                    subject=subject,
                    row_index=i,
                    question_id=str(qid),
                    text=text,
                )
            )
    return items


def load_anchors_from_directory(
    ceval_dir: str,
    patterns: Optional[List[str]] = None,
) -> List[AnchorItem]:
    """
    从目录加载多个 `*_val.csv`（与 download_ceval_val.py 输出一致）。
    patterns: 若指定，仅匹配这些科目名，如 ["basic_medicine", "clinical_medicine", "physician"]。
    """
    root = Path(ceval_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"C-Eval 目录不存在: {ceval_dir}")

    if patterns:
        files = [root / f"{p}_val.csv" for p in patterns]
    else:
        files = sorted(root.glob("*_val.csv"))

    all_items: List[AnchorItem] = []
    for fp in files:
        if not fp.is_file():
            continue
        stem = fp.stem
        subject = stem.replace("_val", "") if "_val" in stem else stem
        if patterns and subject not in patterns:
            continue
        all_items.extend(load_anchors_from_ceval_csv(str(fp), subject=subject))
    if not all_items:
        raise ValueError(f"未从 {ceval_dir} 解析到任何锚点，请确认已运行 download_ceval_val.py")
    return all_items
