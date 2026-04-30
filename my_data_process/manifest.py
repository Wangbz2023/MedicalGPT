"""
输出 manifest.json：记录产物文件路径与 SHA-256（整文件指纹）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    artifacts: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    artifacts: 逻辑名 -> 文件路径，为每个文件计算 sha256。
    """
    files_out: List[Dict[str, Any]] = []
    for name, fpath in artifacts.items():
        p = Path(fpath)
        if not p.is_file():
            raise FileNotFoundError(f"manifest 引用文件不存在: {fpath}")
        files_out.append(
            {
                "name": name,
                "path": str(p.resolve()),
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }
        )
    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files_out,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest_json(manifest: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
