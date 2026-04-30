"""
候选库离线向量化：加载 BGE、批量编码、归一化向量缓存（供 cosine 用点积加速）。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm


def file_quick_fingerprint(path: str) -> str:
    """用于判断源文件是否变化，避免对超大文件做全量 SHA-256。"""
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(min(1024 * 1024, st.st_size)))
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "需要安装 sentence-transformers：pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name, trust_remote_code=True)


def encode_texts(
    model,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """将文本编码为 float32 矩阵，默认 L2 归一化以便 cosine = dot。"""
    out: List[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="encode", unit="batch")
    for i in iterator:
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        out.append(np.asarray(emb, dtype=np.float32))
    if not out:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack(out)


def cache_dir_for_corpus(cache_root: Path, source_path: str, model_name: str) -> Path:
    key = f"{os.path.abspath(source_path)}|{model_name}|{file_quick_fingerprint(source_path)}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() else "_" for c in model_name.replace("/", "_"))
    return cache_root / f"corpus_emb_{safe}_{h}"


def build_or_load_corpus_embeddings(
    corpus_texts: List[str],
    source_path: str,
    model_name: str,
    cache_root: Path,
    batch_size: int = 32,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, Path]:
    """
    构建或读取候选库向量缓存。
    返回 (embeddings, cache_dir)。
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cdir = cache_dir_for_corpus(cache_root, source_path, model_name)
    emb_path = cdir / "embeddings.npy"
    meta_path = cdir / "meta.json"

    meta = {
        "model_name": model_name,
        "num_texts": len(corpus_texts),
        "source_path": os.path.abspath(source_path),
        "source_fingerprint": file_quick_fingerprint(source_path),
    }

    if (
        not force_rebuild
        and emb_path.is_file()
        and meta_path.is_file()
    ):
        with open(meta_path, "r", encoding="utf-8") as f:
            old = json.load(f)
        if (
            old.get("model_name") == meta["model_name"]
            and old.get("num_texts") == meta["num_texts"]
            and old.get("source_fingerprint") == meta["source_fingerprint"]
        ):
            emb = np.load(emb_path)
            if emb.shape[0] == len(corpus_texts):
                return emb, cdir

    model = load_sentence_transformer(model_name)
    emb = encode_texts(model, corpus_texts, batch_size=batch_size, normalize=True)
    cdir.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return emb, cdir


def encode_query_batch(
    model_name: str,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    """仅对锚点查询做在线编码（可配合缓存的候选矩阵做检索）。"""
    model = load_sentence_transformer(model_name)
    return encode_texts(model, texts, batch_size=batch_size, normalize=True)
