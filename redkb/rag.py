"""redkb/rag.py —— 本地红色知识库 + 检索（BM25 + 向量内积，可选 FAISS）

设计要点
--------
* **不联网、不需要 embedding 模型**：中文按「字符二元组 + 英文词」切 token，
  用 BM25 做词面检索；向量部分用特征哈希（char-ngram feature hashing）把同一批
  token 映射成 1024 维单位向量，做内积检索。
* **两路召回后再融合**（Reciprocal Rank Fusion），词面命中与语义相近都能得分，
  比单用一路稳。
* 装了 `faiss-cpu` 就用 `faiss.IndexFlatIP`（与 numpy 内积结果一致，只是更快）；
  没装则退回 numpy 矩阵乘法，行为完全相同。
* 每一条命中的都是**库里的原文片段**，带 entry_id 与片段编号，方便在页面上
  逐条标注 [1][2]，从机制上避免“模型自由发挥”。

对外接口
--------
    kb = RedKB()                       # 默认加载 kb.jsonl
    kb.entries()                       # 全部条目（摘要字段）
    kb.get("kb-008")                   # 单条完整条目
    kb.search("遵义会议 转折点", top_k=5)
        -> [{"pid","entry_id","name","kind","text","score","bm25_rank","vec_rank"}, ...]
    kb.info()                          # 条目数 / 片段数 / 检索后端
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_KB = HERE / "kb.jsonl"

DIM = 1024              # 特征哈希维度
STOP_TOKENS = {"的", "了", "是", "在", "和", "与", "为", "有", "个", "这", "那",
               "一", "不", "我", "你", "他", "它", "们", "上", "下", "中", "之"}

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------------------
# 分词与向量化
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return (text or "").lower()


def tokens(text: str) -> List[str]:
    """中文按字符二元组切分（等价于最朴素的分词），英文数字按词。

    例：「遵义会议」→ 遵义 / 义会 / 会议
    """
    out: List[str] = []
    t = _norm(text)
    for run in _CJK.findall(t):
        if len(run) == 1:
            out.append(run)
            continue
        for i in range(len(run) - 1):
            bigram = run[i:i + 2]
            if bigram not in STOP_TOKENS:
                out.append(bigram)
        # 单字也进一份，短查询时有用
        out.extend(ch for ch in run if ch not in STOP_TOKENS)
    for word in _WORD.findall(t):
        if len(word) > 1 or word.isdigit():
            out.append(word)
    return out


def _hash_index(token: str) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % DIM


def embed(text: str) -> np.ndarray:
    """特征哈希向量（L2 归一化）。"""
    vec = np.zeros(DIM, dtype=np.float32)
    for tok in tokens(text):
        vec[_hash_index(tok)] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------

class Passage:
    __slots__ = ("pid", "entry_id", "name", "kind", "text", "tokens")

    def __init__(self, pid: str, entry_id: str, name: str, kind: str, text: str):
        self.pid = pid
        self.entry_id = entry_id
        self.name = name
        self.kind = kind
        self.text = text
        self.tokens = tokens(text)

    def brief(self) -> Dict[str, Any]:
        return {"pid": self.pid, "entry_id": self.entry_id, "name": self.name,
                "kind": self.kind, "text": self.text}


ENTRY_FIELDS = ("id", "name", "aliases", "type", "place", "era", "summary",
                "facts", "tags", "interpretation", "source_note", "reviewed")


class RedKB:
    """本地知识库：加载 jsonl → 切片段 → 建 BM25 + 向量索引。"""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_KB
        self.entries_list: List[Dict[str, Any]] = self._load(self.path)
        self.by_id: Dict[str, Dict[str, Any]] = {e["id"]: e for e in self.entries_list}

        self.passages: List[Passage] = []
        for e in self.entries_list:
            self.passages.extend(self._passages_of(e))

        self._n = len(self.passages)
        self._df: Dict[str, int] = {}
        self._tf: List[Dict[str, int]] = []
        self._dl: List[int] = []
        for p in self.passages:
            tf: Dict[str, int] = {}
            for tok in p.tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            self._dl.append(len(p.tokens))
            for tok in tf:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._avgdl = (sum(self._dl) / self._n) if self._n else 1.0

        # 向量矩阵（p × DIM）
        if self.passages:
            self._mat = np.vstack([embed(p.text) for p in self.passages])
        else:
            self._mat = np.zeros((0, DIM), dtype=np.float32)

        self.faiss_index = None
        self._faiss_err = ""
        self._build_faiss()

    # ------------------------------------------------------------ 加载
    @staticmethod
    def _load(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not path.is_file():
            return rows
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict) or "id" not in obj:
                    continue
                obj.setdefault("facts", [])
                obj.setdefault("tags", [])
                obj.setdefault("aliases", [])
                rows.append(obj)
        return rows

    @staticmethod
    def _passages_of(entry: Dict[str, Any]) -> List[Passage]:
        """一条目 → 若干片段：概述一段，每条史实各一段（便于精确引用）。"""
        eid = entry["id"]
        name = entry.get("name", eid)
        head = "；".join(x for x in (
            entry.get("name"), entry.get("type"), entry.get("place"),
            entry.get("era"), "、".join(entry.get("tags", [])),
        ) if x)
        out = [Passage(f"{eid}#summary", eid, name, "summary",
                       f"{head}。{entry.get('summary', '')}")]
        for i, fact in enumerate(entry.get("facts", []), 1):
            out.append(Passage(f"{eid}#fact{i}", eid, name, f"fact{i}", str(fact)))
        if entry.get("interpretation"):
            out.append(Passage(f"{eid}#tip", eid, name, "tip",
                               f"讲解要点：{entry['interpretation']}"))
        return out

    def _build_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            if self._mat.shape[0] == 0:
                return
            index = faiss.IndexFlatIP(DIM)
            index.add(self._mat.astype(np.float32))
            self.faiss_index = index
            self._backend = "faiss"
        except Exception as exc:            # 没装 faiss 就用 numpy，结果一致
            self._faiss_err = f"{type(exc).__name__}: {exc}"
            self._backend = "numpy"

    # ------------------------------------------------------------ 检索
    def _bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> List[float]:
        q_tokens = tokens(query)
        scores = [0.0] * self._n
        if not q_tokens or not self._n:
            return scores
        for tok in set(q_tokens):
            df = self._df.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(tok, 0)
                if not f:
                    continue
                dl = self._dl[i] or 1
                scores[i] += idf * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl / self._avgdl))
        return scores

    def _vector(self, query: str) -> List[float]:
        if not self._n:
            return []
        qv = embed(query).reshape(1, -1)
        if self.faiss_index is not None:
            sims, idx = self.faiss_index.search(qv.astype(np.float32), self._n)
            out = [0.0] * self._n
            for score, i in zip(sims[0], idx[0]):
                if 0 <= int(i) < self._n:
                    out[int(i)] = float(score)
            return out
        return [float(x) for x in (self._mat @ qv[0])]

    def search(self, query: str, top_k: int = 6,
               entry_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """混合检索：BM25 + 向量，RRF 融合。可按 entry_id 限定在某一文物内。"""
        query = (query or "").strip()
        if not query or not self._n:
            return []

        bm25 = self._bm25(query)
        vec = self._vector(query)

        def ranks(scores: Sequence[float]) -> Dict[int, int]:
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return {i: r for r, i in enumerate(order)}

        r_bm25, r_vec = ranks(bm25), ranks(vec)
        fused: List[Tuple[float, int]] = []
        for i in range(self._n):
            if entry_id and self.passages[i].entry_id != entry_id:
                continue
            if bm25[i] <= 0 and vec[i] <= 0:
                continue
            score = 1.0 / (60 + r_bm25[i]) + 1.0 / (60 + r_vec[i])
            fused.append((score, i))
        fused.sort(reverse=True)

        out: List[Dict[str, Any]] = []
        for score, i in fused[:top_k]:
            p = self.passages[i]
            out.append({
                **p.brief(),
                "score": round(score, 6),
                "bm25": round(bm25[i], 4),
                "vector": round(vec[i], 4),
                "bm25_rank": r_bm25[i] + 1,
                "vec_rank": r_vec[i] + 1,
            })
        return out

    # ------------------------------------------------------------ 其他
    def passages_for(self, entry_id: str) -> List[Dict[str, Any]]:
        """取某个条目的片段（概述 → 各条史实 → 讲解要点），按库里的顺序。

        讲某一件文物时用它，比用检索结果更稳：不会因为一句弱相关的命中
        而漏掉关键史实。
        """
        rows = [p for p in self.passages if p.entry_id == entry_id]
        return [p.brief() for p in rows]

    def entries(self) -> List[Dict[str, Any]]:
        return [{
            "id": e["id"], "name": e.get("name", ""), "type": e.get("type", ""),
            "place": e.get("place", ""), "era": e.get("era", ""),
            "tags": e.get("tags", []), "reviewed": bool(e.get("reviewed")),
        } for e in self.entries_list]

    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self.by_id.get(entry_id)

    def info(self) -> Dict[str, Any]:
        return {
            "kb_file": str(self.path),
            "entries": len(self.entries_list),
            "passages": self._n,
            "dim": DIM,
            "retrieval": "BM25 + 特征哈希向量（RRF 融合）",
            "vector_backend": self._backend,
            "faiss_error": self._faiss_err or None,
            "reviewed": sum(1 for e in self.entries_list if e.get("reviewed")),
        }


# ---------------------------------------------------------------------------
# 单例（供服务端复用，避免每次请求重建索引）
# ---------------------------------------------------------------------------

_KB: Optional[RedKB] = None
_KB_LOCK = threading.Lock()


def get_kb(path: Optional[Path] = None) -> RedKB:
    global _KB
    if _KB is None:
        with _KB_LOCK:
            if _KB is None:
                _KB = RedKB(path)
    return _KB


if __name__ == "__main__":  # 命令行自测：python redkb/rag.py 遵义会议
    import sys

    kb = get_kb()
    print(json.dumps(kb.info(), ensure_ascii=False, indent=2))
    q = " ".join(sys.argv[1:]) or "遵义会议"
    print(f"\n查询：{q}")
    for hit in kb.search(q, top_k=5):
        print(f"  [{hit['score']:.4f}] {hit['name']} · {hit['kind']} · {hit['text'][:60]}…")
