"""
retrieval.py — استرجاع (RAG) مزدوج لكل خانة:
  (1) grounding: أفضل chunks من normalized_knowledge (المحتوى/التأريض).
  (2) exemplars: أفضل أسئلة حقيقية مشابهة (الأسلوب) لحقن few-shot.

يعتمد افتراضياً على تشابه كلمات مفتاحية خفيف (بدون شبكة). إن توفّر
مزوّد embeddings يمكن تمرير embed_fn لرفع الدقة.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Callable, List, Optional

_AR_EN = re.compile(r"[A-Za-z\u0600-\u06FF]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _AR_EN.findall(text or "")]


def _tf(tokens: List[str]):
    c = Counter(tokens)
    n = sum(c.values()) or 1
    return {t: v / n for t, v in c.items()}


def _cosine_bow(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    db = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return num / (da * db)


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(_s(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_s(x) for x in v.values())
    return str(v)


def _chunk_text(ch: dict) -> str:
    return " ".join(filter(None, [
        _s(ch.get("title")), _s(ch.get("content")),
        _s(ch.get("keywords")), _s(ch.get("embedding_text")),
    ]))


class Retriever:
    def __init__(self, corpus, embed_fn: Optional[Callable[[str], list]] = None):
        self.corpus = corpus
        self.embed_fn = embed_fn
        self._chunk_bows = [(_tf(_tokens(_chunk_text(c))), c) for c in corpus.knowledge_chunks]
        self._q_bows = [(_tf(_tokens(q.get("embedding_text", q["question"]))), q)
                        for q in corpus.exam_questions]
        # فهارس embeddings (تُحسب مرة عند التهيئة إن توفّر embed_fn)
        self._chunk_vecs = None
        self._q_vecs = None
        if embed_fn is not None:
            self._chunk_vecs = [embed_fn(_chunk_text(c)) for c in corpus.knowledge_chunks]
            self._q_vecs = [embed_fn(q.get("embedding_text", q["question"]))
                            for q in corpus.exam_questions]

    @staticmethod
    def _cos_vec(a, b) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a)) or 1e-9
        db = math.sqrt(sum(y * y for y in b)) or 1e-9
        return num / (da * db)

    def _sem_scores(self, query: str, vecs) -> Optional[List[float]]:
        if self.embed_fn is None or vecs is None:
            return None
        qv = self.embed_fn(query)
        return [self._cos_vec(qv, v) for v in vecs]

    # ---- grounding chunks ------------------------------------------------- #
    def grounding_for(self, topic: str, cognitive_level: str, k: int = 5) -> List[dict]:
        query_text = topic + " " + cognitive_level
        query = _tf(_tokens(query_text))
        prefer = {"exam_point": 1.15, "formula": 1.2, "definition": 1.1,
                  "summary": 1.0, "diagram": 0.6}
        sem = self._sem_scores(query_text, self._chunk_vecs)
        scored = []
        for i, (bow, ch) in enumerate(self._chunk_bows):
            s = _cosine_bow(query, bow)
            if sem is not None:                     # هجين: 40٪ كلمات + 60٪ دلالي
                s = 0.4 * s + 0.6 * sem[i]
            s *= prefer.get(ch.get("chunk_type"), 1.0)
            scored.append((s, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for _, ch in scored[:k]]

    # ---- style exemplars -------------------------------------------------- #
    def exemplars_for(self, topic: str, cognitive_level: str, k: int = 3) -> List[dict]:
        query_text = topic + " " + cognitive_level
        query = _tf(_tokens(query_text))
        sem = self._sem_scores(query_text, self._q_vecs)
        scored = []
        for i, (bow, q) in enumerate(self._q_bows):
            s = _cosine_bow(query, bow)
            if sem is not None:
                s = 0.4 * s + 0.6 * sem[i]
            if q["topic"] == topic:
                s += 0.25
            if q.get("academic", {}).get("cognitive_level") == cognitive_level:
                s += 0.15
            scored.append((s, q))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [q for _, q in scored[:k]]

    # ---- dedup check ضد البنك الأصلي ------------------------------------- #
    def max_similarity_to_bank(self, question_text: str) -> float:
        bow = _tf(_tokens(question_text))
        return max((_cosine_bow(bow, qb) for qb, _ in self._q_bows), default=0.0)
