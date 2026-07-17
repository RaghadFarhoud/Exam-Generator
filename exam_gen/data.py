"""data.py — تحميل الملفات وتوحيد الوصول إليها.

يدعم ملفاً واحداً، أو قائمة ملفات، أو مجلداً/نمط glob — لدمج كل
الدورات السابقة وكل المحاضرات المستخرجة في مدوّنة واحدة.
"""
from __future__ import annotations
import glob as _glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, List[str]]


def _resolve_paths(spec: PathLike) -> List[str]:
    """يحوّل str/list/glob/مجلد إلى قائمة مسارات ملفات فعلية."""
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(_resolve_paths(s))
        return out
    p = Path(spec)
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.json"))
    if any(ch in str(spec) for ch in "*?[]"):
        return sorted(_glob.glob(str(spec)))
    return [str(spec)]


@dataclass
class Corpus:
    exam_questions: List[dict]          # all_exam_questions.json -> questions
    knowledge_chunks: List[dict]        # normalized_knowledge.json -> chunks
    style_profile: dict                 # doctor_style_profile.json

    @property
    def topics(self) -> List[str]:
        return sorted({q["topic"] for q in self.exam_questions})

    def questions_by_topic(self, topic: str) -> List[dict]:
        return [q for q in self.exam_questions if q["topic"] == topic]

    def questions_by_level(self, level: str) -> List[dict]:
        return [q for q in self.exam_questions
                if q.get("academic", {}).get("cognitive_level") == level]


def load_corpus(exam_path: PathLike, knowledge_path: PathLike, style_path: str) -> Corpus:
    """
    يحمّل ويدمج مصادر متعددة.
    - exam_path / knowledge_path: ملف واحد، قائمة، مجلد، أو نمط glob.
    - يوسم كل سؤال/chunk بملف مصدره ويزيل التكرار بالمعرّف.
    """
    exam_questions: List[dict] = []
    seen_q = set()
    for fp in _resolve_paths(exam_path):
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        items = data.get("questions", data if isinstance(data, list) else [])
        for q in items:
            qid = q.get("question_id") or q.get("question", "")[:80]
            if qid in seen_q:
                continue
            seen_q.add(qid)
            q.setdefault("_source_file", Path(fp).name)
            exam_questions.append(q)

    chunks: List[dict] = []
    seen_c = set()
    for fp in _resolve_paths(knowledge_path):
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        items = data.get("chunks", data if isinstance(data, list) else [])
        doc = data.get("document_name", Path(fp).stem)
        for c in items:
            cid = f"{doc}:{c.get('chunk_id')}"
            if cid in seen_c:
                continue
            seen_c.add(cid)
            c.setdefault("_source_doc", doc)
            chunks.append(c)

    style = json.loads(Path(_resolve_paths(style_path)[0]).read_text(encoding="utf-8"))
    return Corpus(exam_questions=exam_questions, knowledge_chunks=chunks, style_profile=style)
