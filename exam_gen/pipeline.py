"""
pipeline.py — المنسّق: من الملفات الثلاثة إلى امتحان مولّد ومُتحقَّق منه.

الإخراج بنفس schema ملف all_exam_questions.json (drop-in) مع حقول زيادة:
provenance_chunk_ids, verification, worked_solution, computation_spec.
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict
from typing import List

from .blueprint import build_blueprint, summarize_blueprint, Slot
from .retrieval import Retriever
from . import prompts as P
from .verify import verify_question


def _qid(text: str, i: int) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:4]
    return f"gen_{i:03d}_{h}"


def _to_schema(q: dict, slot: Slot, verdict, idx: int) -> dict:
    """يحوّل مخرجات المولّد إلى نفس بنية بنك الأسئلة الأصلي (دفاعي: لا يفترض
    اكتمال حقول q، لأن مخرجات LLM قد تكون مشوّهة خصوصاً عند الرفض)."""
    question_text = q.get("question", "")
    options = q.get("options", [])
    if not isinstance(options, list):
        options = []
    return {
        "question_id": _qid(question_text or f"malformed_{idx}", idx),
        "question_type": "multiple_choice",
        "question": question_text,
        "options": options,
        "correct_answer": q.get("correct_answer"),
        "topic": slot.topic,
        "academic": {
            "knowledge_points": q.get("provenance_chunk_ids", []),
            "cognitive_level": slot.cognitive_level,
            "difficulty": slot.difficulty,
            "classification_confidence": None,
        },
        "doctor_style": {
            "action_verbs": [],
            "reasoning_steps": None,
            "distractor_pattern": q.get("distractor_rationale", ""),
            "explanation": q.get("worked_solution", ""),
        },
        "references_figure": False,
        "references_formula": bool(q.get("computation_spec") or q.get("verification_code")),
        # حقول إضافية للتتبّع والتحقق:
        "computation_spec": q.get("computation_spec"),
        "verification_code": q.get("verification_code"),
        "provenance_chunk_ids": q.get("provenance_chunk_ids", []),
        "verification": verdict.as_dict(),
        "source": {"document_id": "generated", "page": None},
    }


class ExamPipeline:
    def __init__(self, corpus, llm, embed_fn=None,
                 max_regenerations: int = 2, require_blind: bool = True,
                 course_description: str = "", domain: str | None = None,
                 request_delay: float = 0.0):
        """
        course_description: وصف حر للمادة (أي مادة). إن تُرك فارغاً يُستنتج
                            من بروفايل الأسلوب.
        domain: اسم إضافة مادة مسجّلة في domains.py (اختياري — النظام
                يتحقق عبر verification_code العام حتى بدونها).
        request_delay: مهلة ثوانٍ بين نداءات LLM المتتالية — مفيدة لمفاتيح
                       الطبقة المجانية ذات حدود المعدل الصارمة (تقليل
                       Connection reset by peer / 429).
        """
        self.corpus = corpus
        self.llm = llm
        self.retriever = Retriever(corpus, embed_fn)
        self.max_regenerations = max_regenerations
        self.require_blind = require_blind
        self.course_description = course_description
        self.domain = domain
        self.request_delay = request_delay

    def _generate_one(self, slot: Slot, temperature: float) -> dict:
        if self.request_delay > 0:
            time.sleep(self.request_delay)
        grounding = self.retriever.grounding_for(slot.topic, slot.cognitive_level, k=5)
        exemplars = self.retriever.exemplars_for(slot.topic, slot.cognitive_level, k=3)
        msgs = P.build_generator_messages(
            slot, grounding, exemplars, self.corpus.style_profile,
            course_description=self.course_description, domain=self.domain)
        raw = self.llm.complete_json(msgs, temperature=temperature)
        # اضمن اتساق correct_answer مع is_correct إن أمكن
        if "correct_answer" not in raw:
            for i, o in enumerate(raw.get("options", []) or []):
                if isinstance(o, dict) and o.get("is_correct"):
                    raw["correct_answer"] = o.get("label", chr(65 + i))
                    break
        return raw

    def generate_exam(self, n_questions: int, seed: int = 0,
                      topic_whitelist=None, coverage: bool = False,
                      verbose: bool = True):
        blueprint = build_blueprint(self.corpus, n_questions, seed,
                                    topic_whitelist, coverage=coverage)
        if verbose:
            print(summarize_blueprint(blueprint))

        accepted, review, rejected = [], [], []
        for slot in blueprint:
            best = None
            for attempt in range(self.max_regenerations + 1):
                temp = 0.7 if attempt == 0 else 0.9
                try:
                    q = self._generate_one(slot, temp)
                except Exception as e:
                    if verbose:
                        print(f"  [slot {slot.index}] فشل التوليد: {e}")
                    continue
                verdict = verify_question(q, self.llm, P, self.retriever,
                                          require_blind=self.require_blind)
                record = _to_schema(q, slot, verdict, slot.index)
                if verdict.status == "verified":
                    best = record
                    break
                best = record if best is None else best  # احتفظ بأفضل محاولة
                if verbose:
                    print(f"  [slot {slot.index}] محاولة {attempt+1}: {verdict.status}")
            # تصنيف
            status = best["verification"]["status"]
            (accepted if status == "verified"
             else review if status == "needs_review"
             else rejected).append(best)

        result = {
            "blueprint": [asdict(s) for s in blueprint],
            "counts": {"verified": len(accepted),
                       "needs_review": len(review),
                       "rejected": len(rejected)},
            "verified": accepted,
            "needs_review": review,
            "rejected": rejected,
        }
        if verbose:
            print("\nالنتيجة:", result["counts"])
        return result


def save_exam(result: dict, path: str, include_review: bool = False):
    questions = list(result["verified"])
    if include_review:
        questions += result["needs_review"]

    # توزيع موضع الإجابة الصحيحة (لرصد الانحياز في فصل النتائج)
    from collections import Counter
    pos = Counter(q.get("correct_answer") for q in questions)

    out = {"document_id": "generated_exam",
           "total_questions": len(questions),
           "questions": questions,
           "generation_report": {
               **result["counts"],
               "answer_position_distribution": dict(sorted(pos.items(),
                                                           key=lambda x: str(x[0]))),
               "llm_usage": result.get("llm_usage", {}),
           },
           # شفافية كاملة: تفاصيل المرفوض والموسوم للمراجعة (لا تدخل الامتحان،
           # لكنها ضرورية لفصل النتائج والتدقيق)
           "needs_review_details": result.get("needs_review", []),
           "rejected_details": [
               {"question_id": r.get("question_id"),
                "topic": r.get("topic"),
                "question": r.get("question", "")[:200],
                "reason": (r.get("verification", {}).get("layers") or
                           [{}])[-1].get("detail", "")}
               for r in result.get("rejected", [])
           ]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path
