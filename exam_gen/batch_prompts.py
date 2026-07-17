"""
batch_prompts.py — قوالب النداءات المُدفَّعة (سرّ تخفيض النداءات ~8x).

ثلاثة أنواع نداء فقط في النظام كله:
  1) توليد دفعة: نداء واحد يولّد 5-8 أسئلة متقاربة الموضوع بسياق مشترك.
  2) حلّ دفعة (cross-model): نموذج مختلف يحلّ كل أسئلة الدفعة دون رؤية
     المفاتيح — نداء واحد لعشرات الأسئلة.
  3) إصلاح دفعة: الأسئلة الفاشلة تُجمع بأسباب فشلها وتُصلَح بنداء واحد.

اللغة تُقرأ من بروفايل الأستاذ (resolve_language في prompts.py).
"""
from __future__ import annotations
import json
from typing import Dict, List

from .prompts import resolve_language, _LANG_INSTRUCTION, _fmt_grounding, _fmt_exemplars


# --------------------------------------------------------------------------- #
#  1) توليد دفعة أسئلة
# --------------------------------------------------------------------------- #
BATCH_GENERATOR_SYSTEM = """\
You are an expert university exam-question author for Informatics Engineering
courses. You imitate a specific professor's style and generate MULTIPLE
multiple-choice questions in a single response.

MANDATORY RULES
1. GROUNDING: rely ONLY on the supplied reference material. Never invent
   facts, formulas, or definitions beyond it.
2. STYLE: imitate the style exemplars in phrasing, notation, option format,
   and distractor patterns. {lang_instruction}
3. INDEPENDENCE: each question must be self-contained and must NOT overlap,
   repeat, or contradict any other question in this batch.
4. OPTIONS: exactly one correct option per question. Every distractor must be
   genuinely wrong and model a plausible student error.
5. For computable questions include "worked_solution" showing each step.
6. VERIFICATION CODE — read carefully:
   - ONLY IF the correct answer can be COMPUTED from data given in the
     question (arithmetic, fuzzy operations, algorithm traces, code output,
     probability, etc.): include "verification_code" — a SELF-CONTAINED
     Python program that computes the answer from the question's data and
     ends with exactly:  print(json.dumps({{"answer": <result>}}))
     The answer must be the computed VALUE (number/list/string of the
     result), NEVER an option letter like "B".
     Allowed imports only: math, itertools, fractions, statistics,
     collections, functools, decimal, json. No files, no network, no input().
     The code must genuinely COMPUTE the answer from the given data — never
     print a hard-coded constant.
   - IF the question is conceptual (define, compare, identify the true/false
     statement, theory): set "verification_code": null. Do NOT fabricate
     code for conceptual questions.
7. OUTPUT: a single valid JSON object, nothing outside it.
"""

BATCH_GENERATOR_USER = """\
### Course / المادة
{course_description}

### Professor style summary
- Preferred action verbs: {action_verbs}
- Distractor patterns: {distractor_patterns}
- Target reasoning steps ≈ {reasoning_steps}

### Reference material (grounding — use EXCLUSIVELY)
{grounding}

### Style exemplars (imitate FORM, not content)
{exemplars}

### Questions to generate in THIS batch ({n} questions)
{slot_specs}

### Output EXACTLY this JSON shape:
{{
  "questions": [
    {{
      "slot_index": <int from the specs above>,
      "question": "<question text in the professor's style>",
      "options": [
        {{"label": "A", "text": "...", "is_correct": false}},
        {{"label": "B", "text": "...", "is_correct": true}},
        {{"label": "C", "text": "...", "is_correct": false}},
        {{"label": "D", "text": "...", "is_correct": false}},
        {{"label": "E", "text": "...", "is_correct": false}}
      ],
      "correct_answer": "B",
      "worked_solution": "<step-by-step justification>",
      "verification_code": "<Python per rule 6, or null for conceptual>",
      "provenance_chunk_ids": ["<chunk_id used>"]
    }}
  ]
}}
Generate ALL {n} questions. Every "slot_index" from the specs must appear once.
"""


def build_batch_generation_messages(slots, grounding, exemplars,
                                    style_profile, course_description=""):
    lang = resolve_language(style_profile)
    verbs = ", ".join(style_profile.get("top_action_verbs", [])[:6])
    dpats = "; ".join(p["pattern"] for p in
                      style_profile.get("distractor_patterns", []))
    steps = round(style_profile.get("avg_reasoning_steps", 3))
    if not course_description:
        course_description = ("Inferred from most-tested topics: "
                              + ", ".join(style_profile.get("most_tested_topics", [])[:6]))

    spec_lines = []
    for s in slots:
        spec_lines.append(f"- slot_index={s.index} | topic: {s.topic} | "
                          f"Bloom: {s.cognitive_level} | difficulty: {s.difficulty}")

    # مطابقة الطول: أسئلة الدكتور مقتضبة (قيست فجوة 14.5 مقابل 30.3 كلمة
    # في تقرير مطابقة فعلي) — نقيس متوسط طول الأمثلة ونفرضه صراحة.
    ex_lens = [len(str(q.get("question", "")).split()) for q in exemplars]
    avg_len = round(sum(ex_lens) / len(ex_lens)) if ex_lens else 15
    length_rule = (f"\n7. LENGTH: the professor writes CONCISE stems — the "
                   f"exemplars average ~{avg_len} words. Keep each question "
                   f"stem within ~{avg_len + 6} words. State only the data "
                   f"and the ask; no scene-setting or restating definitions.")

    system = (BATCH_GENERATOR_SYSTEM.format(
        lang_instruction=_LANG_INSTRUCTION[lang]) + length_rule)
    user = BATCH_GENERATOR_USER.format(
        course_description=course_description,
        action_verbs=verbs or "(none listed)",
        distractor_patterns=dpats or "(none listed)",
        reasoning_steps=steps,
        grounding=_fmt_grounding(grounding),
        exemplars=_fmt_exemplars(exemplars),
        n=len(slots),
        slot_specs="\n".join(spec_lines),
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
#  2) حلّ دفعة بنموذج مختلف (Cross-Model Solver)
# --------------------------------------------------------------------------- #
BATCH_SOLVER_SYSTEM = """\
You are an independent expert exam solver. You will receive several
multiple-choice questions WITHOUT the answer keys. Solve EACH question from
scratch, independently, in whatever language it uses. Show brief reasoning,
then commit to exactly one label per question.
Output a single valid JSON object, nothing outside it.
"""

BATCH_SOLVER_USER = """\
Solve all {n} questions below. حُلّ كل الأسئلة بنفسك ثم اختر.

{questions_block}

### Output EXACTLY this JSON shape:
{{
  "solutions": [
    {{"question_id": "<id>", "chosen": "<label>",
      "reasoning": "<concise solution>", "confidence": 0.0}}
  ]
}}
Every question_id above must appear exactly once.
"""


def build_batch_solver_messages(questions: List[dict]):
    blocks = []
    for q in questions:
        opts = "\n".join(f"   {o['label']}) {o['text']}" for o in q["options"])
        blocks.append(f"[question_id: {q['question_id']}]\n{q['question']}\n{opts}")
    user = BATCH_SOLVER_USER.format(n=len(questions),
                                    questions_block="\n\n".join(blocks))
    return [{"role": "system", "content": BATCH_SOLVER_SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
#  3) إصلاح دفعة الأسئلة الفاشلة
# --------------------------------------------------------------------------- #
BATCH_REPAIR_SYSTEM = """\
You are an exam-question editor. You will receive questions that FAILED
quality checks, each with the exact failure reason. Repair each question with
the MINIMAL change that fixes the stated problem while preserving the topic,
difficulty, style, and language. Do not change what is not broken.

VERIFICATION CODE CONTRACT (applies to every repaired question):
- IF the correct answer can be COMPUTED from data given in the question:
  include "verification_code" — a SELF-CONTAINED Python program that computes
  the answer from the question's data and ends with EXACTLY this line:
      print(json.dumps({"answer": <result>}))
  The answer must be the computed VALUE (number/list/string of the result),
  NEVER an option letter like "B".
  Allowed imports only: math, itertools, fractions, statistics, collections,
  functools, decimal, json. No files, no network, no input(). The code must
  genuinely COMPUTE the answer — never print a hard-coded constant.
- IF the question is conceptual: set "verification_code": null.

COMPLETENESS: return each repaired question as a COMPLETE object — full
question text and ALL options — even the parts you did not change. Never
return a partial object.
Output a single valid JSON object, nothing outside it.
"""

BATCH_REPAIR_USER = """\
Repair the following {n} questions. لكل سؤال، سبب فشله مذكور صراحة.

{failed_block}

### Output EXACTLY this JSON shape (same as generation):
{{
  "questions": [
    {{
      "slot_index": <same slot_index>,
      "question": "...",
      "options": [{{"label": "A", "text": "...", "is_correct": false}}, ...],
      "correct_answer": "<label>",
      "worked_solution": "...",
      "verification_code": "<Python that computes the answer, or null if conceptual>",
      "provenance_chunk_ids": ["..."]
    }}
  ]
}}
"""


def build_batch_repair_messages(failed: List[dict]):
    """failed: قائمة عناصر {question: <dict>, reason: <str>, slot_index: <int>}"""
    blocks = []
    for f in failed:
        q = f["question"]
        opts = "\n".join(f"   {o.get('label','?')}) {o.get('text','?')}"
                         for o in q.get("options", []))
        blocks.append(
            f"[slot_index: {f['slot_index']}]\n"
            f"FAILURE REASON: {f['reason']}\n"
            f"QUESTION: {q.get('question','')}\n{opts}\n"
            f"claimed correct: {q.get('correct_answer')}")
    user = BATCH_REPAIR_USER.format(n=len(failed),
                                    failed_block="\n\n".join(blocks))
    return [{"role": "system", "content": BATCH_REPAIR_SYSTEM},
            {"role": "user", "content": user}]