"""
prompts.py — قوالب الـ prompts (عامة، غير متخصصة بمادة، ثنائية اللغة).

مبادئ التصميم:
  - لا ذكر لمادة محددة: اسم المادة/المجال يُمرَّر من الإعدادات أو يُستنتج
    من بروفايل الأسلوب.
  - اللغة تُقرأ من style_profile["language_style"]: النظام يولّد بالعربية
    أو الإنجليزية أو مختلطاً حسب بصمة الأستاذ الفعلية.
  - التحقق عام: المولّد يُخرج verification_code (Python) لكل سؤال حسابي؛
    وإن كانت المادة لها إضافة مسجّلة يمكنه إخراج computation_spec أيضاً.
"""
from __future__ import annotations
import json
from typing import List


# --------------------------------------------------------------------------- #
#  كشف لغة التوليد من البروفايل
# --------------------------------------------------------------------------- #
def resolve_language(style_profile: dict) -> str:
    """يعيد 'ar' أو 'en' أو 'mixed' اعتماداً على language_style في البروفايل."""
    raw = str(style_profile.get("language_style", "")).lower()
    has_ar = any(w in raw for w in ("عرب", "arabic", "مختلط", "mixed"))
    has_en = any(w in raw for w in ("english", "انجليزي", "إنجليزي", "en",
                                    "مختلط", "mixed"))
    if ("مختلط" in raw) or ("mixed" in raw) or (has_ar and has_en):
        return "mixed"
    if has_ar:
        return "ar"
    return "en"


_LANG_INSTRUCTION = {
    "en": ("Write the question, all options, and the worked solution in "
           "fluent, precise academic English."),
    "ar": ("اكتب السؤال وجميع الخيارات والحل المفصّل بعربية أكاديمية دقيقة."),
    "mixed": ("Mirror the professor's mixed Arabic/English register exactly "
              "as shown in the reference exemplars: Arabic framing with "
              "English technical terms and mathematical notation. "
              "حاكِ الخلط اللغوي كما في الأمثلة تماماً."),
}


# --------------------------------------------------------------------------- #
#  1) المولّد (Generator) — عام لأي مادة
# --------------------------------------------------------------------------- #
GENERATOR_SYSTEM = """\
You are an expert university exam-question author. You imitate the style of a
specific professor for the course described below, and every computable
question you produce must be independently verifiable by code.

MANDATORY RULES
1. GROUNDING: rely ONLY on the supplied "reference material". Never invent
   facts, formulas, or definitions beyond it.
2. STYLE: imitate the "style exemplars" in phrasing, notation, option format,
   and distractor patterns. {lang_instruction}
3. VERIFIABILITY: for every question whose answer can be computed, output
   "verification_code": a SELF-CONTAINED Python program that computes the
   correct answer from the question's given data and ends with exactly:
       print(json.dumps({{"answer": <result>}}))
   Allowed imports only: math, itertools, fractions, statistics, collections,
   functools, decimal, json. No files, no network, no input().
   {domain_hint}
   If the question is purely conceptual (not computable), set
   verification_code = null AND computation_spec = null.
4. OPTIONS: exactly one correct option. Every distractor must be genuinely
   wrong and represent a plausible student error.
5. OUTPUT: valid JSON only. No text outside the JSON object.
"""

GENERATOR_USER_TEMPLATE = """\
### Course / المادة
{course_description}

### Slot requirements / متطلبات الخانة
- Topic: {topic}
- Bloom cognitive level: {cognitive_level}
- Difficulty: {difficulty}
- Target reasoning steps ≈ {reasoning_steps}
- Professor's preferred action verbs: {action_verbs}
- Professor's distractor patterns: {distractor_patterns}

### Reference material (grounding — use EXCLUSIVELY)
{grounding}

### Style exemplars (imitate FORM, not content)
{exemplars}

### Output EXACTLY this JSON shape:
{{
  "question": "<question text in the professor's style>",
  "options": [
    {{"label": "A", "text": "...", "is_correct": false}},
    {{"label": "B", "text": "...", "is_correct": false}},
    {{"label": "C", "text": "...", "is_correct": true}},
    {{"label": "D", "text": "...", "is_correct": false}},
    {{"label": "E", "text": "...", "is_correct": false}}
  ],
  "correct_answer": "C",
  "worked_solution": "<step-by-step justification>",
  "verification_code": "<self-contained Python per rule 3, or null>",
  "computation_spec": <domain spec object, or null>,
  "provenance_chunk_ids": ["<chunk_id used>"],
  "distractor_rationale": "<why each distractor is wrong and which common error it models>"
}}
"""


def _fmt_grounding(chunks: List[dict]) -> str:
    out = []
    for c in chunks:
        out.append(f"- [{c.get('chunk_id')}] ({c.get('chunk_type')}) "
                   f"{c.get('title') or ''}: {c.get('content') or c.get('raw_text','')}")
    return "\n".join(out) if out else "(none)"


def _fmt_exemplars(qs: List[dict]) -> str:
    out = []
    for q in qs:
        opts = " | ".join(f"{o['label']}: {o['text']}" for o in q.get("options", []))
        out.append(f"- Q: {q['question']}\n  Options: {opts}\n"
                   f"  Correct: {q.get('correct_answer')} | Topic: {q['topic']}")
    return "\n".join(out) if out else "(none)"


def build_generator_messages(slot, grounding, exemplars, style_profile,
                             course_description: str = "",
                             domain: str | None = None):
    lang = resolve_language(style_profile)
    verbs = ", ".join(style_profile.get("top_action_verbs", [])[:6])
    dpats = "; ".join(p["pattern"] for p in style_profile.get("distractor_patterns", []))
    avg_steps = style_profile.get("avg_reasoning_steps", 3)

    domain_hint = ""
    if domain:
        domain_hint = (f'A verified domain plugin "{domain}" exists: you MAY also '
                       f'output "computation_spec" with "domain": "{domain}" for '
                       f'higher-precision checking, in addition to verification_code.')

    if not course_description:
        course_description = ("Inferred from most-tested topics: "
                              + ", ".join(style_profile.get("most_tested_topics", [])[:6]))

    system = GENERATOR_SYSTEM.format(
        lang_instruction=_LANG_INSTRUCTION[lang], domain_hint=domain_hint)
    user = GENERATOR_USER_TEMPLATE.format(
        course_description=course_description,
        topic=slot.topic, cognitive_level=slot.cognitive_level,
        difficulty=slot.difficulty, reasoning_steps=round(avg_steps),
        action_verbs=verbs or "(none listed)",
        distractor_patterns=dpats or "(none listed)",
        grounding=_fmt_grounding(grounding), exemplars=_fmt_exemplars(exemplars),
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
#  2) المُحقّق مغلق العينين (Blind Verifier) — ثنائي اللغة
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM = """\
You are an independent expert grader. You will be given a multiple-choice
question WITHOUT knowing the intended key. Solve it from scratch yourself,
in whatever language the question uses, then pick the label you believe is
correct. Output JSON only:
{"chosen": "<label>", "reasoning": "<your concise solution>", "confidence": 0.0-1.0}
"""


def build_verifier_messages(question_text: str, options: List[dict]):
    opts = "\n".join(f"{o['label']}) {o['text']}" for o in options)
    user = (f"Question:\n{question_text}\n\nOptions:\n{opts}\n\n"
            "Solve it yourself, then choose. حُلّه بنفسك ثم اختر.")
    return [{"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user}]
