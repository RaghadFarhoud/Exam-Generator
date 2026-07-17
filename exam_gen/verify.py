"""
verify.py — بوابة الجودة متعددة الطبقات (أعلى رافعة دقة).

طبقات القبول لأي سؤال مولّد:
  L1  بنية سليمة (خيار صحيح واحد، لا تكرار، ...)
  L2  تحقق حسابي: إعادة حساب computation_spec ومطابقة الخيار الصحيح المُدّعى.
      (وأيضاً: التأكد أن كل مشتّت خاطئ فعلاً عند القابلية).
  L3  محقّق مغلق العينين (LLM مستقل يحلّ ويوافق).
  L4  فحص تكرار ضد بنك الأسئلة الأصلي.

القرار النهائي: verified / needs_review / rejected.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from . import fuzzy_math as fm
from . import code_verifier as cv
from . import domains


# --------------------------------------------------------------------------- #
#  L2 — تحقق حسابي عام (plugin أولاً، ثم تنفيذ كود)
# --------------------------------------------------------------------------- #
def _compute_via_spec(spec: dict):
    """يشغّل computation_spec عبر إضافة المادة المسجّلة إن وُجدت."""
    domain = spec.get("domain", "fuzzy_logic")   # افتراضي للتوافق الخلفي
    ev = domains.get_domain(domain)
    if ev is None:
        raise ValueError(f"لا توجد إضافة مسجّلة للمادة: {domain}")
    return ev(spec)


def check_computation_generic(q: dict) -> "VerdictLayer | None":
    """
    يعيد None إن كان السؤال مفاهيمياً (لا spec ولا كود).
    الأولوية: computation_spec (إضافة مُراجعة) ثم verification_code (عام).
    """
    spec = q.get("computation_spec")
    code = q.get("verification_code")
    if not spec and not code:
        return None

    computed = None
    label = ""
    # 1) إضافة المادة إن وُجدت
    if spec:
        try:
            computed, label = _compute_via_spec(spec)
        except Exception as e:
            if not code:
                return VerdictLayer("computation", False,
                                    f"فشل spec ولا يوجد كود بديل: {e}")
    # 2) تنفيذ الكود العام
    if computed is None and code:
        ok, ans, detail = cv.run_verification_code(code)
        if not ok:
            return VerdictLayer("computation", False,
                                f"فشل تنفيذ verification_code: {detail}")
        computed, label = ans, "verification_code"

    correct_opt = next((o for o in q["options"] if o.get("is_correct")), None)
    if not correct_opt:
        return VerdictLayer("computation", False, "لا يوجد خيار صحيح")

    match = (fm.answers_match(computed, correct_opt["text"])
             if spec and not isinstance(computed, (dict, bool))
             else cv.generic_answers_match(computed, correct_opt["text"]))
    if not match:
        return VerdictLayer("computation", False,
                            f"المحسوب ({label}) = {computed!r} لا يطابق "
                            f"الخيار الصحيح المُدّعى {correct_opt['text']!r}")

    for o in q["options"]:
        if o.get("is_correct"):
            continue
        also = (fm.answers_match(computed, o["text"])
                if spec and not isinstance(computed, (dict, bool))
                else cv.generic_answers_match(computed, o["text"]))
        if also:
            return VerdictLayer("computation", False,
                                f"مشتّت {o['label']} مطابق للناتج الصحيح — إجابتان صحيحتان")
    return VerdictLayer("computation", True, f"{label} = {computed!r} ✓")


# اسم قديم للتوافق
def check_computation(q: dict):
    return check_computation_generic(q)


@dataclass
class VerdictLayer:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Verdict:
    status: str                       # verified | needs_review | rejected
    layers: List[VerdictLayer] = field(default_factory=list)

    def add(self, name, passed, detail=""):
        self.layers.append(VerdictLayer(name, passed, detail))

    def as_dict(self):
        return {"status": self.status,
                "layers": [vars(l) for l in self.layers]}


# --------------------------------------------------------------------------- #
#  L1 — بنية
# --------------------------------------------------------------------------- #
_TEXT_ALIASES = ("text", "option_text", "content", "value", "answer", "choice")
_DEFAULT_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _normalize_options(q: dict) -> Optional[str]:
    """
    يطبّع q['options'] في مكانه ليتحمّل اختلافات شكل مخرجات LLM
    (أسماء حقول بديلة، تسميات مفقودة، ...). يعيد رسالة خطأ إن تعذّر
    التطبيع، أو None عند النجاح.
    """
    opts = q.get("options")
    if not isinstance(opts, list) or len(opts) < 3:
        return f"options مفقودة/غير صالحة أو عددها غير كافٍ ({opts!r})"

    normalized = []
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            return f"عنصر الخيار عند الفهرس {i} ليس كائناً صالحاً: {o!r}"
        text = None
        for key in _TEXT_ALIASES:
            if key in o and o[key] not in (None, ""):
                text = o[key]
                break
        if text is None:
            return f"الخيار عند الفهرس {i} لا يحوي أي حقل نصّي معروف: {list(o.keys())}"
        label = o.get("label")
        if not label:
            label = _DEFAULT_LABELS[i] if i < len(_DEFAULT_LABELS) else str(i)
        normalized.append({"label": str(label).strip(),
                           "text": text,
                           "is_correct": bool(o.get("is_correct", False))})
    q["options"] = normalized

    # اضبط correct_answer تلقائياً إن غاب أو لم يطابق أي is_correct
    correct_labels = [o["label"] for o in normalized if o["is_correct"]]
    if len(correct_labels) == 1 and q.get("correct_answer") != correct_labels[0]:
        q["correct_answer"] = correct_labels[0]
    return None


def check_structure(q: dict) -> VerdictLayer:
    try:
        err = _normalize_options(q)
        if err:
            return VerdictLayer("structure", False, err)
        opts = q["options"]

        labels = [o["label"] for o in opts]
        if len(set(labels)) != len(labels):
            return VerdictLayer("structure", False, "تسميات مكرّرة")
        texts = [str(o["text"]).strip() for o in opts]
        if len(set(texts)) != len(texts):
            return VerdictLayer("structure", False, "خيارات نصّية مكرّرة")
        n_correct = sum(1 for o in opts if o.get("is_correct"))
        if n_correct != 1:
            return VerdictLayer("structure", False, f"عدد الإجابات الصحيحة = {n_correct}")
        correct_label = next(o["label"] for o in opts if o["is_correct"])
        if q.get("correct_answer") != correct_label:
            return VerdictLayer("structure", False, "correct_answer لا يطابق is_correct")
        if not str(q.get("question", "")).strip():
            return VerdictLayer("structure", False, "نص السؤال فارغ")
        return VerdictLayer("structure", True, "OK")
    except Exception as e:                      # شبكة أمان أخيرة: لا تحطّم أبداً
        return VerdictLayer("structure", False, f"فشل فحص البنية بخطأ غير متوقع: {e}")


# --------------------------------------------------------------------------- #
#  L3 — محقّق مغلق العينين
# --------------------------------------------------------------------------- #
def check_blind_solver(q: dict, llm, prompts) -> VerdictLayer:
    msgs = prompts.build_verifier_messages(q["question"], q["options"])
    try:
        out = llm.complete_json(msgs, temperature=0.0)
    except Exception as e:
        return VerdictLayer("blind_solver", False, f"فشل المحقّق: {e}")
    chosen = out.get("chosen")
    ok = chosen == q.get("correct_answer")
    return VerdictLayer("blind_solver", ok,
                        f"اختار {chosen} (ثقة {out.get('confidence')}) "
                        f"{'يوافق' if ok else 'يخالف'}")


# --------------------------------------------------------------------------- #
#  L4 — تكرار
# --------------------------------------------------------------------------- #
def check_novelty(q: dict, retriever, threshold: float = 0.9) -> VerdictLayer:
    sim = retriever.max_similarity_to_bank(q["question"])
    return VerdictLayer("novelty", sim < threshold,
                        f"أقصى تشابه مع البنك = {round(sim,3)}")


# --------------------------------------------------------------------------- #
#  الدمج
# --------------------------------------------------------------------------- #
def verify_question(q: dict, llm, prompts, retriever,
                    require_blind: bool = True) -> Verdict:
    """
    شبكة أمان: أي خطأ غير متوقع (بنية مخرجات LLM غريبة، إلخ) يُحوَّل إلى
    رفض مُسجَّل بدل تحطيم توليد الامتحان بالكامل. سؤال سيّئ واحد يجب ألا
    يوقف بقية الخانات.
    """
    try:
        return _verify_question_inner(q, llm, prompts, retriever, require_blind)
    except Exception as e:
        v = Verdict(status="rejected")
        v.add("unexpected_error", False, f"خطأ غير متوقع أثناء التحقق: {e}")
        return v


def _verify_question_inner(q: dict, llm, prompts, retriever,
                           require_blind: bool = True) -> Verdict:
    v = Verdict(status="verified")

    l1 = check_structure(q)
    v.layers.append(l1)
    if not l1.passed:
        v.status = "rejected"
        return v

    l2 = check_computation(q)
    if l2 is not None:
        v.layers.append(l2)
        if not l2.passed:
            v.status = "rejected"       # خطأ حسابي = رفض قاطع
            return v

    l4 = check_novelty(q, retriever)
    v.layers.append(l4)

    if require_blind:
        l3 = check_blind_solver(q, llm, prompts)
        v.layers.append(l3)
        computational_ok = (l2 is not None and l2.passed)
        if not l3.passed:
            # إن كان السؤال محسوباً وتحقق حسابياً، خلاف المحقّق => مراجعة لا رفض
            v.status = "needs_review" if computational_ok else "rejected"
            return v

    if l2 is None:                      # مفاهيمي: لا تحقق حسابي => مراجعة بشرية
        v.status = "needs_review"
    if not l4.passed:
        v.status = "needs_review"
    return v
