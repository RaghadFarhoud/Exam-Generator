"""
evaluate.py — وحدة التقييم الأكاديمي (فصل النتائج في مشروعك).

ثلاثة محاور مستقلة:
  1) style_fidelity()    — مطابقة الأسلوب كمياً: مقارنة توزيعات
                           (Bloom × difficulty × طول السؤال × عدد الخيارات)
                           بين المولّد والبنك الأصلي بمقياس Jensen-Shannon.
  2) build_blind_test()  — يولّد مواد اختبار التمييز الأعمى: خليط مُعمّى
                           من أسئلة حقيقية ومولّدة + مفتاح سري للتحليل،
                           ودالة لتحليل ردود المشاركين.
  3) run_ablation()      — دراسة الحذف: يقيس نسبة المفاتيح الصحيحة
                           حسابياً مع/بدون طبقة التحقق، ليثبت أثر مساهمتك.
"""
from __future__ import annotations
import json
import math
import random
import statistics
from collections import Counter
from typing import Dict, List, Optional

from . import fuzzy_math as fm


# --------------------------------------------------------------------------- #
#  أدوات إحصائية
# --------------------------------------------------------------------------- #
def _norm(counter: Dict[str, int], keys) -> List[float]:
    total = sum(counter.get(k, 0) for k in keys) or 1
    return [counter.get(k, 0) / total for k in keys]


def js_divergence(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence (بالـ bits). 0 = تطابق تام، 1 = تباعد أقصى."""
    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    m = [(x + y) / 2 for x, y in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _dist_of(questions: List[dict], getter) -> Counter:
    return Counter(getter(q) for q in questions if getter(q) is not None)


# --------------------------------------------------------------------------- #
#  1) مطابقة الأسلوب كمياً
# --------------------------------------------------------------------------- #
def style_fidelity(original: List[dict], generated: List[dict]) -> dict:
    """
    يقارن البصمة الإحصائية للامتحان المولّد ببنك الدكتور الأصلي.
    JSD قريب من 0 => الشكل الإحصائي متطابق.
    """
    report = {}

    axes = {
        "cognitive_level": lambda q: q.get("academic", {}).get("cognitive_level"),
        "difficulty":      lambda q: q.get("academic", {}).get("difficulty"),
        "n_options":       lambda q: len(q.get("options", [])) or None,
    }
    for name, getter in axes.items():
        c1, c2 = _dist_of(original, getter), _dist_of(generated, getter)
        keys = sorted(set(c1) | set(c2), key=str)
        jsd = js_divergence(_norm(c1, keys), _norm(c2, keys))
        report[name] = {
            "original": {str(k): c1.get(k, 0) for k in keys},
            "generated": {str(k): c2.get(k, 0) for k in keys},
            "js_divergence": round(jsd, 4),
        }

    # طول السؤال (بالكلمات) — مقارنة متوسط وانحراف
    def lens(qs):
        return [len(str(q.get("question", "")).split()) for q in qs] or [0]
    lo, lg = lens(original), lens(generated)
    report["question_length_words"] = {
        "original_mean": round(statistics.mean(lo), 1),
        "generated_mean": round(statistics.mean(lg), 1),
        "original_stdev": round(statistics.pstdev(lo), 1),
        "generated_stdev": round(statistics.pstdev(lg), 1),
    }

    # نسبة الأسئلة المعتمدة على صيغة (references_formula)
    def frac_formula(qs):
        vals = [bool(q.get("references_formula")) for q in qs]
        return round(sum(vals) / len(vals), 3) if vals else 0.0
    report["formula_based_fraction"] = {
        "original": frac_formula(original), "generated": frac_formula(generated),
    }

    jsds = [v["js_divergence"] for k, v in report.items() if isinstance(v, dict)
            and "js_divergence" in v]
    report["overall_mean_jsd"] = round(sum(jsds) / len(jsds), 4) if jsds else None
    return report


# --------------------------------------------------------------------------- #
#  2) اختبار التمييز الأعمى
# --------------------------------------------------------------------------- #
_CONTEXT_MARKERS = ("تتمة", "السؤال السابق", "أعلاه", "في الشكل التالي",
                    "المخطط التالي", "the previous question", "figure below")


def _blind_test_eligible(q: dict) -> bool:
    """
    فلتر جودة لمواد التمييز الأعمى: يستبعد أسئلة البنك ذات عيوب الاستخراج
    التي تكشف الفئة دون أي حسّ بالأسلوب (خيارات مفقودة/فارغة، نص قصير
    مبتور، أسئلة معتمدة على سياق غير معروض كالمخططات أو 'تتمة السؤال
    السابق'). بدون هذا الفلتر تقيس التجربة جودة الاستخراج لا الأسلوب.
    """
    opts = q.get("options") or []
    if len(opts) < 3:
        return False
    if any(not str(o.get("text", "")).strip() for o in opts):
        return False
    qtext = str(q.get("question", "")).strip()
    if len(qtext.split()) < 4:
        return False
    if q.get("references_figure"):
        return False
    low = qtext.lower()
    if any(m in low or m in qtext for m in _CONTEXT_MARKERS):
        return False
    return True


def build_blind_test(original: List[dict], generated: List[dict],
                     n_each: int = 10, seed: int = 42) -> dict:
    """
    يبني مواد التجربة:
      - participant_sheet: قائمة أسئلة مخلوطة ومرقّمة دون أي وسم،
        يُطلب من المشارك تصنيف كل سؤال: "حقيقي" أم "مولّد".
      - secret_key: المفتاح (يبقى عند الباحث فقط).
    يُطبَّق فلتر جودة على الطرفين لإزالة البصمات المسرِّبة غير الأسلوبية.
    التحليل لاحقاً بـ analyze_blind_responses().
    """
    rng = random.Random(seed)
    orig_pool = [q for q in original if _blind_test_eligible(q)]
    gen_pool = [q for q in generated if _blind_test_eligible(q)]
    n = min(n_each, len(orig_pool), len(gen_pool))
    o = rng.sample(orig_pool, n)
    g = rng.sample(gen_pool, n)
    def _fmt_opts(q):
        # فرز بالتسمية: بعض أسئلة البنك مخزّنة بترتيب معكوس (E..A) —
        # بصمة استخراج تكشف الفئة، نوحّد العرض A..E للطرفين
        return [f"{x['label']}) {x['text']}"
                for x in sorted(q.get("options", []),
                                key=lambda o: str(o.get("label", "")))]

    items = [{"question": q["question"], "options": _fmt_opts(q),
              "_truth": "real"} for q in o]
    items += [{"question": q["question"], "options": _fmt_opts(q),
               "_truth": "generated"} for q in g]
    rng.shuffle(items)

    sheet, key = [], {}
    for i, it in enumerate(items, 1):
        sheet.append({"item": i, "question": it["question"], "options": it["options"]})
        key[str(i)] = it["_truth"]
    return {
        "participant_sheet": sheet,
        "secret_key": key,
        "instructions_ar": (
            "لكل سؤال أدناه، حدّد: هل تعتقد أنه من امتحان حقيقي للدكتور (real) "
            "أم مولّد آلياً (generated)؟ لا توجد إجابة محايدة."),
    }


def analyze_blind_responses(secret_key: Dict[str, str],
                            responses: List[Dict[str, str]]) -> dict:
    """
    responses: قائمة، كل عنصر = ردود مشارك واحد {item_id: "real"|"generated"}.
    الناتج: دقة التمييز الكلية + لكل مشارك، مع اختبار ثنائي تقريبي ضد 50٪.
    دقة ≈ 50٪ (صدفة) => التقليد ناجح: البشر لا يميّزون المولّد من الحقيقي.
    """
    per = []
    all_correct = all_n = 0
    for r in responses:
        c = sum(1 for k, v in r.items() if secret_key.get(str(k)) == v)
        n = len(r)
        per.append({"correct": c, "n": n, "accuracy": round(c / n, 3) if n else None})
        all_correct += c
        all_n += n
    acc = all_correct / all_n if all_n else None
    # اختبار ثنائي تقريبي (normal approximation) ضد p=0.5
    z = None
    if all_n:
        se = math.sqrt(0.25 / all_n)
        z = round((acc - 0.5) / se, 3)
    return {"overall_accuracy": round(acc, 3) if acc is not None else None,
            "n_judgments": all_n, "z_vs_chance": z,
            "interpretation": ("قريب من مستوى الصدفة → تقليد ناجح"
                               if acc is not None and abs(acc - 0.5) < 0.1
                               else "المشاركون يميّزون المولّد → الأسلوب يحتاج تحسيناً"),
            "per_participant": per}


# --------------------------------------------------------------------------- #
#  3) دراسة الحذف (Ablation)
# --------------------------------------------------------------------------- #
def computational_key_accuracy(questions: List[dict]) -> dict:
    """
    لكل سؤال يحمل computation_spec: يعيد المحرك حساب الإجابة ويقارن بالمفتاح.
    يُستخدم لقياس دقة المفاتيح في أي دفعة (raw مقابل verified).
    """
    checkable = correct = 0
    failures = []
    for q in questions:
        spec = q.get("computation_spec")
        if not spec:
            continue
        checkable += 1
        try:
            computed, _ = fm.evaluate_spec(spec)
            copt = next((o for o in q.get("options", []) if o.get("is_correct")), None)
            if copt and fm.answers_match(computed, copt["text"]):
                correct += 1
            else:
                failures.append({"question_id": q.get("question_id"),
                                 "computed": repr(computed),
                                 "claimed": copt["text"] if copt else None})
        except Exception as e:
            failures.append({"question_id": q.get("question_id"), "error": str(e)})
    return {"checkable": checkable, "correct_keys": correct,
            "key_accuracy": round(correct / checkable, 3) if checkable else None,
            "failures": failures}


def run_ablation(pipeline, n_questions: int, seed: int = 0) -> dict:
    """
    الشرط A (baseline): توليد خام — نقبل كل ما يخرجه المولّد دون أي تحقق.
    الشرط B (full):     الـ pipeline الكامل بكل طبقات التحقق.
    المقياس: دقة مفاتيح الأسئلة الحسابية في مخرجات كل شرط.
    الفرق (B - A) = الأثر المقاس لمساهمتك.
    """
    from .blueprint import build_blueprint
    from . import prompts as P

    # --- A: خام بدون تحقق --------------------------------------------------
    bp = build_blueprint(pipeline.corpus, n_questions, seed)
    raw_batch = []
    for slot in bp:
        try:
            q = pipeline._generate_one(slot, temperature=0.7)
            raw_batch.append(q)
        except Exception:
            pass
    a = computational_key_accuracy(raw_batch)

    # --- B: كامل ------------------------------------------------------------
    res = pipeline.generate_exam(n_questions, seed=seed, verbose=False)
    b = computational_key_accuracy(res["verified"])

    return {
        "condition_A_raw": {**a, "n_generated": len(raw_batch)},
        "condition_B_full_pipeline": {**b, "counts": res["counts"]},
        "delta_key_accuracy": (round(b["key_accuracy"] - a["key_accuracy"], 3)
                               if a["key_accuracy"] is not None
                               and b["key_accuracy"] is not None else None),
        "note": ("condition B لا يمكن نظرياً أن يحتوي مفتاحاً حسابياً خاطئاً، "
                 "لأن طبقة التحقق ترفضه قبل القبول — الفرق يقيس أثر المساهمة."),
    }