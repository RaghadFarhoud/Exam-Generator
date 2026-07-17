"""
audit_generated_exam.py — تدقيق لاحق لامتحان مولَّد موجود.

الاستخدام:
    python3 audit_generated_exam.py exam.json

يفحص كل سؤال:
  1) بنيوياً (خيار صحيح واحد، اتساق المفتاح).
  2) حسابياً إن كان يحمل verification_code (الامتحانات الجديدة) —
     ينفّذ الكود ويقارن.
  3) للامتحانات القديمة بلا كود: يطبع الأسئلة الحسابية المشتبهة
     (تحوي أرقاماً وعمليات) كقائمة مراجعة يدوية مركزة، بدل ادعاء
     تدقيق آلي لا نملك أساسه.

المخرج: تقرير JSON بجانب الملف + ملخص على الشاشة.
"""
import json
import os
import re
import sys

from exam_gen import code_verifier as cv


def audit(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    qs = d.get("questions", [])
    report = {"file": path, "n_questions": len(qs),
              "structural_issues": [], "code_checked": 0,
              "code_agreed": 0, "code_mismatch": [],
              "code_exec_failed": [], "no_code_math_suspects": []}

    for q in qs:
        qid = q.get("question_id")
        opts = q.get("options", [])
        # 1) بنيوي
        nc = sum(1 for o in opts if o.get("is_correct"))
        lab = next((o.get("label") for o in opts if o.get("is_correct")), None)
        if nc != 1:
            report["structural_issues"].append({"id": qid, "issue": f"n_correct={nc}"})
        elif lab != q.get("correct_answer"):
            report["structural_issues"].append({"id": qid, "issue": "key/label mismatch"})

        # 2) حسابي عبر الكود إن وُجد
        code = q.get("verification_code")
        if code and str(code).strip() and str(code).lower() != "none":
            report["code_checked"] += 1
            ok, computed, detail = cv.run_verification_code(str(code))
            if not ok:
                report["code_exec_failed"].append({"id": qid, "detail": detail[:150]})
                continue
            correct = next((o for o in opts if o.get("is_correct")), None)
            if correct and cv.generic_answers_match(computed, correct["text"]):
                report["code_agreed"] += 1
            else:
                match = next((o.get("label") for o in opts
                              if cv.generic_answers_match(computed, o.get("text", ""))),
                             None)
                report["code_mismatch"].append({
                    "id": qid, "computed": str(computed),
                    "claimed": correct.get("text") if correct else None,
                    "matches_other_option": match,
                    "question": q.get("question", "")[:120]})
        else:
            # 3) مشتبه حسابي بلا كود (امتحانات قديمة)
            text = q.get("question", "")
            if re.search(r"\d", text) and re.search(
                    r"[=+\-*/∪∩⊕]|احسب|ناتج|قيمة|compute|calculate", text):
                report["no_code_math_suspects"].append(
                    {"id": qid, "question": text[:120]})

    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    rep = audit(sys.argv[1])
    base = os.path.basename(sys.argv[1]).rsplit(".", 1)[0]
    out = base + "_audit.json"          # في مجلد التشغيل (المدخل قد يكون للقراءة فقط)
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"أسئلة: {rep['n_questions']} | مشاكل بنيوية: {len(rep['structural_issues'])}")
    print(f"مدقَّق بالكود: {rep['code_checked']} | متفق: {rep['code_agreed']} "
          f"| متعارض: {len(rep['code_mismatch'])} | فشل تنفيذ: {len(rep['code_exec_failed'])}")
    if rep["code_mismatch"]:
        print("\n⚠ مفاتيح متعارضة مع الحساب الآلي:")
        for m in rep["code_mismatch"]:
            hint = (f" (الناتج يطابق الخيار {m['matches_other_option']}!)"
                    if m["matches_other_option"] else "")
            print(f"  - {m['id']}: محسوب={m['computed']} مقابل مُدّعى={m['claimed']}{hint}")
    if rep["no_code_math_suspects"]:
        print(f"\nقائمة مراجعة يدوية (أسئلة حسابية بلا كود): "
              f"{len(rep['no_code_math_suspects'])} سؤالاً — التفاصيل في {out}")
    print(f"\nالتقرير الكامل: {out}")
