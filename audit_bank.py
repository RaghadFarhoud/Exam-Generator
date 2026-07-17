"""
audit_bank.py — يدقّق بنك الأسئلة الأصلي.

لكل سؤال يحوي computation_spec (أو يمكن استنتاجه)، يعيد المحرك حساب
الإجابة ويقارنها بـ correct_answer المخزّن، ويطبع التعارضات.

ملاحظة: بنك أسئلتك الحالي لا يحتوي computation_spec جاهزاً، لذا هذا السكربت
يوضّح آلية التدقيق على أمثلة يدوية مستخرجة من الأسئلة الحسابية. عند إثراء
البنك بـ computation_spec (عبر المولّد أو يدوياً) سيدقّقه آلياً بالكامل.
"""
import json
from exam_gen import fuzzy_math as fm

# أمثلة تدقيق مبنية على أسئلة حسابية من بنكك (مستخرجة يدوياً كبرهان مفهوم)
AUDIT_CASES = [
    {
        "qid": "q_001 (A+A)(7) extension principle",
        "spec": {
            "sets": {"A": "0.3/2 + 0.4/3 + 0.2/4 + 0.1/5"},
            "operation": "extension_binary",
            "params": {"a": "A", "b": "A", "binop": "add", "at": 7},
        },
        "stored_correct_text": "0.6",   # الخيار C المخزّن كصحيح
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print("تدقيق مفاتيح الإجابة في البنك الأصلي")
    print("=" * 60)
    problems = 0
    for case in AUDIT_CASES:
        computed, label = fm.evaluate_spec(case["spec"])
        ok = fm.answers_match(computed, case["stored_correct_text"])
        flag = "✓ متطابق" if ok else "✗ تعارض — مفتاح الإجابة مشكوك فيه"
        if not ok:
            problems += 1
        print(f"\n[{case['qid']}]")
        print(f"  المحسوب: {label} = {computed}")
        print(f"  المخزّن كصحيح: {case['stored_correct_text']}")
        print(f"  النتيجة: {flag}")
    print("\n" + "-" * 60)
    print(f"عدد التعارضات المكتشفة: {problems} / {len(AUDIT_CASES)}")
    print("الخلاصة: التحقق الحسابي يمسك أخطاء المفاتيح قبل أن تصل للطالب.")
