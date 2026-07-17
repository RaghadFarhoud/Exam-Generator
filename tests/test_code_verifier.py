"""tests/test_code_verifier.py — يثبت أن التحقق العام يعمل لمواد مختلفة."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import code_verifier as cv


def test_numeric_answer_statistics_course():
    """مادة إحصاء: متوسط عيّنة — لا علاقة بالمنطق الصبابي."""
    code = (
        "import json, statistics\n"
        "data = [4, 8, 6, 5, 3, 7]\n"
        "print(json.dumps({'answer': statistics.mean(data)}))\n"
    )
    ok, ans, d = cv.run_verification_code(code)
    assert ok, d
    assert cv.generic_answers_match(ans, "5.5")
    assert not cv.generic_answers_match(ans, "6.5")


def test_list_answer_algorithms_course():
    """مادة خوارزميات: ترتيب — ناتج قائمة."""
    code = (
        "import json\n"
        "arr = [5, 2, 9, 1]\n"
        "print(json.dumps({'answer': sorted(arr)}))\n"
    )
    ok, ans, d = cv.run_verification_code(code)
    assert ok, d
    assert cv.generic_answers_match(ans, "[1, 2, 5, 9]")
    assert cv.generic_answers_match(ans, "{1, 2, 5, 9}")     # تدوين مجموعة
    assert not cv.generic_answers_match(ans, "[1, 2, 5]")


def test_dict_answer_fuzzy_notation():
    """ناتج قاموس يقارَن بتدوين a/x + b/y (يخدم الصبابية وغيرها)."""
    code = (
        "import json\n"
        "mu = {1: 0.9, 2: 0.5, 4: 0.7}\n"
        "print(json.dumps({'answer': mu}))\n"
    )
    ok, ans, d = cv.run_verification_code(code)
    assert ok, d
    assert cv.generic_answers_match(ans, "0.9/1 + 0.5/2 + 0.7/4")
    assert not cv.generic_answers_match(ans, "0.9/1 + 0.5/2 + 0.6/4")


def test_english_boolean_answer():
    code = (
        "import json\n"
        "print(json.dumps({'answer': 3**2 + 4**2 == 5**2}))\n"
    )
    ok, ans, d = cv.run_verification_code(code)
    assert ok, d
    assert cv.generic_answers_match(ans, "True")
    assert cv.generic_answers_match(ans, "صحيح")


def test_static_check_blocks_forbidden():
    bad = "import os\nprint(os.listdir('/'))"
    ok, ans, d = cv.run_verification_code(bad)
    assert not ok and "غير مسموح" in d or "محظور" in d


def test_timeout_enforced():
    code = "import json\nwhile True: pass\nprint(json.dumps({'answer': 1}))"
    ok, ans, d = cv.run_verification_code(code, timeout=2)
    assert not ok and "المهلة" in d


def test_contract_violation_detected():
    code = "x = 1 + 1"                 # لا يطبع شيئاً
    ok, ans, d = cv.run_verification_code(code)
    assert not ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
