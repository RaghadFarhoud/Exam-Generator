"""
tests/test_robustness.py — يثبت أن مخرجات LLM المشوّهة لا تُحطّم النظام.

هذا الاختبار موجود بسبب عطل حقيقي واجهه مستخدم: Mistral أرجع خياراً
بدون مفتاح "text"، فتحطّم verify_question بـ KeyError. الإصلاح جعل
كل الوصول إلى حقول مخرجات LLM دفاعياً (طبقة أمان + تطبيع + fallback).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import load_corpus, MockClient
from exam_gen.retrieval import Retriever
from exam_gen.verify import verify_question
from exam_gen.pipeline import _to_schema
from exam_gen.blueprint import Slot
from exam_gen import prompts as P

UP = "/mnt/user-data/uploads"
_corpus = None
_retriever = None
_llm = MockClient()
_slot = Slot(index=1, topic="X", cognitive_level="apply", difficulty="medium")


def _setup():
    global _corpus, _retriever
    if _corpus is None:
        _corpus = load_corpus(UP + "/all_exam_questions.json",
                              UP + "/normalized_knowledge.json",
                              UP + "/doctor_style_profile.json")
        _retriever = Retriever(_corpus)
    return _corpus, _retriever


def _check_no_crash(q: dict):
    """يجب أن يعيد قراراً دائماً، ولا يرمي استثناءً أبداً."""
    _, retriever = _setup()
    v = verify_question(q, _llm, P, retriever, require_blind=False)
    assert v.status in ("verified", "needs_review", "rejected")
    record = _to_schema(q, _slot, v, 1)   # يجب ألا يتحطم أيضاً
    assert isinstance(record, dict)
    return v


def test_missing_text_key_exact_bug():
    """العطل الحقيقي الذي أُبلغ عنه: خيار بلا مفتاح 'text'."""
    q = {
        "question": "What is 2+2?",
        "options": [
            {"label": "A", "option_text": "3", "is_correct": False},
            {"label": "B", "content": "4", "is_correct": True},
            {"label": "C", "value": "5", "is_correct": False},
        ],
        "correct_answer": "B",
    }
    v = _check_no_crash(q)
    assert v.status != "rejected" or "text" not in (v.layers[0].detail or "")


def test_missing_options_key():
    _check_no_crash({"question": "broken", "correct_answer": "A"})


def test_options_is_dict_not_list():
    _check_no_crash({"question": "broken2", "options": {"A": "x"}, "correct_answer": "A"})


def test_options_is_string():
    _check_no_crash({"question": "weird", "options": "not a list", "correct_answer": "A"})


def test_missing_question_key():
    _check_no_crash({"options": [{"label": "A", "text": "x", "is_correct": True},
                                 {"label": "B", "text": "y", "is_correct": False},
                                 {"label": "C", "text": "z", "is_correct": False}]})


def test_option_missing_label():
    q = {
        "question": "no labels",
        "options": [{"text": "opt1", "is_correct": False},
                   {"text": "opt2", "is_correct": True},
                   {"text": "opt3", "is_correct": False}],
        "correct_answer": "B",
    }
    v = _check_no_crash(q)
    assert v.status != "rejected"   # يجب أن يُطبَّع بنجاح بتسميات افتراضية


def test_option_not_a_dict():
    _check_no_crash({"question": "x", "options": ["A", "B", "C"], "correct_answer": "A"})


def test_empty_options_list():
    _check_no_crash({"question": "x", "options": [], "correct_answer": "A"})


def test_none_question():
    _check_no_crash({"question": None,
                     "options": [{"label": "A", "text": "x", "is_correct": True},
                                {"label": "B", "text": "y", "is_correct": False},
                                {"label": "C", "text": "z", "is_correct": False}]})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
