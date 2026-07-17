"""
tests/test_orchestrator.py — يثبت خصائص المعمارية الطبقية:
  1) كفاءة النداءات: 12 سؤالاً بـ ≤ 5 نداءات لكل بوابة.
  2) الميزانية: BudgetExceeded يوقف مبكراً والحالة تُحفظ.
  3) الاستئناف: يكمل من آخر نقطة دون إعادة الدفعات المنجزة.
  4) تعارض النماذج: خيار المحقّق المخالف يرفض السؤال أو يرسله للإصلاح.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import load_corpus, MockClient
from exam_gen.gateway import LLMGateway, BudgetExceeded
from exam_gen.orchestrator import ExamOrchestrator

UP = "/mnt/user-data/uploads"
CKPT = "/tmp/_orch_test_ckpt.json"
_corpus = None


def _get_corpus():
    global _corpus
    if _corpus is None:
        _corpus = load_corpus(UP + "/all_exam_questions.json",
                              UP + "/normalized_knowledge.json",
                              UP + "/doctor_style_profile.json")
    return _corpus


def _fresh(gen=None, ver=None, budget=None):
    if os.path.exists(CKPT):
        os.unlink(CKPT)
    g = LLMGateway(gen or MockClient(), min_interval=0, max_calls=budget)
    v = LLMGateway(ver or MockClient(), min_interval=0, max_calls=budget)
    o = ExamOrchestrator(_get_corpus(), g, v, checkpoint_path=CKPT,
                         verbose=False)
    return o, g, v


def test_call_efficiency():
    o, g, v = _fresh()
    res = o.run(12, coverage=True, batch_size=6)
    assert res["counts"]["verified"] == 12, res["counts"]
    assert g.calls_used <= 5, f"توليد مفرط: {g.calls_used}"
    assert v.calls_used <= 5, f"تحقق مفرط: {v.calls_used}"


def test_budget_stops_early_and_checkpoints():
    o, g, v = _fresh(budget=1)
    try:
        o.run(12, coverage=True, batch_size=6)
        assert False, "كان يجب أن ينقطع بالميزانية"
    except BudgetExceeded:
        pass
    assert os.path.exists(CKPT), "الحالة لم تُحفظ عند الانقطاع"


def test_resume_completes_without_redoing_batches():
    # قطع أولاً
    o1, g1, _ = _fresh(budget=1)
    try:
        o1.run(12, coverage=True, batch_size=6)
    except BudgetExceeded:
        pass
    # استئناف
    g2 = LLMGateway(MockClient(), min_interval=0)
    v2 = LLMGateway(MockClient(), min_interval=0)
    o2 = ExamOrchestrator(_get_corpus(), g2, v2, checkpoint_path=CKPT,
                          verbose=False)
    res = o2.run(12, coverage=True, batch_size=6, resume=True)
    assert res["counts"]["verified"] == 12
    # الدفعة الأولى كانت محفوظة => نداءات التوليد في الاستئناف أقل من الكاملة
    assert g2.calls_used < 4, f"أعاد دفعات منجزة: {g2.calls_used}"


class _DisagreeingVerifier(MockClient):
    """محقّق يختار دائماً C بينما المفتاح B — يحاكي تعارض النماذج."""
    def complete_json(self, messages, temperature=0.7):
        out = super().complete_json(messages, temperature)
        if "solutions" in out:
            for sol in out["solutions"]:
                sol["chosen"] = "C"
                sol["reasoning"] = "أرى أن C هي الصحيحة"
        return out


def test_cross_model_disagreement_blocks_question():
    o, g, v = _fresh(ver=_DisagreeingVerifier())
    res = o.run(6, coverage=True, batch_size=6)
    # لا سؤال يمر verified مع محقّق معارض؛ الكل rejected بعد جولات الإصلاح
    assert res["counts"]["verified"] == 0, res["counts"]
    assert res["counts"]["rejected"] == 6, res["counts"]
    reasons = json.dumps(res["rejected"], ensure_ascii=False)
    assert "تعارض النماذج" in reasons


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
