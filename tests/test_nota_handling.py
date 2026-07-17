"""
tests/test_nota_handling.py — يثبت المعالجة الجذرية لأسئلة 'غير ذلك':

  1) سؤال ناقص المعطيات إجابته 'غير ذلك' ⇒ يُرسل للإصلاح بتعليمة إكمال
     المعطيات؛ إن أعاده الإصلاح بإجابة محددة ⇒ verified طبيعي.
  2) 'غير ذلك' مشروعة (السؤال كامل والنموذج المتقاطع يوافق) ⇒ تُقبل
     verified مع توسيم إعلامي catch_all_correct فقط — لا منع ولا هبوط.
  3) لا حلقة إصلاح لا نهائية: كل سؤال يُرسل لإكمال المعطيات مرة واحدة فقط.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import load_corpus, MockClient
from exam_gen.gateway import LLMGateway
from exam_gen.orchestrator import ExamOrchestrator

UP = "/mnt/user-data/uploads"
CKPT = "/tmp/_nota_test_ckpt.json"
_corpus = None


def _get_corpus():
    global _corpus
    if _corpus is None:
        _corpus = load_corpus(UP + "/all_exam_questions.json",
                              UP + "/normalized_knowledge.json",
                              UP + "/doctor_style_profile.json")
    return _corpus


def _run(client, n=6):
    if os.path.exists(CKPT):
        os.unlink(CKPT)
    g = LLMGateway(client, min_interval=0)
    v = LLMGateway(client, min_interval=0)
    o = ExamOrchestrator(_get_corpus(), g, v, checkpoint_path=CKPT,
                         verbose=False)
    return o.run(n, coverage=True, batch_size=6), client


def _nota_question(base_q):
    """يحوّل سؤالاً عادياً إلى سؤال إجابته 'غير ذلك'.
    NOTA المشروعة بطبيعتها بلا كود يناقضها (الناتج ليس بين الخيارات)."""
    q = dict(base_q)
    q["options"] = [dict(o, is_correct=False) for o in q["options"][:-1]]
    q["options"].append({"label": "E", "text": "غير ذلك", "is_correct": True})
    q["correct_answer"] = "E"
    q["verification_code"] = None
    return q


class IncompleteThenFixedMock(MockClient):
    """يولّد أسئلة 'غير ذلك' (تحاكي نقص معطيات)، وإصلاحه يعيدها محددة."""
    def complete_json(self, messages, temperature=0.7):
        sys_msg = messages[0]["content"]
        user = messages[-1]["content"]
        out = super().complete_json(messages, temperature)
        if "exam-question editor" in sys_msg:
            return out                      # الإصلاح يعيد أسئلة محددة (B)
        if "solutions" in out:
            # حلّ لكل سؤال حسب خياراته الفعلية: NOTA ⇒ E، محدد ⇒ B
            segments = user.split("[question_id:")
            per_q = {}
            for seg in segments[1:]:
                qid = seg.split("]")[0].strip()
                per_q[qid] = "E" if "غير ذلك" in seg else "B"
            for s in out["solutions"]:
                s["chosen"] = per_q.get(str(s["question_id"]), "B")
            return out
        if "questions" in out:              # التوليد الأولي: كلها 'غير ذلك'
            out["questions"] = [_nota_question(q) for q in out["questions"]]
        return out


def test_incomplete_nota_gets_repaired_to_concrete():
    res, client = _run(IncompleteThenFixedMock())
    assert res["counts"]["verified"] == 6, res["counts"]
    # بعد الإصلاح، الإجابات محددة لا 'غير ذلك'
    for q in res["verified"]:
        correct = next(o for o in q["options"] if o["is_correct"])
        assert "غير ذلك" not in correct["text"]
        assert not q.get("catch_all_correct")
    # سبب الفشل الأولي كان تعليمة إكمال المعطيات
    events = " ".join(res["events"])
    assert "مُرسَل لإكمال المعطيات" in events


class LegitimateNotaMock(MockClient):
    """'غير ذلك' مقصودة: تبقى بعد الإصلاح، والمحقّق المتقاطع يوافق عليها."""
    def complete_json(self, messages, temperature=0.7):
        sys_msg = messages[0]["content"]
        out = super().complete_json(messages, temperature)
        if "questions" in out:              # توليداً وإصلاحاً: تبقى NOTA
            out["questions"] = [_nota_question(q) for q in out["questions"]]
        if "solutions" in out:
            for s in out["solutions"]:
                s["chosen"] = "E"           # المحقّق المستقل يوافق
        return out


def test_legitimate_nota_is_verified_not_banned():
    res, client = _run(LegitimateNotaMock())
    # لا منع: تُقبل verified مع التوسيم الإعلامي فقط
    assert res["counts"]["verified"] == 6, res["counts"]
    assert res["counts"]["needs_review"] == 0
    for q in res["verified"]:
        assert q.get("catch_all_correct") is True
        correct = next(o for o in q["options"] if o["is_correct"])
        assert "غير ذلك" in correct["text"]


def test_nota_repair_attempted_only_once():
    """السؤال يُرسل لإكمال المعطيات مرة واحدة — لا حلقة إصلاح مهدرة."""
    res, client = _run(LegitimateNotaMock())
    events = " ".join(res["events"])
    # يظهر الإرسال في المرور الأول فقط (عدّ التكرارات)
    count = events.count("مُرسَل لإكمال المعطيات")
    assert count == 1, f"أُرسل {count} مرة — يجب مرة واحدة"


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
