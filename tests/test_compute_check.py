"""
tests/test_compute_check.py — يثبت سلوك عقدة التحقق الحسابي الحتمي:

  1) مفتاح خاطئ وناتج الكود يطابق خياراً آخر ⇒ قلب المفتاح حتمياً
     (محاكاة سيناريو centroid الحقيقي: 5.1/1.8=2.83 والمفتاح ادّعى 2.57).
  2) الناتج لا يطابق أي خيار ⇒ إرسال للإصلاح والناتج مضمَّن في التعليمة
     (محاكاة السيناريو الغاوسي: 0.595 غير موجود بين الخيارات).
  3) سؤال مفاهيمي (code=null) ⇒ يمر مباشرة، لا كود يُطلب ولا يُنفَّذ.
  4) كود زائف (يطبع ثابتاً بلا حساب) ⇒ يُتجاهل ويعامل كمفاهيمي.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import load_corpus, MockClient
from exam_gen.gateway import LLMGateway
from exam_gen.orchestrator import ExamOrchestrator

UP = "/mnt/user-data/uploads"
CKPT = "/tmp/_compute_test_ckpt.json"
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
    return o.run(n, coverage=True, batch_size=6)


CENTROID_CODE = (
    "import json\n"
    "mu = {1: 0.1, 2: 0.5, 3: 0.8, 4: 0.4}\n"
    "num = sum(z*m for z, m in mu.items())\n"
    "den = sum(mu.values())\n"
    "print(json.dumps({'answer': round(num/den, 2)}))\n"
)   # = 2.83


class WrongKeyRightOptionMock(MockClient):
    """يحاكي سيناريو centroid: المفتاح على 2.57 والصحيح 2.83 موجود كخيار."""
    def _one_question(self, slot_index):
        q = super()._one_question(slot_index)
        q["question"] = (q["question"]
                         + f" ثم احسب الوسط المرجح (centroid) "
                           f"للمجموعة 0.1/1 + 0.5/2 + 0.8/3 + 0.4/4")
        q["options"] = [
            {"label": "A", "text": "2.57", "is_correct": True},   # مفتاح خاطئ
            {"label": "B", "text": "3", "is_correct": False},
            {"label": "C", "text": "2.83", "is_correct": False},  # الصحيح فعلاً
            {"label": "D", "text": "3.2", "is_correct": False},
            {"label": "E", "text": "2.33", "is_correct": False},
        ]
        q["correct_answer"] = "A"
        q["verification_code"] = CENTROID_CODE
        return q

    def complete_json(self, messages, temperature=0.7):
        out = super().complete_json(messages, temperature)
        if "solutions" in out:
            # المحقّق المتقاطع "يوافق المفتاح" أياً كان — يحاكي الخطأ المرتبط.
            # بعد قلب المفتاح آلياً سيوافق على المفتاح المقلوب أيضاً؛ نجعله
            # يختار دائماً حرف المفتاح الحالي عبر حل حقيقي: يختار الخيار 2.83
            for s in out["solutions"]:
                s["chosen"] = None   # يُملأ في الاختبار عبر النص
            # أبسط: اختر الحرف الذي نصه 2.83 من كتلة السؤال
            user = messages[-1]["content"]
            import re
            per = {}
            for seg in user.split("[question_id:")[1:]:
                qid = seg.split("]")[0].strip()
                m = re.search(r"(\w)\)\s*2\.83", seg)
                per[qid] = m.group(1) if m else "A"
            for s in out["solutions"]:
                s["chosen"] = per.get(str(s["question_id"]), "A")
        return out


def test_wrong_key_flipped_to_matching_option():
    res = _run(WrongKeyRightOptionMock())
    assert res["counts"]["verified"] == 6, res["counts"]
    for q in res["verified"]:
        correct = next(o for o in q["options"] if o["is_correct"])
        assert correct["text"] == "2.83", f"لم يُقلب المفتاح: {correct}"
        cvinfo = q.get("compute_verification", {})
        assert cvinfo.get("status") == "key_flipped"
    events = " ".join(res["events"])
    assert "قُلب المفتاح: 6" in events


class NoMatchingOptionMock(MockClient):
    """الناتج المحسوب (2.83) غير موجود بين الخيارات إطلاقاً ⇒ إصلاح بالناتج."""
    def __init__(self):
        super().__init__()
        self.repair_seen_answer = False

    def _one_question(self, slot_index):
        q = super()._one_question(slot_index)
        q["options"] = [
            {"label": "A", "text": "2.57", "is_correct": True},
            {"label": "B", "text": "3", "is_correct": False},
            {"label": "C", "text": "3.5", "is_correct": False},
            {"label": "D", "text": "4", "is_correct": False},
            {"label": "E", "text": "5", "is_correct": False},
        ]
        q["correct_answer"] = "A"
        q["verification_code"] = CENTROID_CODE
        return q

    def complete_json(self, messages, temperature=0.7):
        sys_msg = messages[0]["content"]
        user = messages[-1]["content"]
        if "exam-question editor" in sys_msg:
            if "2.83" in user:              # الناتج المحسوب وصل للإصلاح
                self.repair_seen_answer = True
            # الإصلاح يعيد أسئلة سليمة: الصحيح 2.83 موجود
            import re
            idxs = [int(x) for x in re.findall(r"\[slot_index:\s*(\d+)\]", user)]
            fixed = []
            for i in idxs:
                q = MockClient._one_question(i)
                q["options"] = [
                    {"label": "A", "text": "2.83", "is_correct": True},
                    {"label": "B", "text": "3", "is_correct": False},
                    {"label": "C", "text": "2.5", "is_correct": False},
                ]
                q["correct_answer"] = "A"
                q["verification_code"] = CENTROID_CODE
                fixed.append(q)
            return {"questions": fixed}
        out = super().complete_json(messages, temperature)
        if "solutions" in out:
            import re
            per = {}
            for seg in user.split("[question_id:")[1:]:
                qid = seg.split("]")[0].strip()
                m = re.search(r"(\w)\)\s*2\.83", seg)
                per[qid] = m.group(1) if m else "A"
            for s in out["solutions"]:
                s["chosen"] = per.get(str(s["question_id"]), "A")
        return out


def test_no_matching_option_sends_computed_answer_to_repair():
    client = NoMatchingOptionMock()
    res = _run(client)
    assert client.repair_seen_answer, "الناتج المحسوب لم يصل لتعليمة الإصلاح"
    assert res["counts"]["verified"] == 6, res["counts"]
    for q in res["verified"]:
        correct = next(o for o in q["options"] if o["is_correct"])
        assert correct["text"] == "2.83"


class ConceptualMock(MockClient):
    """أسئلة مفاهيمية: verification_code = null بالتصميم."""
    def _one_question(self, slot_index):
        q = super()._one_question(slot_index)
        q["question"] = (q["question"]
                         + " — نظرياً: قارن دالة الانتماء هنا بالمجموعات "
                           "الكلاسيكية دون أي حساب")
        q["verification_code"] = None
        return q


def test_conceptual_questions_pass_without_code():
    res = _run(ConceptualMock())
    assert res["counts"]["verified"] == 6, res["counts"]
    events = " ".join(res["events"])
    assert "مفاهيمي: 6" in events
    for q in res["verified"]:
        assert "compute_verification" not in q


class FakeCodeMock(MockClient):
    """كود زائف يطبع ثابتاً بلا أي حساب — يجب تجاهله لا الوثوق به."""
    def _one_question(self, slot_index):
        q = super()._one_question(slot_index)
        q["verification_code"] = 'import json\nprint(json.dumps({"answer": "B"}))\n'
        return q


def test_fake_constant_code_is_ignored():
    res = _run(FakeCodeMock())
    events = " ".join(res["events"])
    # عُومل كمفاهيمي (تجاهل الكود الزائف) — لا "نُفِّذ: 6"
    assert "مفاهيمي: 6" in events, events
    assert res["counts"]["verified"] == 6


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
