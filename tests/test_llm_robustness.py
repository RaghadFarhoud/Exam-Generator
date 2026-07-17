"""
tests/test_llm_robustness.py — يثبت إصلاح عطل حقيقي واجهه مستخدم:
JSONDecodeError عند نهاية توليد دفعة (استجابة مقطوعة/مشوّهة من Mistral).

يغطي:
  1) _extract_json يصلح فواصل زائدة واقتباسات ذكية.
  2) _extract_json يستخرج أطول JSON صالح من نص مقطوع (truncation).
  3) رسالة خطأ تشخيصية واضحة عند فشل كل الإصلاحات (لا Traceback خام).
  4) الأورشستريتور يقسّم الدفعة الفاشلة بدل تحطيم التشغيل بالكامل.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen.llm_client import _extract_json, JSONExtractionError
from exam_gen import load_corpus, MockClient
from exam_gen.gateway import LLMGateway
from exam_gen.orchestrator import ExamOrchestrator

UP = "/mnt/user-data/uploads"


def test_repairs_trailing_comma():
    text = '{"questions": [{"a": 1,}, {"b": 2,},]}'
    out = _extract_json(text)
    assert out["questions"] == [{"a": 1}, {"b": 2}]


def test_repairs_smart_quotes():
    text = '{\u201cquestions\u201d: [{\u201ca\u201d: 1}]}'
    out = _extract_json(text)
    assert out["questions"] == [{"a": 1}]


def test_recovers_truncated_array_mid_object():
    """يحاكي بالضبط عطل المستخدم: استجابة مقطوعة فعلياً بلا إغلاق نهائي."""
    truncated = ('{"questions": [{"slot_index": 1, "question": "ok"}, '
                '{"slot_index": 2, "question": "ok2"}, '
                '{"slot_index": 3, "question": "broken mid-strea')  # انقطع هنا فعلياً
    out = _extract_json(truncated)
    # يجب أن يستخرج على الأقل العنصرين الكاملين الأولين دون رمي استثناء فادح
    qs = out.get("questions", [])
    assert len(qs) >= 2
    assert qs[0]["slot_index"] == 1 and qs[1]["slot_index"] == 2


def test_clear_diagnostic_on_total_garbage():
    try:
        _extract_json("this is not json at all, no braces here")
        assert False, "كان يجب أن يرمي JSONExtractionError"
    except JSONExtractionError as e:
        assert "لا يوجد JSON" in str(e) or "JSON" in str(e)


def test_extract_json_never_raises_bare_jsondecodeerror():
    """الفشل يجب أن يخرج كـ JSONExtractionError (تشخيصي) لا JSONDecodeError خام."""
    import json as _json
    bad_inputs = [
        '{"a": "unterminated string',
        '{"a": 1 "b": 2}',            # فاصلة مفقودة (عطل المستخدم الفعلي)
        '{',
        '',
    ]
    for bad in bad_inputs:
        try:
            _extract_json(bad)
        except JSONExtractionError:
            pass                       # متوقَّع ومقبول
        except _json.JSONDecodeError:
            assert False, f"تسرّب JSONDecodeError خام لمدخل: {bad!r}"


def test_orchestrator_survives_batch_generation_crash():
    """
    عميل يفشل بخطأ JSON غير قابل للإصلاح لأول نداء فقط (يحاكي دفعة تالفة)
    ثم ينجح. يجب أن يكمل الأورشستريتور بتقسيم الدفعة، لا أن يتحطم بالكامل.
    """
    corpus = load_corpus(UP + "/all_exam_questions.json",
                         UP + "/normalized_knowledge.json",
                         UP + "/doctor_style_profile.json")

    class FlakyOnFirstCall(MockClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete_json(self, messages, temperature=0.7):
            self.calls += 1
            if self.calls == 1 and "MULTIPLE" in messages[0]["content"]:
                raise JSONExtractionError("محاكاة استجابة مقطوعة")
            return super().complete_json(messages, temperature)

    gen = FlakyOnFirstCall()
    g = LLMGateway(gen, min_interval=0)
    v = LLMGateway(MockClient(), min_interval=0)
    ckpt = "/tmp/_llm_robust_ckpt.json"
    if os.path.exists(ckpt):
        os.unlink(ckpt)
    orch = ExamOrchestrator(corpus, g, v, checkpoint_path=ckpt, verbose=False)
    res = orch.run(6, coverage=True, batch_size=6)   # دفعة واحدة فاشلة أولاً
    total = sum(res["counts"].values())
    assert total == 6, f"فقدت أسئلة: {res['counts']}"
    assert gen.calls > 1, "لم يُعِد المحاولة بتقسيم الدفعة"





def test_partial_repair_output_does_not_destroy_questions():
    """
    عطل حقيقي: الإصلاح أعاد كائنات بلا نص سؤال ('أقل تغيير')، فاستُبدلت
    أسئلة سليمة بفارغة ورُفض 11 سؤالاً بـ 'نص السؤال فارغ'. حارس الدمج
    يجب أن يرث الحقول الناقصة من الأصل بدل الإتلاف.
    """
    from exam_gen.orchestrator import ExamOrchestrator
    corpus = load_corpus(UP + "/all_exam_questions.json",
                         UP + "/normalized_knowledge.json",
                         UP + "/doctor_style_profile.json")

    class PartialRepairMock(MockClient):
        def _one_question(self, slot_index):
            q = super()._one_question(slot_index)
            # اجعل التوليد الأولي فاشلاً حسابياً (مفتاح خاطئ يطابق خياراً آخر لا يوجد)
            q["options"] = [
                {"label": "A", "text": "999", "is_correct": True},
                {"label": "B", "text": "888", "is_correct": False},
                {"label": "C", "text": "777", "is_correct": False},
            ]
            q["correct_answer"] = "A"
            # الكود يحسب [1,2,4] الذي لا يطابق أي خيار => إرسال للإصلاح
            return q

        def complete_json(self, messages, temperature=0.7):
            sys_msg = messages[0]["content"]
            user = messages[-1]["content"]
            if "exam-question editor" in sys_msg:
                import re
                idxs = [int(x) for x in re.findall(r"\[slot_index:\s*(\d+)\]", user)]
                # إصلاح ناقص: بلا نص سؤال، خيارات صحيحة فقط
                return {"questions": [{
                    "slot_index": i,
                    "question": "",                     # <-- الناقص
                    "options": [
                        {"label": "A", "text": "{1, 2, 4}", "is_correct": True},
                        {"label": "B", "text": "888", "is_correct": False},
                        {"label": "C", "text": "777", "is_correct": False},
                    ],
                    "correct_answer": "A",
                } for i in idxs]}
            out = super().complete_json(messages, temperature)
            if "solutions" in out:
                for s in out["solutions"]:
                    s["chosen"] = "A"
            return out

    g = LLMGateway(PartialRepairMock(), min_interval=0)
    v = LLMGateway(PartialRepairMock(), min_interval=0)
    ckpt = "/tmp/_partial_repair_ckpt.json"
    if os.path.exists(ckpt):
        os.unlink(ckpt)
    orch = ExamOrchestrator(corpus, g, v, checkpoint_path=ckpt, verbose=False)
    res = orch.run(6, coverage=True, batch_size=6)
    # لا سؤال يُرفض بسبب 'نص السؤال فارغ' — الحارس ورث النص من الأصل
    all_reasons = str(res["rejected"])
    assert "نص السؤال فارغ" not in all_reasons, "الحارس لم يمنع الإتلاف"
    for q in res["verified"] + res["needs_review"]:
        assert str(q.get("question", "")).strip(), "سؤال فارغ تسرب!"


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
