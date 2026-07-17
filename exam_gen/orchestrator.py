"""
orchestrator.py — منسّق طبقي بحالة مشتركة (مفاهيم LangGraph بلا اعتماد خارجي).

المعمارية:
  ExamState (حالة مشتركة) تمر عبر عقد (Nodes) مرتبة، وحواف شرطية تقرر
  التكرار (إصلاح ← إعادة تحقق، بحد أقصى دورتين). كل عقدة إما حتمية
  (بدون LLM) أو مُدفَّعة (نداء LLM واحد لدفعة كاملة عبر Gateway).

الطبقات/العقد:
  plan        (حتمية)  جدول مواصفات + تجميع الخانات دفعات متجانسة الموضوع
  generate    (LLM)    نداء واحد لكل دفعة يولّد كل أسئلتها
  structure   (حتمية)  فحص البنية + التطبيع الدفاعي
  dedup       (حتمية)  كشف تكرار/تعارض داخل الامتحان وضد بنك الدورات
  cross_verify(LLM)    نموذج مختلف يحلّ كل الأسئلة الناجية بنداء مُدفَّع
  repair      (LLM)    نداء واحد يصلح كل الفاشلات بأسبابها → عودة للتحقق
  assemble    (حتمية)  التجميع النهائي + تقرير الشفافية

Checkpointing: الحالة تُحفظ JSON بعد كل عقدة؛ عند انقطاع (429 مثلاً)
يُستأنف التشغيل من آخر عقدة مكتملة بـ resume=True.
"""
from __future__ import annotations
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .blueprint import build_blueprint, Slot
from .retrieval import Retriever, _tf, _tokens, _cosine_bow
from .gateway import LLMGateway, BudgetExceeded
from . import batch_prompts as BP
from .verify import check_structure
from .pipeline import _to_schema


# --------------------------------------------------------------------------- #
#  الحالة المشتركة
# --------------------------------------------------------------------------- #
@dataclass
class ExamState:
    n_questions: int
    seed: int = 0
    coverage: bool = False
    course_description: str = ""
    batch_size: int = 6
    max_repair_rounds: int = 2

    stage: str = "plan"                      # اسم العقدة التالية
    repair_round: int = 0
    slots: List[dict] = field(default_factory=list)         # asdict(Slot)
    batches: List[List[int]] = field(default_factory=list)  # فهارس slots
    questions: Dict[str, dict] = field(default_factory=dict)  # slot_index -> q
    failures: Dict[str, str] = field(default_factory=dict)    # slot_index -> سبب
    verified: List[dict] = field(default_factory=list)
    needs_review: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def log(self, msg: str, verbose: bool = True):
        self.events.append(msg)
        if verbose:
            print(msg)


def _slot_of(d: dict) -> Slot:
    return Slot(**d)


# --------------------------------------------------------------------------- #
#  المنسّق
# --------------------------------------------------------------------------- #
class ExamOrchestrator:
    """
    generator_gw: بوابة نموذج التوليد.
    verifier_gw:  بوابة نموذج التحقق (نموذج مختلف — cross-model).
    """

    def __init__(self, corpus, generator_gw: LLMGateway,
                 verifier_gw: LLMGateway, embed_fn=None,
                 checkpoint_path: str = ".exam_checkpoint.json",
                 similarity_threshold: float = 0.85,
                 verbose: bool = True):
        self.corpus = corpus
        self.gen = generator_gw
        self.ver = verifier_gw
        self.retriever = Retriever(corpus, embed_fn)
        self.checkpoint_path = checkpoint_path
        self.sim_threshold = similarity_threshold
        self.verbose = verbose

    # ---------------- checkpointing ---------------- #
    def _save(self, state: ExamState):
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False)

    def _load(self) -> Optional[ExamState]:
        if not os.path.exists(self.checkpoint_path):
            return None
        with open(self.checkpoint_path, encoding="utf-8") as f:
            return ExamState(**json.load(f))

    def _clear(self):
        if os.path.exists(self.checkpoint_path):
            os.unlink(self.checkpoint_path)

    # ---------------- التشغيل ---------------- #
    NODES = ["plan", "generate", "structure", "dedup", "compute_check",
             "cross_verify", "repair", "assemble"]

    def run(self, n_questions: int, seed: int = 0, coverage: bool = False,
            course_description: str = "", batch_size: int = 6,
            resume: bool = False) -> dict:
        state = self._load() if resume else None
        if state is None:
            state = ExamState(n_questions=n_questions, seed=seed,
                              coverage=coverage,
                              course_description=course_description,
                              batch_size=batch_size)
        else:
            state.log(f"[resume] استئناف من العقدة: {state.stage}", self.verbose)

        try:
            while state.stage != "done":
                node = getattr(self, f"node_{state.stage}")
                next_stage = node(state)
                state.stage = next_stage
                self._save(state)
        except BudgetExceeded as e:
            state.log(f"[budget] {e} — الحالة محفوظة، أكمل لاحقاً بـ --resume",
                      self.verbose)
            self._save(state)
            raise
        except Exception:
            self._save(state)            # أي انقطاع: الحالة محفوظة للاستئناف
            raise

        result = self._final_result(state)
        self._clear()
        return result

    # ---------------- العقد ---------------- #
    def node_plan(self, s: ExamState) -> str:
        slots = build_blueprint(self.corpus, s.n_questions, s.seed,
                                coverage=s.coverage)
        s.slots = [asdict(x) for x in slots]
        # دفعات متجانسة الموضوع: رتّب بالموضوع ثم قطّع
        order = sorted(range(len(slots)), key=lambda i: slots[i].topic)
        s.batches = [order[i:i + s.batch_size]
                     for i in range(0, len(order), s.batch_size)]
        s.log(f"[plan] {len(slots)} خانة في {len(s.batches)} دفعة "
              f"(حجم الدفعة {s.batch_size})", self.verbose)
        return "generate"

    def node_generate(self, s: ExamState) -> str:
        from .gateway import BudgetExceeded
        for bi, batch in enumerate(s.batches, 1):
            slots = [_slot_of(s.slots[i]) for i in batch]
            # هل الدفعة مولّدة سلفاً (استئناف)؟
            if all(str(x.index) in s.questions for x in slots):
                continue
            self._generate_slots_with_fallback(s, slots, bi)
            s.log(f"[generate] دفعة {bi}/{len(s.batches)} مكتملة "
                  f"({sum(1 for x in slots if str(x.index) in s.questions)}/"
                  f"{len(slots)} سؤالاً)", self.verbose)
            self._save(s)                 # حفظ بعد كل دفعة (استئناف دقيق)
        return "structure"

    def _generate_slots_with_fallback(self, s: ExamState, slots: List[Slot],
                                      batch_label) -> None:
        """
        يولّد دفعة؛ عند فشل النداء (JSON مقطوع/مشوّه، ميزانية...) يقسّم
        الدفعة إلى نصفين ويعيد المحاولة (تقسيم ثنائي حتى حجم 1)، بدل
        تحطيم التشغيل بالكامل. خانة تفشل حتى بحجم 1 تُسجَّل rejected
        بسبب واضح ولا توقف بقية الامتحان.
        """
        from .gateway import BudgetExceeded
        if not slots:
            return
        topics = " ".join(sorted({x.topic for x in slots}))
        levels = " ".join(sorted({x.cognitive_level for x in slots}))
        try:
            grounding = self.retriever.grounding_for(topics, levels, k=8)
            exemplars = self.retriever.exemplars_for(topics, levels, k=4)
            msgs = BP.build_batch_generation_messages(
                slots, grounding, exemplars, self.corpus.style_profile,
                s.course_description)
            out = self.gen.complete_json(msgs, temperature=0.7,
                                         purpose=f"generate_batch_{batch_label}")
            got = {int(q.get("slot_index", -1)): q
                   for q in out.get("questions", []) if isinstance(q, dict)}
            for x in slots:
                if x.index in got:
                    s.questions[str(x.index)] = got[x.index]
                else:
                    s.failures[str(x.index)] = "النموذج لم يُرجع سؤالاً لهذه الخانة"
            return
        except BudgetExceeded:
            raise                          # قرار إيقاف متعمَّد — لا نلتقطه هنا
        except Exception as e:
            if len(slots) == 1:
                s.failures[str(slots[0].index)] = (
                    f"فشل التوليد نهائياً لهذه الخانة: {e}")
                s.log(f"    [generate] خانة {slots[0].index} فشلت نهائياً: {e}",
                      self.verbose)
                return
            mid = len(slots) // 2
            s.log(f"    [generate] دفعة {batch_label} فشلت ({e}) — "
                 f"تقسيم إلى {mid} + {len(slots)-mid} وإعادة المحاولة",
                 self.verbose)
            self._generate_slots_with_fallback(s, slots[:mid], f"{batch_label}a")
            self._generate_slots_with_fallback(s, slots[mid:], f"{batch_label}b")

    def node_structure(self, s: ExamState) -> str:
        ok = bad = 0
        for idx, q in list(s.questions.items()):
            if idx in s.failures:
                continue
            layer = check_structure(q)     # يطبّع q في مكانه أيضاً
            if layer.passed:
                ok += 1
            else:
                s.failures[idx] = f"بنية: {layer.detail}"
                bad += 1
        sent = self._completeness_check(s)
        s.log(f"[structure] سليم: {ok} | فاشل: {bad}"
              + (f" | مُرسَل لإكمال المعطيات: {sent}" if sent else ""),
              self.verbose)
        return "dedup"

    def node_dedup(self, s: ExamState) -> str:
        """كشف تكرار/تشابه مفرط: داخل الامتحان + ضد بنك الدورات."""
        live = [(idx, q) for idx, q in s.questions.items()
                if idx not in s.failures]
        bows = {idx: _tf(_tokens(q.get("question", ""))) for idx, q in live}
        flagged = 0
        # داخل الامتحان (زوجياً؛ الأحدث فهرساً يُعلَّم)
        idxs = [idx for idx, _ in live]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if ib in s.failures:
                    continue
                sim = _cosine_bow(bows[ia], bows[ib])
                if sim >= self.sim_threshold:
                    s.failures[ib] = (f"تكرار داخلي: يشابه سؤال الخانة {ia} "
                                      f"(تشابه {sim:.2f}) — أعد صياغته بمفهوم مختلف")
                    flagged += 1
        # ضد بنك الدورات الأصلي
        for idx, q in live:
            if idx in s.failures:
                continue
            sim = self.retriever.max_similarity_to_bank(q.get("question", ""))
            if sim >= self.sim_threshold:
                s.failures[idx] = (f"تسريب: يشابه سؤالاً حقيقياً من الدورات "
                                   f"(تشابه {sim:.2f}) — ولّد سؤالاً جديداً")
                flagged += 1
        s.log(f"[dedup] معلَّم للتكرار/التسريب: {flagged}", self.verbose)
        return "compute_check"

    # ------------------------------------------------------------------ #
    #  عقدة التحقق الحسابي الحتمي (صفر نداءات LLM)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _looks_like_fake_code(code: str) -> bool:
        """
        حارس الكود الزائف: كود 'يطبع ثابتاً' دون أي حساب من معطيات —
        نمط يظهر إن ولّد النموذج كوداً لسؤال مفاهيمي رغم التعليمات.
        كشف تحفظي: لا معاملات حسابية/مقارنات/حلقات/دوال مكتبية إطلاقاً.
        """
        body = re.sub(r"print\s*\(.*?\)\s*$", "", code.strip(),
                      flags=re.DOTALL)
        computational = re.search(
            r"[+\-*/%<>]|==|min\(|max\(|sum\(|sorted\(|abs\(|range\(|"
            r"for\s|while\s|math\.|statistics\.|fractions\.|len\(", body)
        return computational is None

    def node_compute_check(self, s: ExamState) -> str:
        """
        الحَكَم الحتمي: ينفّذ verification_code (إن وُجد) ويقارن ناتجه
        بالمفتاح المُدّعى. عند التعارض لا يرفض بل **يتبنى ناتج الكود**:
          1) الناتج يطابق خياراً آخر ⇒ قلب المفتاح إليه حتمياً (صفر نداءات).
          2) الناتج لا يطابق أي خيار ⇒ إرسال للإصلاح مع الناتج المحسوب
             مضمَّناً في التعليمة (النموذج يصيغ فقط، لا يحسب).
          3) الكود فشل تنفيذياً ⇒ يمر للتحقق المتقاطع فقط (بلا امتياز الكود).
        الأسئلة المفاهيمية (code=null) تمر مباشرة — لا كود لها بالتصميم.
        بعد أي تعديل هنا يمر السؤال بالتحقق المتقاطع كالمعتاد (حَكَمان).
        """
        from . import code_verifier as cv
        checked = fixed = to_repair = conceptual = failed_exec = 0
        for idx, q in s.questions.items():
            if idx in s.failures:
                continue
            code = q.get("verification_code")
            if not code or not str(code).strip() or str(code).lower() == "none":
                conceptual += 1
                continue
            if self._looks_like_fake_code(str(code)):
                q["_compute_status"] = "fake_code_ignored"
                conceptual += 1
                continue
            checked += 1
            ok, computed, detail = cv.run_verification_code(str(code))
            if not ok:
                q["_compute_status"] = f"exec_failed: {detail[:120]}"
                failed_exec += 1
                continue                     # يبقى للتحقق المتقاطع وحده
            # fallback عملي: الكود أخرج حرف خيار (خرق للعقد لكنه شائع) —
            # نقارنه بالتسمية مباشرة بدل نصوص الخيارات
            labels_set = {str(o.get("label")).strip().upper()
                          for o in q["options"]}
            if isinstance(computed, str) and \
               computed.strip().upper() in labels_set:
                lab = computed.strip().upper()
                if lab == str(q.get("correct_answer", "")).strip().upper():
                    q["_compute_status"] = "agreed_by_label"
                    q["_compute_answer"] = lab
                else:
                    match_opt = next(o for o in q["options"]
                                     if str(o.get("label")).strip().upper() == lab)
                    for o in q["options"]:
                        o["is_correct"] = (o is match_opt)
                    q["correct_answer"] = match_opt["label"]
                    q["_compute_status"] = "key_flipped_by_label"
                    q["_compute_answer"] = lab
                    fixed += 1
                continue
            correct = next((o for o in q["options"] if o.get("is_correct")), None)
            if correct and cv.generic_answers_match(computed, correct["text"]):
                q["_compute_status"] = "agreed"
                q["_compute_answer"] = str(computed)
                continue
            # تعارض: تبنّى ناتج الكود
            match_opt = next(
                (o for o in q["options"]
                 if cv.generic_answers_match(computed, o.get("text", ""))), None)
            if match_opt is not None:
                for o in q["options"]:
                    o["is_correct"] = (o is match_opt)
                q["correct_answer"] = match_opt["label"]
                q["_compute_status"] = "key_flipped"
                q["_compute_answer"] = str(computed)
                q["worked_solution"] = (
                    f"[صُحِّح المفتاح آلياً: الناتج المحسوب = {computed}] "
                    + str(q.get("worked_solution", "")))
                fixed += 1
            else:
                s.failures[idx] = (
                    f"الناتج المحسوب آلياً بتنفيذ الكود هو {computed} ولا "
                    f"يطابق أي خيار. أعد صياغة الخيارات بحيث يكون أحدها "
                    f"يساوي {computed} تماماً واجعله الصحيح، مع تحديث الحل "
                    f"خطوة بخطوة ليصل إلى هذا الناتج. لا تغيّر معطيات السؤال.")
                to_repair += 1
        s.log(f"[compute_check] مفاهيمي: {conceptual} | نُفِّذ: {checked} "
              f"| قُلب المفتاح: {fixed} | للإصلاح بالناتج: {to_repair} "
              f"| فشل تنفيذ: {failed_exec}", self.verbose)
        return "cross_verify"

    def node_cross_verify(self, s: ExamState) -> str:
        """نموذج مختلف يحلّ الأسئلة الناجية دون رؤية المفاتيح — مُدفَّعاً."""
        live = []
        for idx, q in s.questions.items():
            if idx in s.failures:
                continue
            qq = dict(q)
            qq["question_id"] = idx
            live.append(qq)
        if not live:
            return "repair"

        chunk = 10                        # أسئلة لكل نداء حلّ
        agreements = disagreements = 0
        from .gateway import BudgetExceeded
        for i in range(0, len(live), chunk):
            part = live[i:i + chunk]
            try:
                msgs = BP.build_batch_solver_messages(part)
                out = self.ver.complete_json(msgs, temperature=0.0,
                                             purpose=f"cross_verify_{i//chunk+1}")
                sols = {str(x.get("question_id")): x
                        for x in out.get("solutions", []) if isinstance(x, dict)}
            except BudgetExceeded:
                raise
            except Exception as e:
                s.log(f"    [cross_verify] فشل نداء الحلّ لهذه الدفعة ({e}) — "
                     f"تُرسَل للمراجعة البشرية بدل الرفض", self.verbose)
                sols = {}
            for q in part:
                idx = q["question_id"]
                sol = sols.get(idx)
                if sol is None:
                    s.questions[idx]["_cross_verify_unavailable"] = True
                    continue
                if sol.get("chosen") == s.questions[idx].get("correct_answer"):
                    agreements += 1
                    s.questions[idx]["_cross_verified"] = True
                    s.questions[idx]["_verifier_confidence"] = sol.get("confidence")
                else:
                    disagreements += 1
                    s.failures[idx] = (
                        f"تعارض النماذج: المحقّق المستقل اختار "
                        f"{sol.get('chosen')} بينما المفتاح "
                        f"{s.questions[idx].get('correct_answer')} — "
                        f"حجة المحقّق: {str(sol.get('reasoning'))[:200]}")
        s.log(f"[cross_verify] اتفاق: {agreements} | تعارض: {disagreements}",
              self.verbose)
        return "repair"

    def node_repair(self, s: ExamState) -> str:
        from .gateway import BudgetExceeded
        fixable = [idx for idx in s.failures
                   if idx in s.questions]     # فشل له سؤال يمكن إصلاحه
        if not fixable or s.repair_round >= s.max_repair_rounds:
            return "assemble"
        s.repair_round += 1
        got = {}
        REPAIR_CHUNK = 5              # نداء إصلاح كبير (13 سؤالاً) تجاوز
        for ci in range(0, len(fixable), REPAIR_CHUNK):   # المهلة في تشغيل حقيقي
            part = fixable[ci:ci + REPAIR_CHUNK]
            payload = [{"slot_index": int(idx),
                        "question": s.questions[idx],
                        "reason": s.failures[idx]} for idx in part]
            try:
                msgs = BP.build_batch_repair_messages(payload)
                out = self.gen.complete_json(
                    msgs, temperature=0.5,
                    purpose=f"repair_r{s.repair_round}_c{ci//REPAIR_CHUNK+1}")
                got.update({str(q.get("slot_index")): q
                            for q in out.get("questions", [])
                            if isinstance(q, dict)})
                self._save(s)          # تقدّم جزئي محفوظ بين القطع
            except BudgetExceeded:
                raise
            except Exception as e:
                s.log(f"[repair] فشلت قطعة إصلاح ({e}) — تُستكمل البقية",
                     self.verbose)
        repaired = 0
        for idx in fixable:
            if idx not in got:
                continue
            new_q, old_q = got[idx], s.questions[idx]
            # حارس الدمج: لا تستبدل سؤالاً بآخر ناقص بنيوياً.
            # "أقل تغيير" قد يجعل النموذج يعيد كائناً بلا نص سؤال/خيارات —
            # نرث الحقول الناقصة من الأصل بدل إتلاف محتوى سليم.
            if not str(new_q.get("question", "")).strip():
                new_q["question"] = old_q.get("question", "")
            if not new_q.get("options"):
                new_q["options"] = old_q.get("options", [])
            if not str(new_q.get("worked_solution", "")).strip():
                new_q["worked_solution"] = old_q.get("worked_solution", "")
            if not new_q.get("verification_code"):
                new_q["verification_code"] = old_q.get("verification_code")
            if not str(new_q.get("question", "")).strip() or \
               len(new_q.get("options") or []) < 3:
                continue                   # ما زال ناقصاً: أبقِ الأصل وفشله
            # حافظ على وسوم الدورة عبر الاستبدال (منع إعادة إرسال بلا داعٍ)
            if old_q.get("_nota_repair_attempted"):
                new_q["_nota_repair_attempted"] = True
            s.questions[idx] = new_q
            del s.failures[idx]
            repaired += 1
        s.log(f"[repair] جولة {s.repair_round}: أُصلح {repaired}/{len(fixable)} "
              f"— إعادة التحقق", self.verbose)
        # المُصلَح يعود لسلسلة التحقق كاملة
        return "structure" if repaired else "assemble"

    # ---- خطوات حتمية قبل التجميع ---- #
    _CATCH_ALL = ("غير ذلك", "لا شيء مما سبق", "كل ما سبق", "جميع ما سبق",
                  "none of the above", "all of the above", "other")

    def _is_catch_all_correct(self, q: dict) -> bool:
        correct = next((o for o in q.get("options", [])
                        if o.get("is_correct")), None)
        return bool(correct and any(
            c in str(correct.get("text", "")).lower() for c in self._CATCH_ALL))

    def _completeness_check(self, s: ExamState) -> int:
        """
        إصلاح جذري لا منع: 'غير ذلك' كإجابة صحيحة غالباً عرضٌ لسؤال ناقص
        المعطيات رقّعه النموذج بخيار جامع. نرسله لعقدة الإصلاح بتعليمة
        إكمال المعطيات — **مرة واحدة فقط** لكل سؤال. إن عاد بعد الإصلاح
        وما زال 'غير ذلك' صحيحة (أي أنها مقصودة وسؤاله كامل)، يُقبل ويمر
        بالتحقق المتقاطع كأي سؤال، ويُوسم توسيماً إعلامياً فقط للمدرّس.
        """
        sent = 0
        for idx, q in s.questions.items():
            if idx in s.failures:
                continue
            if self._is_catch_all_correct(q) and not q.get("_nota_repair_attempted"):
                q["_nota_repair_attempted"] = True
                s.failures[idx] = (
                    "يُشتبه أن السؤال ناقص المعطيات (الإجابة الصحيحة خيار "
                    "جامع مثل 'غير ذلك'). إن كانت المعطيات ناقصة: أضف القيم "
                    "العددية المفقودة واجعل السؤال قابلاً للحل بإجابة محددة "
                    "واحدة صحيحة من الخيارات، مع تحديث الحل خطوة بخطوة. "
                    "أما إن كان السؤال كامل المعطيات وخيار 'غير ذلك' مقصود "
                    "فعلاً (الناتج المحسوب ليس بين الخيارات الأخرى): أبقِ "
                    "السؤال كما هو وبيّن في الحل الحساب الكامل الذي يثبت ذلك.")
                sent += 1
        return sent

    def _balance_answer_positions(self, s: ExamState) -> None:
        """
        يكسر انحياز موضع الإجابة الصحيحة بضمانة حتمية لا احتمالية:
        **توزيع دوري (round-robin)** لموضع الصحيح عبر A..E — كل موضع يحصل
        على حصة متساوية (±1) بدل الاعتماد على خلط عشوائي قد ينحرف في
        العينات الصغيرة (لوحظ E=36% في تشغيل حقيقي رغم الخلط).
        بقية المشتّتات تُخلط ببذرة قابلة لإعادة الإنتاج. الخيارات الجامعة
        ('غير ذلك') تُثبَّت آخر القائمة، ويُدوَّر الصحيح على المواضع
        المتحركة فقط عندها. صفر نداءات LLM.
        """
        import random as _random
        rng = _random.Random(s.seed + 9973)
        live = [idx for idx in sorted(s.questions, key=lambda x: int(x))
                if idx not in s.failures]
        for turn, idx in enumerate(live):
            q = s.questions[idx]
            opts = q.get("options", [])
            if len(opts) < 3:
                continue
            labels = [o.get("label") for o in opts]
            movable = [o for o in opts
                       if not any(c in str(o.get("text", "")).lower()
                                  for c in self._CATCH_ALL)]
            fixed_tail = [o for o in opts if o not in movable]
            correct = next((o for o in opts if o.get("is_correct")), None)
            if correct is None:
                continue
            if correct in movable:
                # الموضع المستهدف للصحيح: دوري على المواضع المتحركة
                target = turn % len(movable)
                distractors = [o for o in movable if o is not correct]
                rng.shuffle(distractors)
                movable = (distractors[:target] + [correct]
                           + distractors[target:])
            else:
                rng.shuffle(movable)          # الصحيح خيار جامع مثبَّت آخراً
            reordered = movable + fixed_tail
            for o, lab in zip(reordered, labels):
                o["label"] = lab
            q["options"] = reordered
            q["correct_answer"] = next(o["label"] for o in reordered
                                       if o.get("is_correct"))

    def node_assemble(self, s: ExamState) -> str:
        self._balance_answer_positions(s)
        slot_by_idx = {str(d["index"]): _slot_of(d) for d in s.slots}

        class _V:                          # verdict خفيف للتوافق مع _to_schema
            def __init__(self, status, detail=""):
                self.status = status
                self._detail = detail
            def as_dict(self):
                return {"status": self.status,
                        "layers": [{"name": "orchestrated", "passed":
                                    self.status == "verified",
                                    "detail": self._detail}]}

        for idx, q in s.questions.items():
            slot = slot_by_idx[idx]
            if idx in s.failures:
                rec = _to_schema(q, slot, _V("rejected", s.failures[idx]),
                                 int(idx))
                s.rejected.append(rec)
            elif q.get("_cross_verified"):
                detail = "cross-model agreement"
                if q.get("_compute_status") in ("agreed", "key_flipped"):
                    detail = ("computational + cross-model agreement"
                              if q["_compute_status"] == "agreed"
                              else "key auto-corrected by code, then cross-verified")
                rec = _to_schema(q, slot, _V("verified", detail), int(idx))
                if q.get("_compute_status"):
                    rec["compute_verification"] = {
                        "status": q["_compute_status"],
                        "computed_answer": q.get("_compute_answer")}
                if self._is_catch_all_correct(q):
                    # توسيم إعلامي فقط (لا يهبط الحالة): 'غير ذلك' صحيحة
                    # بعد فرصة إكمال المعطيات وموافقة المحقّق المتقاطع —
                    # يُترك للمدرّس قرار الإبقاء
                    rec["catch_all_correct"] = True
                s.verified.append(rec)
            else:
                reason = q.get("_needs_review_reason",
                               "لم يمر بالتحقق المتقاطع")
                rec = _to_schema(q, slot, _V("needs_review", reason), int(idx))
                if self._is_catch_all_correct(q):
                    rec["catch_all_correct"] = True
                s.needs_review.append(rec)
        # خانات لم يصلها سؤال إطلاقاً
        for idx, reason in s.failures.items():
            if idx not in s.questions:
                s.rejected.append({"question_id": f"missing_{idx}",
                                   "topic": slot_by_idx[idx].topic,
                                   "verification": {"status": "rejected",
                                                    "layers": [{"name": "generation",
                                                                "passed": False,
                                                                "detail": reason}]}})
        s.log(f"[assemble] verified={len(s.verified)} "
              f"needs_review={len(s.needs_review)} rejected={len(s.rejected)}",
              self.verbose)
        return "done"

    # ---------------- النتيجة ---------------- #
    def _final_result(self, s: ExamState) -> dict:
        return {
            "counts": {"verified": len(s.verified),
                       "needs_review": len(s.needs_review),
                       "rejected": len(s.rejected)},
            "verified": s.verified,
            "needs_review": s.needs_review,
            "rejected": s.rejected,
            "llm_usage": {"generator": self.gen.report(),
                          "verifier": self.ver.report()},
            "events": s.events,
        }
