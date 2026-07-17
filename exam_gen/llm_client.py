"""
llm_client.py — واجهة LLM قابلة للتبديل.

- LLMClient: الواجهة المجرّدة (complete_json).
- MistralClient: مثال حقيقي (يتطلب مفتاح + شبكة).
- MockClient: لتشغيل الـ pipeline بالكامل دون شبكة (للاختبار المعماري).

المولّد والمُحقّق يتوقعان JSON، لذا complete_json تُعيد dict مُحلَّلاً.
"""
from __future__ import annotations
import json
import re
import time
import random
from typing import List, Optional


def _retry_after_seconds(exc) -> float | None:
    """يحاول استخراج مهلة الانتظار الموصى بها من استجابة الخادم (Retry-After)."""
    for attr in ("response", "raw_response", "http_response"):
        resp = getattr(exc, attr, None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        if headers:
            for key in ("Retry-After", "retry-after", "x-ratelimit-reset"):
                if key in headers:
                    try:
                        return float(headers[key])
                    except (TypeError, ValueError):
                        pass
    return None


def _retry_with_backoff(fn, max_retries: int = 5, base_delay: float = 2.0,
                        max_delay: float = 30.0):
    """
    يُعيد المحاولة عند أخطاء الشبكة/الاتصال المؤقتة أو تجاوز حدود المعدل
    (شائعة مع الحسابات المجانية). يميّز 429/rate_limited عن غيره لأنه
    يحتاج مهلة أطول عادةً، ويحترم رأس Retry-After إن توفّر.
    تراجع أُسّي + jitter عشوائي لتفادي "عاصفة إعادة المحاولات".
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = any(s in msg for s in (
                "429", "rate_limited", "rate limit exceeded", "too many requests"))
            transient = is_rate_limit or any(s in msg for s in (
                "connection reset", "104", "timeout", "timed out",
                "connection aborted", "temporarily unavailable",
                "502", "503", "504", "connection error"))
            last_err = e
            if not transient or attempt == max_retries - 1:
                raise

            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                delay = retry_after + random.uniform(0, 1)
            else:
                # تجاوز المعدل يحتاج تراجعاً أبطأ من انقطاع اتصال عابر
                eff_base = max(base_delay, 5.0) if is_rate_limit else base_delay
                delay = min(max_delay, eff_base * (2 ** attempt)) + random.uniform(0, 1)

            kind = "تجاوز حدود المعدل (429)" if is_rate_limit else "انقطاع اتصال عابر"
            print(f"    [retry] محاولة {attempt+1}/{max_retries} فشلت — {kind}: {e}\n"
                 f"    إعادة المحاولة بعد {delay:.1f}s...")
            time.sleep(delay)
    raise last_err


class TruncatedResponseError(RuntimeError):
    """الاستجابة انقطعت قبل اكتمالها (تجاوزت max_tokens) — ليست خطأ شبكة."""


class JSONExtractionError(ValueError):
    """تعذّر استخراج JSON صالح من مخرجات النموذج، حتى بعد محاولات الإصلاح."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    return re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()


def _find_json_span(text: str) -> tuple[int, int]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise JSONExtractionError(f"لا يوجد JSON في مخرجات النموذج: {text[:200]!r}")
    return start, end


def _repair_json_text(text: str) -> str:
    """
    إصلاحات شائعة لمخرجات LLM غير الصالحة JSON-حرفياً:
      - فواصل زائدة قبل ] أو }.
      - علامات اقتباس ذكية (" ") بدل القياسية.
      - أسطر جديدة حرفية داخل قيم نصية (يجب أن تكون \\n مهرَّبة).
    لا تضمن هذه الإصلاحات النجاح دائماً — هي محاولة قبل الاستسلام.
    """
    t = text
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = re.sub(r",\s*([\]}])", r"\1", t)              # فاصلة زائدة قبل الإغلاق
    return t


def _close_truncated_json(text: str) -> Optional[str]:
    """
    يعالج القطع الحقيقي (الكائن الجذر نفسه غير مغلق، لا فقط عنصر داخلي).
    يتتبّع مكدس الأقواس المفتوحة؛ عند كل إغلاق داخلي كامل يسجّل "نقطة قطع
    آمنة" مع نسخة من حالة المكدس عندها. يأخذ آخر نقطة آمنة (أكبر محتوى
    محفوظ) ويُغلق يدوياً كل بنية ما تزال مفتوحة عندها.
    """
    stack: list[str] = []
    in_str = False
    escape = False
    safe_cuts: list[tuple[int, list[str]]] = []
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            safe_cuts.append((i + 1, list(stack)))
    if not safe_cuts:
        return None
    cut_pos, remaining = safe_cuts[-1]
    closers = {"{": "}", "[": "]"}
    suffix = "".join(closers[c] for c in reversed(remaining))
    return text[:cut_pos] + suffix


def _extract_json(text: str) -> dict:
    """
    يستخرج JSON من مخرجات LLM بسلسلة محاولات متصاعدة الصرامة، بدل الفشل
    عند أول عائق. يرمي JSONExtractionError برسالة تشخيصية واضحة عند فشل
    الجميع (بدل تراجع Python الخام غير المفيد).
    """
    raw = _strip_fences(text)
    start, end = _find_json_span(raw)
    candidate = raw[start:end + 1]

    # محاولة 1: كما هو
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        last_err = e

    # محاولة 2: إصلاحات نصية شائعة (فواصل زائدة، اقتباسات ذكية)
    try:
        return json.loads(_repair_json_text(candidate))
    except json.JSONDecodeError as e:
        last_err = e

    # محاولة 3: إغلاق القطع الحقيقي (الكائن الجذر نفسه غير مكتمل)
    closed = _close_truncated_json(_repair_json_text(candidate))
    if closed:
        try:
            return json.loads(closed)
        except json.JSONDecodeError as e:
            last_err = e

    # محاولة 4: إغلاق القطع على النص الخام غير المُصلَح (شبكة أمان أخيرة)
    closed_raw = _close_truncated_json(candidate)
    if closed_raw:
        try:
            return json.loads(closed_raw)
        except json.JSONDecodeError as e:
            last_err = e

    snippet_start = max(0, getattr(last_err, "pos", 0) - 80)
    snippet = candidate[snippet_start:snippet_start + 160]
    raise JSONExtractionError(
        f"تعذّر تحليل JSON من مخرجات النموذج بعد محاولات الإصلاح "
        f"({last_err}). على الأرجح استجابة مقطوعة (زد max_tokens أو قلّل "
        f"batch_size) أو تنسيق غير قياسي. مقتطف حول موضع الخطأ: {snippet!r}")


class LLMClient:
    def complete_json(self, messages: List[dict], temperature: float = 0.7) -> dict:
        raise NotImplementedError


class MistralClient(LLMClient):
    """
    مثال إنتاجي عبر Mistral AI.
    يتطلب: pip install "mistralai==1.5.1"  + متغيّر البيئة MISTRAL_API_KEY.
    الواجهة (SDK v1.x): from mistralai import Mistral
                        client.chat.complete(model=..., messages=[...])
    عدّل اسم النموذج حسب المتاح لك (مثلاً: mistral-large-latest,
    mistral-medium-latest, mistral-small-latest).

    ملاحظة عن 429 / Connection reset by peer: شائع مع مفاتيح الطبقة
    المجانية (حدود معدل صارمة جداً) أو شبكات ذات بروكسي/جدار حماية. هذا
    العميل يعيد المحاولة تلقائياً بتراجع أُسّي، ويعطي 429 مهلة أطول
    خصيصاً. تحقّق من حدك الفعلي في: https://admin.mistral.ai (Limits).

    ملاحظة عن الانقطاع (truncated JSON / finish_reason='length'): مخرجات
    مُدفَّعة (عدة أسئلة بنداء واحد) تحتاج max_tokens أكبر بكثير من سؤال
    واحد، خصوصاً بالعربية (كثافة توكن أعلى من الإنجليزية). هذا العميل
    يكتشف الانقطاع صراحةً (لا يخمّنه من خطأ JSON) ويعيد المحاولة تلقائياً
    بحد أقصى أعلى بنسبة 60% (مرتين كحد أقصى) قبل الاستسلام.
    """
    def __init__(self, model: str = "mistral-large-latest", max_tokens: int = 4000,
                max_retries: int = 8, max_retry_delay: float = 60.0,
                timeout_ms: int = 180_000, max_tokens_ceiling: int = 16_000):
        from mistralai import Mistral  # مؤجّل حتى لا يكسر الاستيراد دون تثبيت
        import os
        self.client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"),
                              timeout_ms=timeout_ms)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay
        self.max_tokens_ceiling = max_tokens_ceiling
        self.last_raw_text: str = ""     # لتشخيص الأعطال دون كسر الواجهة

    def _raw_call(self, messages, temperature: float, max_tokens: int) -> tuple[str, str]:
        """يعيد (النص، finish_reason)."""
        resp = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        choice = resp.choices[0]
        return choice.message.content, getattr(choice, "finish_reason", "") or ""

    def complete_json(self, messages, temperature: float = 0.7) -> dict:
        budget = self.max_tokens
        text = ""
        for escalation in range(3):          # المحاولة الأصلية + تصعيدان
            def _call():
                return self._raw_call(messages, temperature, budget)

            text, finish_reason = _retry_with_backoff(
                _call, max_retries=self.max_retries, max_delay=self.max_retry_delay)
            self.last_raw_text = text

            if finish_reason == "length" and budget < self.max_tokens_ceiling:
                new_budget = min(self.max_tokens_ceiling, int(budget * 1.6))
                print(f"    [truncation] استجابة مقطوعة (finish_reason=length) "
                     f"عند max_tokens={budget}؛ إعادة المحاولة بـ {new_budget}...")
                budget = new_budget
                continue
            break

        try:
            return _extract_json(text)
        except JSONExtractionError as e:
            raise JSONExtractionError(
                f"{e}\n    تلميح: جرّب --batch-size أصغر أو --max-tokens أكبر "
                f"(الحالي: {budget}).") from e


class MockClient(LLMClient):
    """
    يحاكي مخرجات مولّد/محقّق دون شبكة، لإثبات أن الـ pipeline ووحدات التحقق
    تعمل من الطرف للطرف. يدعم النداءات المفردة والمُدفَّعة (batched).
    """
    def __init__(self):
        self._toggle = 0

    # ---- سؤال قالب صحيح (منوَّع لكل خانة حتى لا يعلَق في dedup) ---- #
    @staticmethod
    def _one_question(slot_index: int) -> dict:
        variants = [
            ("لتكن A = 0.9/1 + 0.5/2 + 0.3/3 + 0.7/4 على X. "
             "حدّد alpha-cut عند α=0.5 للمجموعة الصبابية A."),
            ("Given the fuzzy set B = 0.9/1 + 0.5/2 + 0.3/3 + 0.7/4, "
             "compute the crisp set of elements whose membership ≥ 0.5."),
            ("إذا كانت درجات الانتماء للعناصر {1,2,3,4} هي "
             "(0.9, 0.5, 0.3, 0.7) على الترتيب، فما مجموعة القطع عند 0.5؟"),
            ("Consider μ(1)=0.9, μ(2)=0.5, μ(3)=0.3, μ(4)=0.7. "
             "Determine the 0.5-level set."),
            ("في نظام استدلال صبابي، المجموعة C معرّفة بـ "
             "0.9/1 + 0.5/2 + 0.3/3 + 0.7/4. أوجد C₀.₅ (alpha-cut)."),
            ("Apply the alpha-cut operation (α=0.5) to the discrete fuzzy "
             "set with memberships 0.9, 0.5, 0.3, 0.7 over {1,2,3,4}."),
            ("احسب scalar cardinality للمجموعة الصبابية المعطاة درجات "
             "انتمائها أدناه، ثم قارنها بعتبة القطع."),
            ("Which crisp subset results from thresholding the membership "
             "function below at the given alpha level?"),
            ("طبّق مبدأ القطع على المجموعة الغائمة التالية واذكر العناصر "
             "الناتجة مرتبةً تصاعدياً."),
            ("For the fuzzy relation values listed below, extract the "
             "support elements satisfying the alpha condition."),
            ("بيّن أي العناصر التالية تنتمي إلى مجموعة القطع القوية "
             "(strong alpha-cut) للمجموعة المعطاة."),
            ("Identify the level set induced by the threshold on the "
             "membership grades enumerated below."),
        ]
        mu = [round(0.9 - slot_index * 0.017, 2),
              round(0.5 + slot_index * 0.013, 2),
              round(0.3 + slot_index * 0.011, 2),
              round(0.7 - slot_index * 0.019, 2)]
        base = variants[slot_index % len(variants)]
        uniq = (f" المجموعة المدروسة: {mu[0]}/{slot_index+1} + "
                f"{mu[1]}/{slot_index+2} + {mu[2]}/{slot_index+3} + "
                f"{mu[3]}/{slot_index+4}.")
        return {
            "slot_index": slot_index,
            "question": f"[Q{slot_index}] {base}{uniq}",
            "verification_code": (
                "import json\n"
                "mu = {1: 0.9, 2: 0.5, 3: 0.3, 4: 0.7}\n"
                "cut = sorted(x for x, m in mu.items() if m >= 0.5)\n"
                "print(json.dumps({'answer': cut}))\n"
            ),
            "options": [
                {"label": "A", "text": "{1, 3, 4}", "is_correct": False},
                {"label": "B", "text": "{1, 2, 4}", "is_correct": True},
                {"label": "C", "text": "{1, 4}", "is_correct": False},
                {"label": "D", "text": "{1, 2, 3, 4}", "is_correct": False},
                {"label": "E", "text": "∅", "is_correct": False},
            ],
            "correct_answer": "B",
            "worked_solution": "نأخذ العناصر التي μ ≥ α.",
            "provenance_chunk_ids": ["mock_chunk"],
        }

    def complete_json(self, messages, temperature: float = 0.7) -> dict:
        sys_msg = messages[0]["content"]
        user = messages[-1]["content"]

        # 1) حلّ دفعة (cross-model solver)
        if "independent expert exam solver" in sys_msg:
            qids = re.findall(r"\[question_id:\s*([^\]]+)\]", user)
            return {"solutions": [
                {"question_id": qid.strip(), "chosen": "B",
                 "reasoning": "α-cut = {x: μ(x) ≥ α}", "confidence": 0.9}
                for qid in qids]}

        # 2) إصلاح دفعة
        if "exam-question editor" in sys_msg:
            idxs = [int(x) for x in re.findall(r"\[slot_index:\s*(\d+)\]", user)]
            return {"questions": [self._one_question(i) for i in idxs]}

        # 3) توليد دفعة
        if "MULTIPLE" in sys_msg or "slot_index=" in user:
            idxs = [int(x) for x in re.findall(r"slot_index=(\d+)", user)]
            return {"questions": [self._one_question(i) for i in idxs]}

        # 4) محقّق أعمى مفرد (النظام القديم)
        if "حُلّه بنفسك" in user or "independent expert grader" in sys_msg.lower():
            return {"chosen": "B", "reasoning": "0.5-cut = {عناصر μ≥0.5}",
                    "confidence": 0.9}

        # 5) مولّد مفرد (النظام القديم)
        q = self._one_question(1)
        q["verification_code"] = (
            "import json\n"
            "mu = {1: 0.9, 2: 0.5, 3: 0.3, 4: 0.7}\n"
            "cut = sorted(x for x, m in mu.items() if m >= 0.5)\n"
            "print(json.dumps({'answer': cut}))\n"
        )
        q["computation_spec"] = {
            "domain": "fuzzy_logic",
            "sets": {"A": "0.9/1 + 0.5/2 + 0.3/3 + 0.7/4"},
            "operation": "alpha_cut",
            "params": {"set": "A", "alpha": 0.5},
        }
        q["distractor_rationale"] = "A: نسي 2؛ C: strong cut؛ D: الكل."
        return q
