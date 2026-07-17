"""
gateway.py — بوابة نداءات LLM المركزية (الطبقة التي تمنع فوضى الطلبات).

المبدأ الهندسي: كل نداء LLM في النظام كله يمر من هنا حصراً. هذا يضمن:
  1) التسلسل: يستحيل معمارياً إرسال طلبين متوازيين.
  2) التهدئة: مهلة دنيا إلزامية بين كل نداءين (min_interval).
  3) الميزانية: عدّاد نداءات بحد أقصى — يفشل مبكراً بوضوح بدل استنزاف الحصة.
  4) السجل: كل نداء يُسجَّل (الغرض، الزمن، النجاح) لتقرير الشفافية.

البوابة تغلّف أي LLMClient (مولّد أو محقّق) دون تغيير واجهته.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional


class BudgetExceeded(RuntimeError):
    """تجاوزت النداءات الميزانية المحددة — أوقفنا مبكراً بدل استنزاف الحصة."""


@dataclass
class CallRecord:
    purpose: str
    started_at: float
    duration_s: float
    ok: bool
    error: str = ""


@dataclass
class LLMGateway:
    client: object                       # أي LLMClient (complete_json)
    min_interval: float = 3.0            # ثوانٍ دنيا بين نداءين متتاليين
    max_calls: Optional[int] = None      # ميزانية النداءات (None = بلا حد)
    log: List[CallRecord] = field(default_factory=list)
    _last_call_at: float = field(default=0.0, repr=False)

    @property
    def calls_used(self) -> int:
        return len(self.log)

    def complete_json(self, messages, temperature: float = 0.7,
                      purpose: str = "unspecified") -> dict:
        if self.max_calls is not None and self.calls_used >= self.max_calls:
            raise BudgetExceeded(
                f"بلغت ميزانية النداءات ({self.max_calls}). "
                f"ارفعها بـ --llm-budget أو قلّل عدد الأسئلة.")

        # تهدئة إلزامية بين النداءات (تحترم حدود الطبقة المجانية)
        wait = self.min_interval - (time.time() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)

        t0 = time.time()
        try:
            out = self.client.complete_json(messages, temperature=temperature)
            self.log.append(CallRecord(purpose, t0, time.time() - t0, True))
            self._last_call_at = time.time()
            return out
        except Exception as e:
            self.log.append(CallRecord(purpose, t0, time.time() - t0, False, str(e)))
            self._last_call_at = time.time()
            raise

    def report(self) -> dict:
        by_purpose = {}
        for r in self.log:
            d = by_purpose.setdefault(r.purpose, {"calls": 0, "failed": 0,
                                                  "total_s": 0.0})
            d["calls"] += 1
            d["total_s"] = round(d["total_s"] + r.duration_s, 1)
            if not r.ok:
                d["failed"] += 1
        return {"total_calls": self.calls_used, "by_purpose": by_purpose}
