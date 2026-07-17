"""
code_verifier.py — التحقق الحسابي العام (غير المتخصص بمادة).

الفكرة: بدل مكتبة رياضية لكل مادة، المولّد يُخرج مع كل سؤال حسابي
`verification_code`: كود Python مستقل يحسب الإجابة ويطبعها كـ JSON.
النظام ينفّذه في عملية معزولة (subprocess بوضع Python المعزول -I،
بمهلة زمنية، دون وسائط خارجية) ويقارن الناتج بالخيار المُدّعى.

عقد الكود المولّد (يُفرض في الـ prompt):
  - Python قياسي فقط + مكتبات: math, itertools, fractions, statistics, json.
  - لا شبكة، لا ملفات، لا input().
  - آخر سطر:  print(json.dumps({"answer": <النتيجة>}))
    حيث النتيجة رقم أو نص أو قائمة أو قاموس.

ملاحظة أمنية (اذكرها في تقريرك): العزل هنا عملي (عملية منفصلة + مهلة +
وضع -I) ويكفي لمشروع تخرج حيث الكود يولّده نموذجك أنت؛ للإنتاج العام
يوصى بحاوية/صندوق رمل كامل.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
import sys
import tempfile
import os
from typing import Any, Optional, Tuple

ALLOWED_IMPORTS = {"math", "itertools", "fractions", "statistics", "json",
                   "collections", "functools", "random", "decimal", "cmath"}

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)",
                        re.MULTILINE)
_FORBIDDEN = re.compile(
    r"\b(open|exec|eval|__import__|input|os\.|sys\.|subprocess|socket|"
    r"shutil|pathlib|requests|urllib)\b")


def static_check(code: str) -> Optional[str]:
    """فحص ساكن قبل التنفيذ. يعيد رسالة الخطأ أو None إذا سليم."""
    for m in _IMPORT_RE.finditer(code):
        if m.group(1) not in ALLOWED_IMPORTS:
            return f"استيراد غير مسموح: {m.group(1)}"
    if _FORBIDDEN.search(code):
        return "الكود يحوي استدعاءً محظوراً (ملفات/نظام/شبكة)"
    if "print" not in code:
        return "الكود لا يطبع الناتج (العقد يتطلب print(json.dumps({'answer': ...})))"
    return None

def _auto_heal_imports(code: str) -> str:
    """
    علاج حتمي لأشيع خرق للعقد: النموذج يكتب سطراً واحداً وينسى import json.
    نحقن الاستيرادات الناقصة للمكتبات المسموحة المستخدمة فعلاً في الكود —
    صفر كلفة، يستعيد أسئلة كانت ستفقد حَكَمها الحتمي.
    """
    prelude = []
    for mod in ("json", "math", "statistics", "itertools", "fractions",
                "collections", "functools", "decimal"):
        if re.search(rf"\b{mod}\.", code) and \
           not re.search(rf"^\s*(import|from)\s+{mod}\b", code, re.MULTILINE):
            prelude.append(f"import {mod}")
    return ("\n".join(prelude) + "\n" + code) if prelude else code
def run_verification_code(code: str, timeout: float = 8.0) -> Tuple[bool, Any, str]:
    """
    ينفّذ الكود في عملية معزولة.
    يعيد (نجح؟, الناتج answer, تفصيل/خطأ).
    """
    code = _auto_heal_imports(code)
    err = static_check(code)
    if err:
        return False, None, f"فشل الفحص الساكن: {err}"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", path],
            capture_output=True, text=True, timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"},
        )
    except subprocess.TimeoutExpired:
        return False, None, f"تجاوز المهلة ({timeout}s)"
    finally:
        os.unlink(path)

    if proc.returncode != 0:
        return False, None, f"خطأ تنفيذ: {proc.stderr.strip()[:300]}"

    # آخر سطر JSON في stdout هو الناتج
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if "answer" in obj:
                    return True, obj["answer"], "OK"
            except json.JSONDecodeError:
                continue
    return False, None, f"لم يُطبع JSON بالعقد المطلوب. stdout: {proc.stdout[:200]!r}"


# --------------------------------------------------------------------------- #
#  مقارنة عامة: الناتج المحسوب مقابل نص الخيار المُدّعى
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s{}\[\]()،,]+", " ", s).strip()
    return s


def _nums_of(s: str):
    return [round(float(x), 6) for x in _NUM_RE.findall(str(s))]


def generic_answers_match(computed: Any, claimed: str,
                          tol: float = 1e-6) -> bool:
    """
    مقارنة غير متخصصة بمادة:
      - رقم مقابل رقم (بتفاوت).
      - قائمة أرقام مقابل أرقام مستخرجة من النص (كمجموعات).
      - نص مقابل نص (بعد تطبيع).
      - قاموس {عنصر: قيمة} مقابل تدوين a/x + b/y (يدعم المجموعات الصبابية وغيرها).
    """
    if isinstance(computed, bool):
        c = _norm_text(claimed)
        return c in ({"true", "صح", "صحيح", "yes", "نعم"} if computed
                     else {"false", "خطأ", "خاطئ", "no", "لا"})

    if isinstance(computed, (int, float)):
        nums = _nums_of(claimed)
        return len(nums) >= 1 and abs(float(computed) - nums[0]) < tol \
            if len(nums) == 1 else \
            any(abs(float(computed) - n) < tol for n in nums) and len(nums) == 1

    if isinstance(computed, (list, tuple)):
        comp = sorted(round(float(x), 6) for x in computed)
        got = sorted(_nums_of(claimed))
        return comp == got

    if isinstance(computed, dict):
        # يدعم تدوين "0.3/2 + 0.4/3" أو JSON في نص الخيار
        pairs = re.findall(r"(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)", str(claimed))
        if pairs:
            claimed_map = {round(float(x), 6): round(float(m), 6)
                           for m, x in pairs}
            comp_map = {round(float(k), 6): round(float(v), 6)
                        for k, v in computed.items()}
            keys = set(claimed_map) | set(comp_map)
            return all(abs(claimed_map.get(k, 0) - comp_map.get(k, 0)) < tol
                       for k in keys)
        return _norm_text(json.dumps(computed, sort_keys=True)) == _norm_text(claimed)

    return _norm_text(computed) == _norm_text(claimed)
