"""
domains.py — سجل إضافات المواد (Domain Plugins).

النظام عام: التحقق الافتراضي لأي مادة = تنفيذ verification_code
(code_verifier.py). إن توفّرت إضافة متخصصة لمادة (مثل fuzzy_logic)
تُستخدم كطبقة تحقق إضافية أعلى دقة، لأنها كود مُراجع ومختبَر مسبقاً.

لإضافة مادة جديدة (مثلاً "graph_algorithms"):
    def my_evaluator(spec: dict): -> (result, label)
    register_domain("graph_algorithms", my_evaluator)
ثم يستطيع المولّد إخراج computation_spec بـ "domain": "graph_algorithms".
"""
from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple, Any

_REGISTRY: Dict[str, Callable[[dict], Tuple[Any, str]]] = {}


def register_domain(name: str, evaluator: Callable[[dict], Tuple[Any, str]]):
    _REGISTRY[name] = evaluator


def get_domain(name: str) -> Optional[Callable]:
    return _REGISTRY.get(name)


def available_domains():
    return sorted(_REGISTRY)


# ---- تسجيل إضافة المنطق الصبابي (اختيارية — النظام يعمل بدونها) ---------- #
try:
    from . import fuzzy_math as _fm

    def _fuzzy_evaluator(spec: dict):
        return _fm.evaluate_spec(spec)

    register_domain("fuzzy_logic", _fuzzy_evaluator)
except Exception:                      # pragma: no cover
    pass
