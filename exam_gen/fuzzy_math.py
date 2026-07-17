"""
fuzzy_math.py  —  محرك التحقق الحسابي (Computational Verification Engine)

هذه أهم وحدة في النظام. أي سؤال حسابي يُولّده الـ LLM لا يُقبل إلا إذا
أعاد هذا المحرك حساب الإجابة بشكل مستقل وطابَقها.

FuzzySet يُمثَّل كقاموس {element: membership}.
مثال: A = 0.3/2 + 0.4/3 + 0.2/4 + 0.1/5  ==>  {2:0.3, 3:0.4, 4:0.2, 5:0.1}

evaluate_spec() يأخذ "computation_spec" منظّماً يُخرجه المولّد ويعيد الإجابة
الحقيقية — فيصير التحقق: "الكود حسب X، الخيار المُدّعى = X ؟".
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import math
import re

Number = float
EPS = 1e-9


# --------------------------------------------------------------------------- #
#  FuzzySet
# --------------------------------------------------------------------------- #
@dataclass
class FuzzySet:
    """مجموعة صبابية: عنصر -> درجة انتماء في [0,1]."""
    mu: Dict[Number, Number]

    def __post_init__(self):
        clean = {}
        for x, m in self.mu.items():
            m = round(float(m), 10)
            if m < -EPS or m > 1 + EPS:
                raise ValueError(f"membership {m} for {x} خارج [0,1]")
            if m > EPS:                       # نتجاهل الانتماء الصفري
                clean[float(x)] = min(1.0, max(0.0, m))
        self.mu = clean

    # ---- عمليات أساسية ---------------------------------------------------- #
    def membership(self, x: Number) -> Number:
        return self.mu.get(float(x), 0.0)

    def support(self) -> List[Number]:
        return sorted(x for x, m in self.mu.items() if m > EPS)

    def core(self) -> List[Number]:
        return sorted(x for x, m in self.mu.items() if abs(m - 1) < EPS)

    def height(self) -> Number:
        return max(self.mu.values(), default=0.0)

    def is_normal(self) -> bool:
        return abs(self.height() - 1) < EPS

    def alpha_cut(self, alpha: Number, strong: bool = False) -> List[Number]:
        if strong:
            return sorted(x for x, m in self.mu.items() if m > alpha + EPS)
        return sorted(x for x, m in self.mu.items() if m >= alpha - EPS)

    def scalar_cardinality(self) -> Number:      # sigma-count / |A|
        return round(sum(self.mu.values()), 10)

    # ---- عمليات مجموعاتية ------------------------------------------------- #
    def union(self, other: "FuzzySet", tconorm: Callable = None) -> "FuzzySet":
        op = tconorm or (lambda a, b: max(a, b))
        keys = set(self.mu) | set(other.mu)
        return FuzzySet({x: op(self.membership(x), other.membership(x)) for x in keys})

    def intersection(self, other: "FuzzySet", tnorm: Callable = None) -> "FuzzySet":
        op = tnorm or (lambda a, b: min(a, b))
        keys = set(self.mu) | set(other.mu)
        return FuzzySet({x: op(self.membership(x), other.membership(x)) for x in keys})

    def complement(self, kind: str = "standard", w: float = 2.0) -> "FuzzySet":
        keys = set(self.mu)
        out = {}
        for x in keys:
            m = self.membership(x)
            out[x] = _complement_value(m, kind, w)
        # المكمّل يعرّف عادةً على الكون؛ هنا نحدّه بعناصر المجموعة الظاهرة
        return FuzzySet(out)

    def __repr__(self):
        parts = [f"{round(m,4)}/{_fmt(x)}" for x, m in sorted(self.mu.items())]
        return " + ".join(parts) if parts else "∅"


def _fmt(x: Number) -> str:
    return str(int(x)) if abs(x - round(x)) < EPS else str(x)


# --------------------------------------------------------------------------- #
#  دوال المكمّل / t-norms / t-conorms  (تطابق مواضيع الدكتور)
# --------------------------------------------------------------------------- #
def _complement_value(a: float, kind: str, w: float = 2.0) -> float:
    if kind == "standard":
        return 1 - a
    if kind == "sugeno":                 # c_λ(a) = (1-a)/(1+λa)  ؛ w=λ
        return (1 - a) / (1 + w * a)
    if kind == "yager":                  # c_w(a) = (1-a^w)^(1/w)
        return (1 - a ** w) ** (1 / w)
    raise ValueError(f"complement kind غير معروف: {kind}")


TNORMS: Dict[str, Callable[[float, float], float]] = {
    "min":         lambda a, b: min(a, b),
    "product":     lambda a, b: a * b,
    "lukasiewicz": lambda a, b: max(0.0, a + b - 1),
    "drastic":     lambda a, b: (a if abs(b - 1) < EPS else b if abs(a - 1) < EPS else 0.0),
}
TCONORMS: Dict[str, Callable[[float, float], float]] = {
    "max":         lambda a, b: max(a, b),
    "prob_sum":    lambda a, b: a + b - a * b,
    "lukasiewicz": lambda a, b: min(1.0, a + b),
    "drastic":     lambda a, b: (a if abs(b) < EPS else b if abs(a) < EPS else 1.0),
}


# --------------------------------------------------------------------------- #
#  مبدأ التوسّع (Extension Principle)
# --------------------------------------------------------------------------- #
def extension_unary(A: FuzzySet, f: Callable[[float], float]) -> FuzzySet:
    """B = f(A) ؛  μB(y) = sup_{x: f(x)=y} μA(x)."""
    out: Dict[float, float] = {}
    for x, m in A.mu.items():
        y = round(f(x), 10)
        out[y] = max(out.get(y, 0.0), m)
    return FuzzySet(out)


def extension_binary(A: FuzzySet, B: FuzzySet,
                     f: Callable[[float, float], float]) -> FuzzySet:
    """C = f(A,B) ؛  μC(z) = sup_{f(x,y)=z} min(μA(x), μB(y))."""
    out: Dict[float, float] = {}
    for x, ma in A.mu.items():
        for y, mb in B.mu.items():
            z = round(f(x, y), 10)
            out[z] = max(out.get(z, 0.0), min(ma, mb))
    return FuzzySet(out)


_BINOPS: Dict[str, Callable[[float, float], float]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


# --------------------------------------------------------------------------- #
#  مقاييس (تطابق مواضيع: consistency / subsethood / defuzzification)
# --------------------------------------------------------------------------- #
def degree_of_consistency(A: FuzzySet, B: FuzzySet) -> float:
    """poss(A,B) = sup_x min(μA(x), μB(x))."""
    keys = set(A.mu) | set(B.mu)
    return round(max((min(A.membership(x), B.membership(x)) for x in keys), default=0.0), 10)


def degree_of_subsethood(A: FuzzySet, B: FuzzySet) -> float:
    """S(A,B) = |A ∩ B| / |A|  (Kosko)."""
    denom = A.scalar_cardinality()
    if denom < EPS:
        return 1.0
    inter = A.intersection(B).scalar_cardinality()
    return round(inter / denom, 10)


def defuzzify_centroid(A: FuzzySet) -> float:
    """COG = Σ x·μ / Σ μ  (نسخة متقطعة)."""
    num = sum(x * m for x, m in A.mu.items())
    den = sum(A.mu.values())
    return round(num / den, 10) if den > EPS else float("nan")


def defuzzify_mom(A: FuzzySet) -> float:
    """Mean of Maxima."""
    h = A.height()
    maxima = [x for x, m in A.mu.items() if abs(m - h) < EPS]
    return round(sum(maxima) / len(maxima), 10) if maxima else float("nan")


# --------------------------------------------------------------------------- #
#  محلّل تدوين المجموعة الصبابية:  "0.3/2 + 0.4/3 + ..."
# --------------------------------------------------------------------------- #
def parse_fuzzy_set(text: str) -> FuzzySet:
    text = text.replace("−", "-").strip()
    terms = re.split(r"\+(?![^()]*\))", text)
    mu: Dict[float, float] = {}
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if "/" not in t:
            raise ValueError(f"حدّ غير صالح (لا يحوي '/'): {t!r}")
        m_str, x_str = t.split("/", 1)
        mu[float(x_str)] = float(m_str)
    return FuzzySet(mu)


# --------------------------------------------------------------------------- #
#  مُقيّم الـ SPEC:  الواجهة التي يستخدمها المولّد والمُحقّق
# --------------------------------------------------------------------------- #
def evaluate_spec(spec: dict):
    """
    يأخذ computation_spec منظّماً ويعيد (result, human_readable).
    هذا ما يجعل السؤال قابلاً للتحقق آلياً.

    شكل الـ spec:
    {
      "sets": {"A": "0.3/2 + 0.4/3 + 0.2/4 + 0.1/5", "B": "..."},
      "operation": "<اسم العملية>",
      "params": { ... }        # اختياري حسب العملية
    }
    """
    sets = {name: parse_fuzzy_set(s) for name, s in spec.get("sets", {}).items()}
    op = spec["operation"]
    p = spec.get("params", {})

    def S(name):  # اختصار
        return sets[name]

    if op == "membership":
        return S(p["set"]).membership(p["x"]), f"μ_{p['set']}({p['x']})"

    if op == "alpha_cut":
        return S(p["set"]).alpha_cut(p["alpha"], p.get("strong", False)), \
               f"{p['alpha']}-cut of {p['set']}"

    if op == "scalar_cardinality":
        return S(p["set"]).scalar_cardinality(), f"|{p['set']}|"

    if op == "height":
        return S(p["set"]).height(), f"height({p['set']})"

    if op in ("support", "core"):
        return getattr(S(p["set"]), op)(), f"{op}({p['set']})"

    if op == "union":
        tc = TCONORMS[p.get("tconorm", "max")]
        return S(p["a"]).union(S(p["b"]), tc), f"{p['a']} ∪ {p['b']}"

    if op == "intersection":
        tn = TNORMS[p.get("tnorm", "min")]
        return S(p["a"]).intersection(S(p["b"]), tn), f"{p['a']} ∩ {p['b']}"

    if op == "complement":
        return S(p["set"]).complement(p.get("kind", "standard"), p.get("w", 2.0)), \
               f"complement({p['set']})"

    if op == "extension_binary":
        f = _BINOPS[p["binop"]]
        res = extension_binary(S(p["a"]), S(p["b"]), f)
        if "at" in p:
            return res.membership(p["at"]), f"({p['a']} {p['binop']} {p['b']})({p['at']})"
        return res, f"{p['a']} {p['binop']} {p['b']}"

    if op == "consistency":
        return degree_of_consistency(S(p["a"]), S(p["b"])), f"cons({p['a']},{p['b']})"

    if op == "subsethood":
        return degree_of_subsethood(S(p["a"]), S(p["b"])), f"S({p['a']},{p['b']})"

    if op == "defuzzify":
        method = p.get("method", "centroid")
        fn = {"centroid": defuzzify_centroid, "mom": defuzzify_mom}[method]
        return fn(S(p["set"])), f"defuzz_{method}({p['set']})"

    raise ValueError(f"عملية غير مدعومة في الـ spec: {op!r}")


# --------------------------------------------------------------------------- #
#  مطابقة الإجابة: هل الخيار المُدّعى يساوي الناتج المحسوب؟
# --------------------------------------------------------------------------- #
def answers_match(computed, claimed, tol: float = 1e-6) -> bool:
    """يقارن ناتج المحرك بالإجابة المُدّعاة (رقم، أو مجموعة، أو قائمة)."""
    # قائمة (alpha-cut, support, core)
    if isinstance(computed, list):
        c = _parse_list(claimed)
        return c is not None and sorted(c) == sorted(round(x, 6) for x in computed)
    # FuzzySet
    if isinstance(computed, FuzzySet):
        try:
            claimed_fs = claimed if isinstance(claimed, FuzzySet) else parse_fuzzy_set(str(claimed))
        except Exception:
            return False
        keys = set(computed.mu) | set(claimed_fs.mu)
        return all(abs(computed.membership(k) - claimed_fs.membership(k)) < tol for k in keys)
    # رقم
    try:
        return abs(float(computed) - float(_extract_number(claimed))) < tol
    except Exception:
        return False


def _parse_list(s):
    if isinstance(s, (list, tuple)):
        return [round(float(x), 6) for x in s]
    nums = re.findall(r"-?\d+\.?\d*", str(s))
    return [round(float(x), 6) for x in nums] if nums else None


def _extract_number(s):
    if isinstance(s, (int, float)):
        return s
    m = re.search(r"-?\d+\.?\d*", str(s))
    if not m:
        raise ValueError(f"لا يوجد رقم في {s!r}")
    return float(m.group())
