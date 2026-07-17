"""tests/test_fuzzy.py — يثبت صحّة محرك الرياضيات الصبابية."""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exam_gen import fuzzy_math as fm


def approx(a, b, t=1e-9):
    return abs(a - b) < t


def test_parse_and_cardinality():
    A = fm.parse_fuzzy_set("0.3/2 + 0.4/3 + 0.2/4 + 0.1/5")
    assert approx(A.scalar_cardinality(), 1.0)
    assert A.membership(3) == 0.4
    assert A.membership(99) == 0.0


def test_alpha_cut():
    A = fm.parse_fuzzy_set("0.9/1 + 0.5/2 + 0.3/3 + 0.7/4")
    assert A.alpha_cut(0.5) == [1.0, 2.0, 4.0]
    assert A.alpha_cut(0.5, strong=True) == [1.0, 4.0]


def test_union_intersection_min_max():
    A = fm.parse_fuzzy_set("0.2/1 + 0.8/2")
    B = fm.parse_fuzzy_set("0.5/1 + 0.3/2")
    assert A.union(B).mu == {1.0: 0.5, 2.0: 0.8}
    assert A.intersection(B).mu == {1.0: 0.2, 2.0: 0.3}


def test_tnorm_product_lukasiewicz():
    A = fm.parse_fuzzy_set("0.6/1")
    B = fm.parse_fuzzy_set("0.5/1")
    assert approx(A.intersection(B, fm.TNORMS["product"]).membership(1), 0.30)
    assert approx(A.intersection(B, fm.TNORMS["lukasiewicz"]).membership(1), 0.10)


def test_complement_standard():
    A = fm.parse_fuzzy_set("0.3/1 + 0.7/2")
    c = A.complement()
    assert approx(c.membership(1), 0.7)
    assert approx(c.membership(2), 0.3)


def test_extension_principle_binary():
    A = fm.parse_fuzzy_set("0.3/2 + 0.4/3 + 0.2/4 + 0.1/5")
    AA = fm.extension_binary(A, A, lambda a, b: a + b)
    # (A+A)(7): sup min over {(2,5),(3,4),(4,3),(5,2)} = 0.2
    assert approx(AA.membership(7), 0.2)
    # (A+A)(6): {(2,4),(3,3),(4,2)} -> min(0.3,0.2),min(0.4,0.4),min(0.2,0.3)=0.4
    assert approx(AA.membership(6), 0.4)


def test_defuzzify_centroid():
    A = fm.parse_fuzzy_set("1/1 + 0.6/2 + 0.2/3")
    # (1*1 + 0.6*2 + 0.2*3) / (1+0.6+0.2) = 2.8/1.8
    assert approx(fm.defuzzify_centroid(A), 2.8 / 1.8)


def test_subsethood():
    A = fm.parse_fuzzy_set("0.4/1 + 0.6/2")
    B = fm.parse_fuzzy_set("1/1 + 1/2")
    assert approx(fm.degree_of_subsethood(A, B), 1.0)


def test_spec_evaluator_and_match():
    spec = {"sets": {"A": "0.9/1 + 0.5/2 + 0.3/3 + 0.7/4"},
            "operation": "alpha_cut", "params": {"set": "A", "alpha": 0.5}}
    computed, _ = fm.evaluate_spec(spec)
    assert fm.answers_match(computed, "{1, 2, 4}")
    assert not fm.answers_match(computed, "{1, 4}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
