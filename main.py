"""
main.py — واجهة النظام الموحّدة.

الأوامر:
  generate   توليد امتحان متحقَّق منه (يدعم --coverage للامتحان الشامل)
  audit      تدقيق مفاتيح الإجابة لأي ملف أسئلة يحوي computation_spec
  evaluate   مطابقة الأسلوب كمياً (JSD) بين المولّد والبنك الأصلي
  ablation   دراسة الحذف: أثر طبقات التحقق على دقة المفاتيح
  blindtest  بناء مواد اختبار التمييز الأعمى

أمثلة:
  python3 main.py generate --exams data/exams --lectures data/lectures \\
      --style doctor_style_profile.json -n 40 --coverage -o exam.json
  python3 main.py evaluate --exams data/exams --generated exam.json
  python3 main.py blindtest --exams data/exams --generated exam.json -o blind/
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from exam_gen import load_corpus, ExamPipeline, save_exam, MockClient
from exam_gen import evaluate as EV

from dotenv import load_dotenv


load_dotenv()


def _llm(args):
    if args.mock:
        return MockClient()
    from exam_gen import MistralClient
    return MistralClient(model=args.model, max_retries=args.retries,
                         max_retry_delay=args.max_retry_delay)


def _corpus(args):
    return load_corpus(args.exams, args.lectures, args.style)


def _gateways(args):
    """بوابتان: نموذج توليد ونموذج تحقق مختلف (cross-model)."""
    from exam_gen.gateway import LLMGateway
    if args.mock:
        gen_client, ver_client = MockClient(), MockClient()
    else:
        from exam_gen import MistralClient
        gen_client = MistralClient(model=args.gen_model,
                                   max_retries=args.retries,
                                   max_retry_delay=args.max_retry_delay,
                                   max_tokens=args.max_tokens,
                                   max_tokens_ceiling=args.max_tokens_ceiling)
        ver_client = MistralClient(model=args.verify_model,
                                   max_retries=args.retries,
                                   max_retry_delay=args.max_retry_delay,
                                   max_tokens=args.max_tokens,
                                   max_tokens_ceiling=args.max_tokens_ceiling)
    budget = args.llm_budget if args.llm_budget > 0 else None
    gen_gw = LLMGateway(gen_client, min_interval=args.rate_delay,
                        max_calls=budget)
    ver_gw = LLMGateway(ver_client, min_interval=args.rate_delay,
                        max_calls=budget)
    return gen_gw, ver_gw


def cmd_generate(args):
    corpus = _corpus(args)
    print(f"المدوّنة: {len(corpus.exam_questions)} سؤال، "
          f"{len(corpus.knowledge_chunks)} chunk، {len(corpus.topics)} موضوع")
    from exam_gen.prompts import resolve_language
    print(f"لغة التوليد (من البروفايل): {resolve_language(corpus.style_profile)}")

    from exam_gen.orchestrator import ExamOrchestrator
    gen_gw, ver_gw = _gateways(args)
    orch = ExamOrchestrator(corpus, gen_gw, ver_gw,
                            checkpoint_path=args.checkpoint)
    result = orch.run(args.n, seed=args.seed, coverage=args.coverage,
                      course_description=args.course or "",
                      batch_size=args.batch_size, resume=args.resume)
    save_exam(result, args.out, include_review=args.include_review)
    usage = result["llm_usage"]
    print(f"نداءات LLM: توليد={usage['generator']['total_calls']} "
          f"تحقق={usage['verifier']['total_calls']}")
    print(f"حُفظ في: {args.out}")


def cmd_audit(args):
    data = json.load(open(args.file, encoding="utf-8"))
    qs = data.get("questions", data if isinstance(data, list) else [])
    rep = EV.computational_key_accuracy(qs)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if rep["failures"]:
        sys.exit(1)


def cmd_evaluate(args):
    corpus = _corpus(args)
    gen = json.load(open(args.generated, encoding="utf-8"))
    gen_qs = gen.get("questions", [])
    rep = EV.style_fidelity(corpus.exam_questions, gen_qs)
    out = args.out or "style_fidelity_report.json"
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"overall_mean_jsd = {rep['overall_mean_jsd']}  (0 = تطابق تام)")
    print(f"التقرير الكامل: {out}")


def cmd_ablation(args):
    corpus = _corpus(args)
    pipe = ExamPipeline(corpus, _llm(args), require_blind=not args.no_blind,
                        request_delay=args.rate_delay)
    rep = EV.run_ablation(pipe, n_questions=args.n, seed=args.seed)
    out = args.out or "ablation_report.json"
    json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "note"},
                     ensure_ascii=False, indent=2))
    print(f"التقرير: {out}")


def cmd_blindtest(args):
    corpus = _corpus(args)
    gen = json.load(open(args.generated, encoding="utf-8")).get("questions", [])
    mat = EV.build_blind_test(corpus.exam_questions, gen,
                              n_each=args.n_each, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    p1 = os.path.join(args.out, "participant_sheet.json")
    p2 = os.path.join(args.out, "SECRET_key.json")
    json.dump({"instructions": mat["instructions_ar"],
               "items": mat["participant_sheet"]},
              open(p1, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(mat["secret_key"], open(p2, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"استمارة المشاركين: {p1}\nالمفتاح السري (لا يُوزَّع): {p2}")
    print("بعد جمع الردود: استخدم exam_gen.evaluate.analyze_blind_responses().")


def build_parser():
    p = argparse.ArgumentParser(description="مولّد امتحانات بأسلوب الدكتور")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_corpus=True, needs_llm=False):
        if needs_corpus:
            sp.add_argument("--exams", required=True,
                            help="ملف/مجلد/نمط لأسئلة الدورات السابقة")
            sp.add_argument("--lectures", required=True,
                            help="ملف/مجلد/نمط للمحاضرات المستخرجة")
            sp.add_argument("--style", required=True,
                            help="doctor_style_profile.json")
        if needs_llm:
            sp.add_argument("--mock", action="store_true",
                            help="تشغيل دون شبكة (للاختبار)")
            sp.add_argument("--gen-model", default="mistral-large-latest",
                            help="نموذج التوليد")
            sp.add_argument("--verify-model", default="mistral-small-latest",
                            help="نموذج التحقق المتقاطع (يُفضَّل مختلفاً عن التوليد)")
            sp.add_argument("--model", default="mistral-large-latest",
                            help="(للأوامر القديمة) نموذج واحد")
            sp.add_argument("--no-blind", action="store_true",
                            help="(للأوامر القديمة) تعطيل المحقّق الأعمى")
            sp.add_argument("--retries", type=int, default=8,
                            help="عدد محاولات إعادة الاتصال عند أخطاء "
                                 "الشبكة/تجاوز المعدل (429, Connection reset, ...)")
            sp.add_argument("--max-retry-delay", type=float, default=60.0,
                            help="أقصى انتظار بين محاولة وأخرى (ثوانٍ)")
            sp.add_argument("--rate-delay", type=float, default=3.0,
                            help="ثوانٍ انتظار دنيا بين كل نداء LLM (بوابة مركزية)")
            sp.add_argument("--llm-budget", type=int, default=0,
                            help="حد أقصى لنداءات LLM لكل بوابة (0 = بلا حد)")
            sp.add_argument("--batch-size", type=int, default=6,
                            help="عدد الأسئلة المولّدة في النداء الواحد")
            sp.add_argument("--max-tokens", type=int, default=4000,
                            help="أقصى توكن للاستجابة الواحدة — الدفعات "
                                 "الكبيرة والعربية تحتاج قيمة أعلى")
            sp.add_argument("--max-tokens-ceiling", type=int, default=16000,
                            help="سقف التصعيد التلقائي عند اكتشاف انقطاع "
                                 "الاستجابة (finish_reason=length)")
            sp.add_argument("--resume", action="store_true",
                            help="استئناف تشغيل منقطع من آخر نقطة حفظ")
            sp.add_argument("--checkpoint", default=".exam_checkpoint.json",
                            help="مسار ملف نقطة الحفظ")
        sp.add_argument("--seed", type=int, default=0)

    g = sub.add_parser("generate"); common(g, needs_llm=True)
    g.add_argument("-n", type=int, default=30)
    g.add_argument("--coverage", action="store_true",
                   help="امتحان شامل: كل موضوع مرة على الأقل")
    g.add_argument("--course", default="",
                   help="وصف حر للمادة (أي مادة، أي لغة)")
    g.add_argument("--domain", default=None,
                   help="إضافة مادة مسجّلة (مثل fuzzy_logic) — اختياري")
    g.add_argument("--include-review", action="store_true")
    g.add_argument("-o", "--out", default="generated_exam.json")
    g.set_defaults(fn=cmd_generate)

    a = sub.add_parser("audit"); a.add_argument("--file", required=True)
    a.set_defaults(fn=cmd_audit)

    e = sub.add_parser("evaluate"); common(e)
    e.add_argument("--generated", required=True)
    e.add_argument("-o", "--out")
    e.set_defaults(fn=cmd_evaluate)

    ab = sub.add_parser("ablation"); common(ab, needs_llm=True)
    ab.add_argument("-n", type=int, default=10)
    ab.add_argument("-o", "--out")
    ab.set_defaults(fn=cmd_ablation)

    b = sub.add_parser("blindtest"); common(b)
    b.add_argument("--generated", required=True)
    b.add_argument("--n-each", type=int, default=10)
    b.add_argument("-o", "--out", default="blind_test")
    b.set_defaults(fn=cmd_blindtest)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
