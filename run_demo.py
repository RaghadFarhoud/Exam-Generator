"""
run_demo.py — تشغيل الـ pipeline من الطرف للطرف بدون شبكة (MockClient).

للتشغيل الحقيقي: استبدل MockClient بـ MistralClient (يتطلب مفتاح + شبكة):
    from exam_gen import MistralClient
    llm = MistralClient(model="mistral-large-latest")
    # يتطلب: pip install "mistralai==1.5.1"  +  متغيّر MISTRAL_API_KEY
"""
import os
from exam_gen import load_corpus, ExamPipeline, save_exam, MockClient

HERE = os.path.dirname(os.path.abspath(__file__))
UP = "/mnt/user-data/uploads"


def main():
    corpus = load_corpus(
        exam_path=os.path.join(UP, "all_exam_questions.json"),
        knowledge_path=os.path.join(UP, "normalized_knowledge.json"),
        style_path=os.path.join(UP, "doctor_style_profile.json"),
    )
    print(f"محمّل: {len(corpus.exam_questions)} سؤال، "
          f"{len(corpus.knowledge_chunks)} chunk، "
          f"{len(corpus.topics)} موضوع\n")

    llm = MockClient()   # <-- استبدلها بـ MistralClient للإنتاج
    pipeline = ExamPipeline(corpus, llm, require_blind=True)

    result = pipeline.generate_exam(n_questions=5, seed=7, verbose=True)

    out_path = os.path.join(HERE, "generated_exam_demo.json")
    save_exam(result, out_path, include_review=True)
    print(f"\nحُفظ الامتحان في: {out_path}")

    # اعرض عيّنة سؤال متحقَّق منه
    if result["verified"]:
        q = result["verified"][0]
        print("\n--- عيّنة سؤال متحقَّق منه حسابياً ---")
        print(q["question"])
        for o in q["options"]:
            mark = " (✓)" if o["is_correct"] else ""
            print(f"  {o['label']}) {o['text']}{mark}")
        print("طبقات التحقق:")
        for layer in q["verification"]["layers"]:
            print(f"   - {layer['name']}: "
                  f"{'✓' if layer['passed'] else '✗'} {layer['detail']}")


if __name__ == "__main__":
    main()
