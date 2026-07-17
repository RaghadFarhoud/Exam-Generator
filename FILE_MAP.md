# خريطة ملفات المشروع — أين كل ملف وماذا يفعل

```
exam_generator/                    ← المجلد الجذر للمشروع (ضعه أينما تريد)
│
├── main.py                        ★ نقطة الدخول الرئيسية (CLI بخمسة أوامر)
├── run_demo.py                      تشغيل تجريبي سريع دون شبكة (MockClient)
├── audit_bank.py                    تدقيق مفاتيح إجابة بنك الأسئلة الأصلي
├── README.md                        التوثيق الكامل: المعمارية + أوامر التشغيل
│
├── exam_gen/                      ← الحزمة الأساسية (كل منطق النظام هنا)
│   ├── __init__.py                  تصدير الواجهات العامة للحزمة
│   │
│   │  ── تحميل البيانات والتخطيط ──
│   ├── data.py                      تحميل ودمج ملفات المحاضرات والامتحانات
│   │                                (ملف واحد / قائمة / مجلد كامل / glob)
│   ├── blueprint.py                 جدول المواصفات: توزيع الأسئلة على
│   │                                (موضوع × مستوى بلوم × صعوبة) + وضع
│   │                                التغطية الشاملة (--coverage)
│   │
│   │  ── الاسترجاع والتوليد ──
│   ├── retrieval.py                 استرجاع مزدوج لكل سؤال: grounding من
│   │                                المحاضرات + أمثلة أسلوب من الامتحانات
│   │                                (هجين: كلمات مفتاحية + embeddings)
│   ├── embeddings.py                محوّلا embeddings جاهزان: محلي مجاني
│   │                                (SentenceTransformer) أو عبر API (Voyage)
│   ├── prompts.py                   قوالب المولّد والمحقّق الأعمى — عامة لأي
│   │                                مادة، واللغة (ar/en/mixed) تُقرأ تلقائياً
│   │                                من بروفايل الأستاذ
│   ├── llm_client.py                واجهة LLM قابلة للتبديل: MistralClient
│   │                                للإنتاج + MockClient للاختبار دون شبكة
│   │
│   │  ── التحقق (قلب دقة النظام) ──
│   ├── verify.py                    بوابة الجودة متعددة الطبقات: بنية ←
│   │                                تحقق حسابي ← جِدّة ← محقّق أعمى؛
│   │                                القرار: verified/needs_review/rejected
│   ├── code_verifier.py             التحقق العام لأي مادة: تنفيذ كود Python
│   │                                مولّد في عملية معزولة (فحص ساكن + مهلة)
│   │                                ومقارنة الناتج بالخيار المُدّعى
│   ├── domains.py                   سجل إضافات المواد (plugins) — إضافة أي
│   │                                مادة جديدة = دالة evaluator واحدة
│   ├── fuzzy_math.py                إضافة المنطق الصبابي (مثال plugin):
│   │                                extension principle، alpha-cuts، t-norms،
│   │                                defuzzification... — اختيارية للنظام
│   │
│   │  ── التقييم الأكاديمي (فصل النتائج) ──
│   ├── evaluate.py                  مطابقة الأسلوب كمياً (JSD) + مواد اختبار
│   │                                التمييز الأعمى + دراسة الحذف (Ablation)
│   │                                + قياس دقة مفاتيح أي دفعة أسئلة
│   │
│   └── pipeline.py                  المنسّق: يربط كل ما سبق من المدخلات إلى
│                                    امتحان متحقَّق منه بنفس schema بياناتك
│
└── tests/                         ← الاختبارات (شغّلها بعد أي تعديل)
    ├── test_fuzzy.py                9 اختبارات لمحرك الرياضيات الصبابية
    └── test_code_verifier.py        7 اختبارات للتحقق العام (مواد متعددة:
                                     إحصاء، خوارزميات، منطق، عربي/إنجليزي)
```

## مدخلات النظام (ملفاتك أنت — ليست في الحزمة)

رتّبها هكذا بجانب المشروع:
```
data/
├── exams/          ← كل ملفات أسئلة الدورات (بصيغة all_exam_questions.json)
├── lectures/       ← كل المحاضرات المستخرجة (بصيغة normalized_knowledge.json)
└── doctor_style_profile.json
```

## أوامر التشغيل الأساسية

```bash
# اختبار أن كل شيء سليم بعد فك الضغط (دون شبكة ودون مفاتيح):
python3 tests/test_fuzzy.py
python3 tests/test_code_verifier.py
python3 run_demo.py

# توليد امتحان شامل (أضف --mock للتجربة دون LLM):
python3 main.py generate --exams data/exams --lectures data/lectures \
    --style data/doctor_style_profile.json \
    -n 40 --coverage --course "اسم المادة" -o exam.json

# للإنتاج الحقيقي: pip install "mistralai==1.5.1" + متغير البيئة MISTRAL_API_KEY
# ثم احذف --mock. لمادتك تحديداً أضف: --domain fuzzy_logic

# التقييم لفصل النتائج:
python3 main.py audit    --file exam.json
python3 main.py evaluate --exams data/exams --lectures data/lectures \
    --style data/doctor_style_profile.json --generated exam.json
python3 main.py ablation --exams data/exams --lectures data/lectures \
    --style data/doctor_style_profile.json -n 15
python3 main.py blindtest --exams data/exams --lectures data/lectures \
    --style data/doctor_style_profile.json --generated exam.json -o blind/
```
