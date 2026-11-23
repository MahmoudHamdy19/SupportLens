# الخطوات المتبقية لإكمال SupportLens (بالترتيب)

الملف ده بيكمّل `AGENTS.md`، ومش بيلغي أي قاعدة فيه. كل الأوامر بتتشغّل من فولدر المشروع.
الخطوات المكتوب جنبها **بشري** يعملها ناس حقيقيين. ممنوع أي سكربت أو AI أو مساعد يعملها أو يقترح أو يملأ أي حاجة فيها.

---

## الخطوة 0: أرشفة التشغيل السابق (آلي)

التعليقات والتحكيم والنتائج اللي فاتت ما ينفعش يتعاد استخدامها، لأن العينة هتتغير. **انقلها، متحذفهاش**، لفولدر `archive\run_2026-11-19\` في جذر المشروع:

- `data\annotations\`: كل الملفات المتولدة، **ما عدا** `annotation_guidelines.md` و`label_studio_config.xml` و`README.md` (دول يفضلوا مكانهم).
- `data\ground_truth\`: كل الملفات ما عدا `README.md`.
- `data\evaluation\`: كل الملفات وفولدر `expert\` ما عدا `README.md`.
- `models\`: أي أوزان متدربة لو موجودة.
- في `README.md` رجّع قيم جدول النتائج لـ `not run yet` من غير ما تغيّر شكل الـ markers.

**متلمسش** `data\processed\` ولا `data\audit\`، لأن الجمع بيكمّل منهم.

## الخطوة 1: جمع 600 سياسة (آلي)

الرسالة (§3.3.1) بتقول "عينة طبقية عشوائية 300"، فلازم يتجمع عدد أكبر يتسحب منه.
```
.venv\Scripts\python scripts\collect_dataset.py --target 600
```
الجمع بيكمّل من الـ 300 الموجودين. لو وقف، شغّل نفس الأمر تاني. ومينفعش جهازين يجمعوا في نفس الوقت.

## الخطوة 2: سحب العينة (آلي)
```
.venv\Scripts\python scripts\sample_dataset.py
```
بيختار 300 من الـ 600 بطريقة طبقية: فئة النجوم × وجود ثغرة، بـ seed 42.

## الخطوة 3: تجهيز ملفات التعليق (آلي)
```
.venv\Scripts\python scripts\prepare_annotation.py
```
الملفات بتطلع في `data\annotations\label_studio\`، ملف لكل معلّق بترتيب مختلف.

**توقّف هنا وابعت المشروع للمراجعة قبل ما التعليق يبدأ.**

## الخطوة 4: التعليق (بشري، معلّقين اتنين)

- معلّقين **اتنين مختلفين**، **كل واحد بحساب Label Studio منفصل**.
- إعدادات الواجهة: `data\annotations\label_studio_config.xml`.
- المعلّق الأول يستورد `tasks_annotator1.json` بس، والتاني `tasks_annotator2.json` بس.
- كل واحد يقرأ `data\annotations\annotation_guidelines.md` الأول.
- ميشوفوش شغل بعض، ومحدش يستخدم AI.
- لكل سياسة بيعلّموا:
  - جملة الالتزام t، حرفياً من النص.
  - الإصدارات v.
  - نوع الالتزام τ.
  - المرجع الزمني T، وجملته في النص لو موجودة.
  - الحكم المتوقع (واحد من الخمسة).
- خد وقتك في كل سياسة؛ الطبيعي دقايق مش ثواني.
- في الآخر كل واحد يعمل Export بصيغة JSON ويحطه في `data\annotations\` بالاسم `annotator1_export.json` أو `annotator2_export.json`.

## الخطوة 5: حساب الاتفاق (آلي)
```
.venv\Scripts\python scripts\check_agreement.py
```
بيطلع `agreement_report.json` فيه kappa ولستة الخلافات. هدف الرسالة κ ≥ 0.80. لو أقل، المعلّقين يراجعوا دليل التعليق والحالات المختلف عليها قبل التحكيم.

## الخطوة 6: تحكيم المشرف (بشري)
```
.venv\Scripts\python scripts\adjudicate.py --template
```
بيطلع ورقة فيها كل خلاف. **المشرف بنفسه** يقرر الصح في كل حالة ويكتب سبب حقيقي، مش جملة واحدة متكررة، ويحفظها `data\annotations\adjudication.json`. بعدها:
```
.venv\Scripts\python scripts\adjudicate.py
```
ده بيعمل `data\ground_truth\ground_truth.jsonl`.

**ابعت المشروع للمراجعة هنا.**

## الخطوة 7: التقسيم 70/15/15 (آلي)
```
.venv\Scripts\python scripts\make_splits.py
```

## الخطوة 8: التدريب والتقييم على GPU (آلي على Colab)

افتح `notebooks\SupportLens_Colab_Train_and_Eval.ipynb` على Colab بـ GPU (T4)، وشغّله من الأول للآخر. بيدرّب Qwen2.5-7B + LoRA وRoBERTa، ويشغّل:
```
python scripts/run_evaluation.py --strict
```
رجّع ملفات `data\evaluation\` (والأوزان في `models\` لو عايزها) للمشروع. ممنوع تعديل أي ملف نتائج بإيدك.

## الخطوة 9: تقييم الخبراء (بشري)
```
.venv\Scripts\python scripts\run_expert_evaluation.py --prepare
```
بيطلع ورقة تقييم لكل خبير في `data\evaluation\expert\`. خبيرين حقيقيين على الأقل، كل واحد لوحده، يدّوا درجة من 1 لـ 5 في الصحة والاكتمال وقابلية التتبع. بعدها:
```
.venv\Scripts\python scripts\run_expert_evaluation.py --ingest
```

## الخطوة 10: الأرقام النهائية (آلي)
```
.venv\Scripts\python scripts\render_results.py
.venv\Scripts\python scripts\verify_results.py
```
لازم يطلع `VERIFICATION PASSED`، وساعتها الأرقام دي هي اللي تنكتب في الفصل 4 كما هي، حتى لو ضعيفة. **ابعت المشروع للمراجعة النهائية.**

---

## قواعد ثابتة طول الخطوات

- متعدّلش الكود إلا لو صاحب المشروع طلب، ومتعدّلش الرسالة.
- مفيش أي تسمية أو تقييم أو رقم يتعمل بسكربت أو AI أو باليد.
- `.env` (فيه الـ token) ميتبعتش ولا يدخل أي zip.
- لما تضغط المشروع، شيل `.venv` و`logs` و`data\cache` و`__pycache__` و`.env`.
