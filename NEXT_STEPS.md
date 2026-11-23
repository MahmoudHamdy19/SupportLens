# الخطوات المتبقية لإكمال SupportLens (بالترتيب)

الملف ده بيكمّل `AGENTS.md`، ومش بيلغي أي قاعدة فيه. كل الأوامر بتتشغّل من فولدر المشروع.
الخطوات المكتوب جنبها **بشري** يعملها ناس حقيقيين. ممنوع أي سكربت أو AI أو مساعد يعملها أو يقترح أو يملأ أي حاجة فيها.

---

## تحديث 23 نوفمبر 2026: بعد مراجعة الكود (اقرأه الأول على الجهاز التاني)

### اللي اتغيّر في الكود (branch `fix/review-findings`)

| البند | التغيير | الملف |
|---|---|---|
| NVD 404 | مفتاح `.env.example` الافتراضي بيتجاهل، ولو NVD رجّع 404 بالمفتاح (مفتاح غلط) الكود بيكمّل من غيره ويسجّل `api_key_rejected: true` | `src/layer2_evidence_layer/evidence_sources.py` |
| قص السياسة | لو السياسة أطول من 6000 حرف وقسم "Supported Versions" بعد الحد، النص بيبدأ من القسم ده. وبلوكات PGP بتتشال. النتيجة: الـ 300 سجل كلهم بقوا صالحين للتدريب (قبل كده 207/210 و44/45) | `src/layer1_ai_extraction/prompting.py` |
| urllib3 / latest_only | فحص بعد استخراج Qwen في التشغيل العادي بس (CLI والواجهة): لو الجملة مش حرفية أو مفيهاش تعهد دعم (زي تعليمات الإبلاغ) يرجع للقواعد، ولو فيه رقم إصدار مع `latest_only` يصححها `explicit_range`. **مش بيشتغل في التقييم** (`--strict`)، فأرقام Qwen في الفصل 4 هي مخرجات الموديل نفسه | `src/layer1_ai_extraction/commitment_guard.py` |
| RoBERTa | cross-entropy بأوزان للـ labels النادرة (VERSION). نفس أرقام الرسالة (lr 2e-5، batch 16، 5 epochs). ومقاييس val لكل label بتتكتب في `run_metadata.json` | `src/layer1_ai_extraction/roberta_baseline.py` |
| الإصلاح الجزئي | ranges جوه نفس الـ major (زي `>=2.1,<2.3`) بتتقيّم على مستوى الفرع الفرعي، والوصف اتظبط في الكود والتوثيق | `src/layer3_rule_assessment/rule_engine.py` |
| التعليق | `check_agreement.py` بيرفض ملفات التعليق اللي شكلها متولّد بسكربت (نفس `lead_time` لكل السجلات، أو كلها اتعملت في أقل من دقيقة). الملفات الحالية بيترفضوا بالسبب ده | `src/data_pipeline/annotation.py`, `cli.py` |

### الخطوات على الجهاز التاني بالترتيب

**أ. تجهيز (آلي، مرة واحدة)**
```
git fetch origin
git checkout fix/review-findings
git pull
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
```
الاختبارات لازم تنجح كلها. لو فيه أي فشل، ابعت الـ output كله.

**ب. مفتاح NVD (بشري)**
- اطلب مفتاح مجاني من `https://nvd.nist.gov/developers/request-an-api-key` (بيوصل على الإيميل).
- حطه في `.env` كده: `NVD_API_KEY=المفتاح`. لو مفيش مفتاح، **امسح السطر** وما تسيبش `your_nvd_api_key_here`.
- اختبار سريع:
```
rmdir /s /q data\cache\nvd
.venv\Scripts\python supportlens_cli.py --pypi django
```
افتح آخر تقرير `data\reports\django_*.json`، ودوّر على `"nvd"`. المطلوب `"status": "SUCCESS"` بمفتاح صالح، أو `"PARTIAL"` من غير مفتاح (الحد 25 CVE). لو لقيت `"api_key_rejected": true` يبقى المفتاح غلط.

**ج. أرشفة التشغيل القديم (آلي):** اعمل **الخطوة 0** اللي تحت بالظبط. ملفات التعليق الحالية (`annotator1_export.json` و`annotator2_export.json`) فيها كل الـ 300 تعليق اتعملوا في نفس الثانية وبنفس `lead_time = 180`، فمينفعش تتستخدم. **انقلها للأرشيف، متحذفهاش.**

**د. الخطوات 1 لـ 3 (آلي):** جمع، سحب العينة، تجهيز ملفات Label Studio، زي ما هي مكتوبة تحت.

**هـ. التعليق (بشري، الخطوة 4)، ومهم جداً:**
- كل معلّق يشتغل في Label Studio بنفسه، ويعمل Export من Label Studio مباشرة. الملف لازم يكون فيه أوقات حقيقية.
- **علّموا العبارة الزمنية** (label `TemporalExpression`) في أي سياسة فيها جملة زمنية (زي "for 12 months after release" أو "until the next major release"). في التشغيل القديم كانت صفر، فـ RoBERTa ما اتدربش عليها. لو السياسة مفيهاش جملة زمنية، سيبوها فاضية عادي.
- بعدها `check_agreement.py` (الخطوة 5). لو طلع `error: the annotator exports look script-generated`، **متكمّلش**. ارجع للمعلّقين.

**و. التحكيم والتقسيم (الخطوات 6 و7):** زي ما هي.

**ز. التدريب على Colab (الخطوة 8):**
- لازم تدرّب **Qwen وRoBERTa من جديد**، لأن البرومبت وتجهيز النص اتغيروا. الـ adapter القديم مبقاش صالح.
- بعد Qwen: افتح `models\qwen_lora_adapters\run_metadata.json` واكتب `records` و`used` و`skipped` لكل من train وval. دي الأعداد اللي تتكتب في الرسالة.
- بعد RoBERTa: في الـ log دوّر على `val B-VERSION: f1=`. لو الـ F1 صفر أو `predicted=0`، **وقّف وابعتلي** قبل التقييم.
- بعدها `run_evaluation.py --strict` زي ما هو.

**ح. باقي الخطوات (9 و10):** تقييم الخبراء، ثم `render_results.py` و`verify_results.py`.

### اللي يتكتب في الرسالة (بشري، صاحب الرسالة يكتبه بنفسه)
- الفصل 3: دور NVD الفعلي (بيكمّل تاريخ النشر بس، والتأثر والإصلاح من OSV وGitHub Advisory).
- الفصل 3: RoBERTa بـ class-weighted cross-entropy، وتعريف الإصلاح الجزئي على مستوى الفرع (major أو major.minor)، وتجهيز نص السياسة (أولوية قسم الدعم وشيل PGP).
- الفصل 4: أعداد التدريب الفعلية، وقيد البيانات الزمنية لو فضلت قليلة.

### لو حاجة فشلت
ابعت للمساعد: الأمر اللي شغلته، والـ output كامل، وملف التقرير أو `run_metadata.json` المرتبط. **متبعتش `.env` أبداً.**

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
