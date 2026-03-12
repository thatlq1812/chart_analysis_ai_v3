# Bao cao Phan hoi - Phan tich Kien truc Stage 3 va Dinh huong Nang cap

| Phien ban | Ngay | Tac gia | Mo ta |
| --- | --- | --- | --- |
| 2.0.0 | 2026-03-05 | AI Agent (Senior ML Engineer) | Cap nhat: OCR root cause, User review, Lo trinh MVP/Stretch |
| 1.0.0 | 2026-03-05 | AI Agent (Senior ML Engineer) | Bao cao phan hoi instruction6.md |

**Tai lieu tham chieu:** `instruction6.md` - Thao luan voi Gemini ve Stage 3 OCR & Extraction

---

## Muc luc

1. [Tom tat noi dung thao luan](#1-tom-tat-noi-dung-thao-luan)
2. [Danh gia tung luan diem](#2-danh-gia-tung-luan-diem)
3. [Doi chieu voi bang chung thuc te tu he thong](#3-doi-chieu-voi-bang-chung-thuc-te-tu-he-thong)
4. [Phan tich bo sung va goc nhin khac](#4-phan-tich-bo-sung-va-goc-nhin-khac)
5. [Phat hien moi: Nguyen nhan goc re OCR texts=[]](#5-phat-hien-moi-nguyen-nhan-goc-re-ocr-texts)
6. [Phan hoi tu System Architect (User Review)](#6-phan-hoi-tu-system-architect-user-review)
7. [Lo trinh hanh dong da cap nhat](#7-lo-trinh-hanh-dong-da-cap-nhat)

---

## 1. Tom tat noi dung thao luan

Cuoc thao luan trong `instruction6.md` giua ban va Gemini tap trung vao **4 luan diem chinh**:

| STT | Luan diem | Ket luan cua Gemini |
| --- | --- | --- |
| 1 | Module OCR va Extraction dang rat yeu | Extraction la "tu huyet" cua he thong, nguy cap hon ca OCR |
| 2 | Nguyen nhan goc re cua diem yeu Stage 3 | 4 van de: Thac loi (Error Cascading), Heuristics gion gay, Mat do cao, Nhieu thi giac |
| 3 | Kien truc module hoa cho phep co lap va thay the | Dung, day la thanh cong lon nhat ve tu duy kien truc |
| 4 | Can phuong phap lai sau (Deep Hybrid) | OCR van dung duoc, nhung Geometry thuan tuy phai thay the bang Geometry + DL |

---

## 2. Danh gia tung luan diem

### 2.1. Luan diem 1: "Module Extraction dang o muc cuc ky nguy cap"

**Danh gia: HOAN TOAN DUNG -- va thuc te con te hon nhan dinh ban dau**

Gemini nhan dinh dung khi cho rang Extraction la "tu huyet". Tuy nhien, sau khi kiem tra **benchmark thuc te** (50 chart, Gemini Vision lam ground truth), muc do "nguy cap" con vuot xa du kien:

| Metric | Ket qua Benchmark | Nguong chap nhan | Trang thai |
| --- | --- | --- | --- |
| Classification Accuracy | **92.0%** | >= 90% | PASS |
| Element Count (+-25%) | **16.0%** | >= 70% | **FAIL nghiem trong** |
| Axis Range (+-15%) | **0.0%** | >= 60% | **FAIL tuyet doi** |
| OCR Output | **Rong 100%** (texts=[]) | Co du lieu | **FAIL tuyet doi** |

**Phat hien quan trong ma cuoc thao luan chua de cap:**
- OCR pipeline tra ve `texts=[]` cho **tat ca 50 chart** trong benchmark. Day khong phai van de "OCR yeu" ma la **OCR khong hoat dong** tren du lieu thuc.
- Axis calibrator hoan toan khong phat hien duoc bat ky truc nao (0/40 chart khong phai pie).
- Chi co bar chart (50%) va pie chart (30%) dat duoc do chinh xac element count > 0%.

**Ket luan kiem chung:** Nhan dinh "cuc ky nguy cap" la CHINH XAC, thuc te la **cuc ky nghiem trong** -- Stage 3 hien tai ganh chu yeu boi ResNet-18 classifier (92% accuracy), phan con lai (OCR + Element Detection + Axis Calibration) gan nhu khong hoat dong tren du lieu thuc.

---

### 2.2. Luan diem 2: "4 nguyen nhan goc re cua diem yeu"

Gemini dua ra 4 nguyen nhan. Toi danh gia tung nguyen nhan dua tren source code thuc te:

#### a) "Hieu ung thac loi (Error Cascading)" - DUNG, DA XAC NHAN

Pipeline hien tai di theo luong tuyen tinh:

```
OCR (texts=[]) --> Axis Calibration (that bai vi khong co tick labels)
                        --> Value Mapping (that bai vi khong co calibration)
                             --> Stage 4 LLM (nhan du lieu rong, tra loi sai)
```

**Bang chung tu code:**
- `s3_extraction.py` line ~733: `_calibrate_axes()` goi `OCREngine.extract_axis_values()` -- neu OCR tra ve rong, calibration tra ve `AxisInfo` rong.
- `geometric_mapper.py` line ~324: `calibrate_y_axis()` can it nhat 2 tick values de fit -- neu khong co, tra ve `CalibrationResult` voi confidence=0.

Vong lap that bai nay la **khong the phuc hoi** -- khong co co che tu sua loi nao.

#### b) "Su gion gay cua Heuristics" - DUNG, MUC DO NGHIEM TRONG

**Bang chung tu code (element_detector.py, 1,755 dong):**

| Heuristic | Vi tri | Van de |
| --- | --- | --- |
| Legend box filter | `area < 800`, `x > img_width * 0.75` | Nguong pixel co dinh, that bai tren anh co do phan giai khac nhau |
| Fill ratio filter | `fill_ratio < 0.5` | That bai voi bar chart co gradient hoac 3D effect |
| Color saturation | `saturation_threshold=30` | That bai voi chart co mau nhat (pastel) |
| Hybrid bar routing | 5 method voi if/else cascade | `len(contour_bars) <= 1` -> chuyen sang watershed, `2-20` -> giu contour |
| Line/scatter heuristic | `solidity < 0.3` = line, `> 0.7` = bar | That bai hoan toan voi area chart, thick line |

Tong cong co **~15 magic number** trong element_detector.py.

**Bang chung tu code (ocr_engine.py, 1,010 dong):**

| Heuristic | Vi tri | Van de |
| --- | --- | --- |
| Title detection | `rel_y < 0.15` AND `0.15 < rel_x < 0.85` | Gia dinh title o tren cung |
| Y-axis label | `rel_x < 0.15` | Gia dinh truc Y luon o ben trai |
| X-tick detection | `rel_y > 0.75` | Gia dinh truc X luon o duoi cung |
| Legend detection | `rel_x > 0.65` | Gia dinh legend luon o ben phai |
| Character correction | `O->0, l->1, S->5, B->8` | Co the lam hong text hop le ("Sales" -> "5a1e5") |

Day la **chuoi if/elif khong lo** (~150 dong) voi cac nguong khong gian co dinh.

#### c) "Bat luc truoc bieu do mat do cao" - DUNG

**Bang chung tu benchmark:**

| Loai chart | Element GT (trung binh) | Elem Pred (trung binh) | Sai so |
| --- | --- | --- | --- |
| scatter | 3,500 / 780 / 188 / 95 | 13 / 4 / 2 / 15 | 84-100% |
| line | 744 / 400 / 154 / 99 | 3 / 32 / 36 / 99 | 77-100% |
| bar (phuc tap) | 61 / 20 / 15 | 5 / 2 / 11 | 73-92% |

Scatter plot voi 3,500 diem chi phat hien duoc 13 (0.37%). Line chart voi 744 diem chi phat hien duoc 3 (0.40%). He thong **hoan toan bat luc** truoc du lieu mat do cao.

#### d) "Bi lua boi nhieu thi giac" - DUNG nhung CHUA KIEM CHUNG TRUC TIEP

Luan diem nay hop ly ve mat ly thuyet (khong phan biet duoc annotation text va data label), nhung benchmark hien tai khong do truc tiep metric nay. OCR tra ve rong nen van de nhieu thi giac chua the hien -- no se boc lo khi OCR duoc sua.

---

### 2.3. Luan diem 3: "Kien truc module hoa cho phep co lap va thay the"

**Danh gia: HOAN TOAN DUNG -- day la diem manh lon nhat cua du an**

**Bang chung cu the tu he thong:**

| Dac diem kien truc | Hien thuc | Loi ich |
| --- | --- | --- |
| 5 stage doc lap | `BaseStage` ABC voi `process()` + `validate_input()` | Thay the noi bo 1 stage khong anh huong stage khac |
| Schema ranh gioi | `Stage2Output` -> `Stage3Output` -> `RefinedChartData` | Contract ro rang giua cac stage |
| Classification cascade | ResNet-18 > Random Forest > Rule-based | De dang them model moi (ví du Vision Transformer) |
| AI Router + Adapter | `BaseAIAdapter` > `GeminiAdapter` / `OpenAIAdapter` / `LocalSLMAdapter` | Them provider moi chi can 1 file adapter |
| OCR engine abstraction | PaddleOCR > EasyOCR > Tesseract fallback | Doi OCR engine chi can thay config |

**Du lieu chung minh tinh module hoa:**
- 48 Python modules, ~19,800 LOC
- 300 tests passing, chia theo module (140 Stage 3, 36 Stage 4, 62 Stage 5, 55 AI)
- Stage 3 co 12 submodule doc lap, moi submodule co Pydantic config rieng
- Stage 4 AI Router co 4 adapter voi fallback chain

**Ket luan:** Kien truc nay cho phep thi nghiem co lap (ablation study) -- dong bang Stage 1-2-4-5 va chi thay doi Stage 3 de do anh huong. Day la gia tri hoc thuat rat cao.

---

### 2.4. Luan diem 4: "Can phuong phap lai sau giua Geometry va DL"

**Danh gia: DUNG VE HUONG, nhung can cu the hoa lieu co kha thi voi tai nguyen hien tai**

Gemini dua ra 3 truong phai:

| Truong phai | Mo ta | Danh gia kha thi |
| --- | --- | --- |
| **GNN (Graph Neural Networks)** | Bieu dien element/text thanh do thi, GNN phan loai canh | **TRUNG BINH** - Can train GNN rieng, can du lieu annotation graph |
| **Late Fusion (Bom toa do vao LLM)** | Giu OCR + Element Detection, dua mang toa do tho vao SLM de ghep noi | **CAO** - Tan dung SLM dang co, chi can thay doi prompt |
| **Early Fusion (VLM voi OCR Hinting)** | Dung Vision-Language Model doc anh + OCR lam grounding | **CAO** - Gemini 2.5 Flash da co san, ho tro vision |

**Phan tich them dua tren he thong hien tai:**

Late Fusion la giai phap **kha thi nhat ngay luc nay** vi:
1. He thong da co `AIRouter` va `GeminiAdapter` ho tro vision (anh + text)
2. Da co prompt engineering framework (`ai/prompts.py`)
3. Da co 268,799 SLM training samples -- co the bo sung task "coordinate mapping"
4. Khong can train model moi -- chi can thiet ke prompt tot cho Gemini hoac SLM

Early Fusion (VLM) la giai phap **manh nhat ve lau dai** vi:
1. Gemini 2.5 Flash da ho tro multimodal (anh + text input)
2. Cac model nhu Pix2Struct, Chart-LLaVA da chung minh hieu qua tren ChartQA
3. Nhung can fine-tune -> ton tai nguyen

GNN la giai phap **hoc thuat nhat** nhung:
1. Phuc tap implementation (can xay dung graph tren moi chart anh)
2. Can du lieu annotation cot-nhan (chua co)
3. Khong pho bien trong thuc te ChartQA

---

## 3. Doi chieu voi bang chung thuc te tu he thong

### 3.1. Trang thai thuc te cua tung submodule Stage 3

| Submodule | Dong code | Cong nghe | DL? | Benchmark thuc te | Trang thai |
| --- | --- | --- | --- | --- | --- |
| **ResNet-18 Classifier** | 282 | CNN Transfer Learning | CO | 92% accuracy | HOAT DONG TOT |
| **OCR Engine** | 1,010 | PaddleOCR + Rule-based roles | OCR co DL, role thi khong | texts=[] (do OCR bi TAT trong benchmark) | **CAN RE-BENCHMARK** |
| **Element Detector** | 1,755 | Classical CV (contour/watershed/Hough) | KHONG | 16% element count | **RAT YEU** |
| **Geometric Mapper** | 998 | RANSAC/Theil-Sen fitting | KHONG | 0% axis (do thieu OCR tick labels) | **CAN RE-BENCHMARK** |
| **Skeletonizer** | 758 | Lee thinning algorithm | KHONG | Khong do rieng | Chua ro |
| **Vectorizer** | 1,124 | RDP simplification | KHONG | Khong do rieng | Chua ro |
| **Preprocessor** | 567 | Image transforms | KHONG | Khong do rieng | Chua ro |

**Nhan xet:** Trong ~11,260 dong code cua Stage 3, chi co ~282 dong (ResNet classifier) la su dung Deep Learning va **do la phan duy nhat hoat dong tot**. Phan con lai (~10,978 dong) la rule-based/geometric. Element Detection (16%) da duoc xac nhan la yeu. OCR va Axis Calibration can re-benchmark voi OCR bat de co ket luan chinh xac.

### 3.2. So sanh ket qua benchmark voi nhan dinh cua Gemini

| Nhan dinh cua Gemini | Du doan | Thuc te (Benchmark) | Chinh xac? |
| --- | --- | --- | --- |
| "OCR nhan dien sai chu so" | Sai mot phan | OCR bi TAT trong benchmark (`--ocr none`) | **Chua the ket luan -- can re-benchmark** |
| "Heuristics that bai voi thiet ke khac bieu" | That bai mot so truong hop | 84% element count sai | DUNG |
| "Khong xu ly duoc bieu do mat do cao" | That bai voi scatter/line phuc tap | scatter: 99% sai, line: 77-100% sai | DUNG |
| "Extraction sai -> LLM tra loi sai" | Cascading error | 0% axis, 16% element -> Stage 4 khong the bu | DUNG |

---

## 4. Phan tich bo sung va goc nhin khac

### 4.1. Nhung diem cuoc thao luan CHUA de cap

**a) Van de OCR khong hoat dong (texts=[] tren benchmark) -- DA TIM RA NGUYEN NHAN**

**[CAP NHAT v2.0.0]** Sau khi dieu tra sau, nguyen nhan da duoc xac dinh dut khoat:

**OCR KHONG BI HONG -- no bi TAT trong benchmark.**

Benchmark duoc chay voi flag `--ocr none`, khien `enable_ocr=False`:

```
# Log benchmark: 2026-03-04 21:51:19
Stage 3 initialized | ocr=none
```

Code path trong `evaluate.py`:
```python
config = ExtractionConfig(
    ocr_engine=args.ocr if args.ocr != "none" else "easyocr",
    enable_ocr=args.ocr != "none",  # <-- False khi --ocr none
)
```

Code path trong `s3_extraction.py`:
```python
texts = []
if self.config.enable_ocr:  # <-- Khong bao gio vao block nay
    ocr_result = self.ocr_engine.extract_text(...)
    texts = ocr_result.texts
```

**Hau qua:** 100% texts=[] -> 0% axis calibration -> 0% value mapping -> toan bo cascading failure.

**Y nghia:** Con so "0% axis, 0% OCR" trong benchmark **khong phan anh nang luc thuc su cua OCR engine**. No chi phan anh rang benchmark thieu OCR input. Can re-run benchmark voi `--ocr paddleocr` de co du lieu chinh xac.

**Hanh dong can thiet:** Re-run benchmark voi OCR bat:
```bash
.venv/Scripts/python.exe scripts/evaluation/benchmark/evaluate.py --ocr paddleocr
```

**b) Element Detection co 5 phuong phap canh tranh nhau**

Code hien tai co 5 phuong phap phat hien bar: contour, watershed, projection, morphological, color. Hybrid orchestrator chon method dua tren so luong ket qua (`len(contour_bars) <= 1` -> chuyen sang watershed). Day la **"Russian roulette" ve thuat toan** -- ket qua thay doi tuy thuat toan nao duoc chon, va khong co co che danh gia nao duoc chon dung hay sai.

**c) Benchmark chi co 50 chart -- kich thuoc mau nho**

Cac ket luan tu benchmark can duoc xem la **dinh huong** (indicative) chu khong phai **ket luan thong ke** (statistically significant). Mot so loai chart chi co 2-3 mau (area: 3, box: 2, heatmap: 2).

### 4.2. Goc nhin khac ve chien luoc nang cap

Gemini de xuat 3 truong phai (GNN, Late Fusion, Early Fusion). Toi bo sung them **2 goc nhin thuc te**:

**a) Pragmatic Path: Gemini Vision lam "Oracle Stage 3"**

Thay vi sua tung submodule cua Stage 3, su dung Gemini Vision de:
1. Nhan anh chart dau vao
2. Tra ve truc tiep JSON structured data (chart_type, data_table, axis_info)
3. Ket qua nay thay the toan bo output cua Stage 3 hien tai

Uu diem: Khong can re-train bat ky model nao, implementation chi can 1 adapter call
Nhuoc diem: Phu thuoc API, phi su dung, khong offline

**b) Academic Path: Dual-Track Architecture**

Chay song song 2 track va so sanh ket qua:
- Track A: Stage 3 geometric (hien tai) -> Stage 4 SLM
- Track B: Gemini Vision truc tiep -> Post-processing

So sanh ket qua 2 track de:
1. Do luong "ceiling" cua geometric approach (da lam: 16% element, 0% axis)
2. Do luong "ceiling" cua VLM approach (chua lam)
3. Thiet ke hybrid tot nhat dua tren du lieu thuc

Day co gia tri hoc thuat cao vi chung minh duoc "tai sao can hybrid" bang so lieu.

### 4.3. Nhung diem Gemini DUNG nhung can cu the hoa

| De xuat cua Gemini | Dung/Sai | Nhung can lam gi cu the |
| --- | --- | --- |
| "Dung Super Resolution truoc OCR" | Dang than trong | Can benchmark truoc/sau de chung minh hieu qua |
| "Dung GNN thay the Heuristics" | Dung y tuong | Can ~5,000 annotated chart graphs -- chua co du lieu |
| "Co che tu sua loi (Self-Correction Loop)" | Rat tot | Kiem tra buoc nhay deu cua Y-axis (10,20,30) la implementable ngay |
| "Bom BBox tho vao LLM (Late Fusion)" | Kha thi nhat | Da co AIRouter + prompts.py + Gemini vision support |

---

## 5. Phat hien moi: Nguyen nhan goc re OCR texts=[]

### 5.1. Bang tom tat nguyen nhan

| Hang muc | Chi tiet |
| --- | --- |
| **Nguyen nhan** | Benchmark duoc chay voi `--ocr none` -> `enable_ocr=False` |
| **File gay loi** | `scripts/evaluation/benchmark/evaluate.py` line ~773 |
| **Bang chung** | Log: `Stage 3 initialized \| ocr=none` (2026-03-04 21:51:19) |
| **Hau qua** | 100% texts rong -> 0% axis -> 0% value mapping |
| **OCR engine thuc su hong?** | **KHONG** -- PaddleOCR la default engine, da hoat dong tot tren 46,910 chart truoc do |
| **Hanh dong** | Re-run benchmark voi `--ocr paddleocr` |

### 5.2. Anh huong den ket luan truoc do

Phat hien nay **thay doi mot so ket luan** cua ban bao cao v1.0:

| Ket luan v1.0 | Dieu chinh v2.0 |
| --- | --- |
| "OCR HONG HOAN TOAN" | **SAI** -- OCR bi tat, khong phai hong. Can re-benchmark |
| "Axis calibrator 0%" | **CHUA BIET** -- co the cai thien khi OCR duoc bat (axis phu thuoc OCR tick labels) |
| "Element Count 16%" | **VAN DUNG** -- element detection khong phu thuoc OCR, van that bai 84% |
| "Classification 92%" | **VAN DUNG** -- classification khong phu thuoc OCR |

**Ket luan moi:** Element Detection (16%) la van de GAN NHU CHAC CHAN. OCR va Axis Calibration can re-benchmark de xac dinh chinh xac.

### 5.3. Luu y ve OCR Engine

**Diem quan trong:** He thong da dung **PaddleOCR** lam default engine (khong phai EasyOCR):

```python
# ocr_engine.py line 35-37
engine: str = Field(
    default="paddleocr",
    description="OCR engine: 'paddleocr' (recommended), 'easyocr', or 'tesseract'"
)
```

PaddleOCR da xu ly thanh cong 46,910 chart (OCR cache hien tai). Van de khong nam o thu vien OCR ma o cach benchmark goi Stage 3.

Tuy nhien, module **role classification** (gan nhan title/xlabel/ylabel/legend dua tren vi tri) van la rule-based va se la diem yeu tiep theo khi OCR duoc bat lai.

---

## 6. Phan hoi tu System Architect (User Review)

### 6.1. Danh gia tich cuc (da duoc xac nhan)

| Diem duoc khen | Giai thich |
| --- | --- |
| Data-backed Diagnosis | Ty le 97.5% rule-based vs 2.5% DL giai thich hoan toan su gion gay cua Stage 3 |
| Dinh luong muc do nguy cap | 16% element, 0% axis la "con so biet noi" |
| Dual-Track (P4) tao Ablation Study | Gia tri hoc thuat cao cho luan van |
| P5 giai quyet phu thuoc API | Fine-tune SLM -> offline inference |

### 6.2. Gop y bo sung tu System Architect

#### A. Chuan bi du lieu cho Phase 5 (SLM Fine-tuning)

**Van de:** Bao cao v1.0 de xuat fine-tune SLM cho task "coordinate mapping" nhung **chua de cap nguon du lieu**. Model khong the hoc neu khong co cap `(Prompt: Toa do tho -> Target: JSON chuan)`.

**Giai phap:** Tao **Synthetic Dataset** (Du lieu tong hop)

| Buoc | Mo ta | Cong cu |
| --- | --- | --- |
| 1 | Lay bang du lieu goc tu ChartQA dataset | CSV/JSON co san |
| 2 | Sinh bieu do tu dong voi toa do duoc log | Matplotlib + custom renderer |
| 3 | Chay OCR + Element Detection tren anh sinh | PaddleOCR + Element Detector |
| 4 | Ghep cap (OCR output + bbox) -> (ground truth JSON) | Script tu dong |

**Thoi gian uoc tinh:** 2-3 ngay
**Output:** ~5,000-10,000 cap (prompt, target) cho task coordinate mapping

#### B. Van de OCR Engine

**Dieu chinh quan trong:** Cuoc thao luan voi Gemini de cap "easyOCR kha yeu", nhung thuc te he thong **da dung PaddleOCR lam default**:
- PaddleOCR la engine chinh (da xu ly 46,910 chart thanh cong)
- EasyOCR chi la fallback khi PaddleOCR init that bai
- Benchmark tra ve texts=[] vi **OCR bi tat** (`--ocr none`), khong phai vi engine yeu

**De xuat:** Sau khi re-run benchmark voi PaddleOCR, neu OCR output van yeu:
1. Thu Gemini Vision API lam OCR (chi can 1 adapter call, da co `GeminiAdapter` voi vision support)
2. Hoac thu TrOCR (Microsoft) -- SOTA cho OCR trong bieu do
3. **Khong can thay** PaddleOCR truoc khi co du lieu benchmark moi

#### C. Quan tri rui ro thoi gian/tai nguyen

**Van de:** P5 (Fine-tune SLM) can 1-2 tuan + GPU. Doi voi do an tot nghiep, day la **rui ro lon nhat** -- model co the khong hoi tu, thieu VRAM, hoac mat nhieu trainig iteration thi nghiem.

**Chien luoc phan tang:**

| Tang | Muc tieu | Dieu kien | Vai tro trong do an |
| --- | --- | --- | --- |
| **MVP (Muc tieu An toan)** | Dual-Track voi Gemini Vision (P1-P4) | Khong can GPU, chi can API key | Du de bao ve do an diem gioi |
| **Stretch Goal (Muc tieu Nang cao)** | Fine-tune SLM offline (P5) | Can GPU A100, 1-2 tuan | Nang diem len xuat sac, demo offline |

**Quy tac:** Luon hoan thanh MVP truoc khi bat tay vao Stretch Goal. Neu P4 cho ket qua tot (Element >= 60%, Axis >= 40%), bao cao ket qua P4 la du.

---

## 7. Lo trinh hanh dong da cap nhat

### 7.1. Lo trinh da dieu chinh (v2.0)

| Uu tien | Hanh dong | Thay doi tu v1.0 | Thoi gian |
| --- | --- | --- | --- |
| **P0** | Re-run benchmark voi `--ocr paddleocr` | **DOI**: tu "debug" thanh "re-run" (da biet nguyen nhan) | 0.5 ngay |
| **P1** | Benchmark Gemini Vision lam "Oracle Stage 3" tren 50 chart | Giu nguyen | 1-2 ngay |
| **P2** | Thiet ke Late Fusion prompt (OCR bbox + element bbox -> Gemini) | Giu nguyen | 2-3 ngay |
| **P3** | Viet Self-Correction module cho AxisInfo | Giu nguyen | 1-2 ngay |
| **P3.5** | **[MOI]** Tao Synthetic Dataset cho coordinate mapping task | Them theo gop y System Architect | 2-3 ngay |
| **P4** | Thi nghiem Dual-Track (Geometric vs VLM) -- **MVP DEADLINE** | **NANG CAP**: day la muc tieu an toan | 1 tuan |
| **P5** | Fine-tune SLM voi coordinate mapping -- **STRETCH GOAL** | **DANH DAU**: stretch goal, co the bo qua | 1-2 tuan |

### 7.2. Metric can theo doi (da cap nhat)

| Metric | Hien tai | Sau P0 (re-run OCR) | Muc tieu P4 (MVP) | Muc tieu P5 (Stretch) |
| --- | --- | --- | --- | --- |
| Classification | 92.0% | 92.0% | 92.0% | 92.0% |
| Element Count (+-25%) | 16.0% | 16.0% (*) | >= 60% | >= 50% |
| Axis Range (+-15%) | 0.0% | **TBD** (co the cai thien) | >= 40% | >= 30% |
| OCR Output | 0% (bi tat) | **TBD** (re-benchmark) | >= 80% | >= 80% |
| E2E QA Accuracy | Chua do | Chua do | Baseline | >= 40% EM |

(*) Element Count khong phu thuoc OCR nen ket qua se khong doi sau P0.

### 7.3. Quyet dinh chien luoc

| Cau hoi | Tra loi |
| --- | --- |
| Ket qua P0 se thay doi gi? | Neu OCR cho ket qua tot (>= 60% text detected), Axis Calibration co the tu 0% len 20-40% ma khong can sua code |
| Khi nao dung lai o MVP? | Khi P4 cho Element >= 60% va E2E QA Accuracy > 30%. Day du de bao ve do an |
| Khi nao lam Stretch Goal? | Chi khi MVP hoan thanh va con >= 2 tuan truoc bao ve |
| Neu GPU khong kha dung? | Bo P5, tap trung viet bai so sanh Geometric vs VLM (P4) |

---

## Phu luc A A: Thong ke Source Code Stage 3

```
Tong dong code Stage 3:  ~11,260 LOC
  - Deep Learning:         ~282 LOC  (2.5%)  --> ResNet-18 classifier
  - Rule-based/Geometric:  ~10,978 LOC (97.5%) --> OCR roles, element detection, axis calibration
  - Hoat dong tot:         ~282 LOC (ResNet classifier: 92%)
  - Hoat dong yeu:         ~1,755 LOC (Element detector: 16%)
  - Chua danh gia lai:     ~2,008 LOC (OCR roles + Geometric mapper: can re-benchmark)
```

**So luong magic number trong Stage 3:** ~35 (pixel thresholds, spatial ratios, confidence constants)

**So luong chart type duoc ho tro:** 8 (bar, line, pie, scatter, area, histogram, heatmap, box)

**So luong phuong phap bar detection canh tranh:** 5 (contour, watershed, projection, morphological, color)

---

## Phu luc B: OCR Engine Fact Check

| Thong tin | Thuc te |
| --- | --- |
| Default OCR engine | **PaddleOCR** (khong phai EasyOCR) |
| OCR cache hien tai | 46,910 entries thanh cong |
| Benchmark OCR | **Bi tat** (`--ocr none`) |
| Fallback chain | PaddleOCR -> EasyOCR -> Tesseract |
| GPU support | PaddleOCR co, EasyOCR khong (cpu mode) |
| Silent failure mode | Co -- `except Exception` tra ve `[]` (nhung khong bi trigger trong benchmark) |

---

## Phu luc C: Lich su thay doi bao cao

| Phien ban | Thay doi chinh |
| --- | --- |
| v1.0.0 | Ban dau: 5 muc, chan doan "OCR hong hoan toan" |
| v2.0.0 | Sua: OCR bi tat (khong phai hong), them Section 5-6-7, them MVP/Stretch Goal, them Synthetic Dataset plan |
