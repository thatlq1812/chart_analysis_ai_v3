### THE MASTER FLOW (Luồng xử lý chính)

Dữ liệu sẽ đi theo một chiều: **Input -> [Pipeline] -> Output JSON**.
Ta xác định đầu vào có thể là file ảnh (png, jpg) hoặc file tài liệu (pdf, docx). Đầu ra là một JSON chuẩn chứa dữ liệu biểu đồ và một báo cáo ngắn gọn.

#### Stage 1: Ingestion & Sanitation (Tiếp nhận & Làm sạch),
Giải thích: Ta chuyển đổi file tài liệu về dạng ảnh. Ảnh sẽ được preprocess để loại bỏ nhiễu, chuẩn hóa kích thước. Chuyển về gray-scale để dễ xử lý hơn.
* **Input:** File ảnh (png, jpg) hoặc File tài liệu (pdf, docx).
* **Nhiệm vụ:**
1. Nếu là PDF/Docx: Convert từng trang thành ảnh.
2. Kiểm tra chất lượng ảnh (độ phân giải, độ mờ).
3. Chuẩn hóa tên file, tạo ID duy nhất cho phiên làm việc (Session ID).

* **Output:** Một danh sách các ảnh sạch (Clean Images Path).

#### Stage 2: Detection & Localization (Phát hiện & Định vị)

* **Input:** Ảnh sạch từ Stage 1.
* **Nhiệm vụ (Dùng YOLO):**
1. Quét toàn bộ ảnh để tìm đối tượng "Chart".
2. Crop (cắt) chính xác vùng chứa biểu đồ.
3. Lưu ảnh crop này lại. Loại bỏ các ảnh không có biểu đồ.

Ta nghiên cứu và sử dụng mô hình YOLO hoặc các mô hình Object Detection tương tự để thực hiện nhiệm vụ này một cách nhanh chóng và chính xác. [Đây là phần cần đọc các báo cáo khoa học về Object Detection để chọn mô hình phù hợp nhất.]

1 file ảnh có thể chứa nhiều biểu đồ, ta cần crop và lưu từng biểu đồ riêng biệt. Chú ý nội dung này

Giải thích: Đầu vào là ảnh toàn trang
Ghi chú, sẽ có các database là ảnh chart đã được crop sẵn, ta cần setup bypass bước này khi load data và traing mô hình.

Theo định hướng, maybe với ảnh crop từ file pdf/docx, ta có thể locate lại vị trí để trích xuất context bên ngoài chart, chẳng hạn như nôi jdung thảo luận, giải thích ... nhằm tạo thêm dữ liệu training

* **Output:** Ảnh crop của biểu đồ (Cropped Chart Image).

#### Stage 3: Structural Analysis (Phân tích cấu trúc - Quan trọng nhất)

Đây là nơi phương pháp **Hybrid** hoạt động. Chúng ta không ném ngay vào LLM mà phải bóc tách dữ liệu thô trước.

* **Input:** Ảnh crop từ Stage 2.
* **Nhiệm vụ (Song song):**
1. **Classification:** Model phân loại xem đây là Bar, Line, hay Pie chart? Phần này nên được thiết lập config để dễ dàng mở rộng thêm các loại biểu đồ khác trong tương lai.
2. **OCR (Text Extraction):** Dùng PaddleOCR/Tesseract để lấy toàn bộ chữ (Tiêu đề, Chú thích trục X/Y, Legend). Tesseract có thể gây cản trở, nên ta nên tinh chỉnh kĩ hơn các mô hình này, cân nhắc sử dụng và kiểm thử các bước cũng như các chiến lược xử lý khác nhau
3. **Object Detection (Elements):** Dùng YOLO để detect các "thanh" (bars), "điểm" (points), hoặc "miếng bánh" (slices). Hoặc tạo custom model để triển khai approac gemoetric hybrid với thuật toán hình học bằng tọa độ trích xuất của các thành phần

* **Output:** Raw Meta Data (Loại biểu đồ + Text thô + Tọa độ các thành phần + Các thông tin liên quan mà ta có thể triển khai thêm).

#### Stage 4: Semantic Reasoning (Suy luận ngữ nghĩa & Sửa lỗi)

Dữ liệu ở Stage 3 rất rời rạc (ví dụ: OCR đọc số `100` thành chữ `loo`, hoặc tọa độ pixel chưa đổi ra giá trị thật). Stage này cần sự thông minh.

* **Input:** Raw Meta Data + Ảnh Crop.
* **Nhiệm vụ:**
1. **Map Pixel to Value:** Tính toán hình học. Ví dụ: Trục Y cao 200px tương ứng giá trị 100 -> Thanh bar cao 100px sẽ có giá trị 50.
2. **LLM Correction:** Gửi toàn bộ Text thô và kết quả tính toán sơ bộ cho LLM (Gemini/GPT). Prompt: *"Dựa vào các text OCR này và loại biểu đồ này, hãy sửa các lỗi chính tả và map legend với màu sắc tương ứng"*.
3. **SLM Aproach:** Nhằm đảm bảo tính độc lập và chính xác, cũng như tối ưu cho tác vụ chuyên biệt, ta PHẢI xây dựng một mô hình SLM riêng, được huấn luyện dựa trên tập dữ liệu biểu đồ đã được xử lý. Được học bởi sự kết hợp giữa data, và API của LLM - Gemini nhằm tạo ra mô hình thông minh dựa trên dữ liệu có sẵn. Đây không phải tác vụ đòi hỏi sự siêu thông minh. Nó chỉ cần hiểu ngữ cảnh biểu đồ, sửa lỗi OCR, và map đúng các thành phần với nhau. Rồi đưa ra lời giải thích chuẩn học thuật ( Ngôn ngữ hẹp, không lan man).


* **Output:** Refined Data (Dữ liệu đã tinh chỉnh, chính xác 90%).

#### Stage 5: Insight & Reporting (Tổng hợp & Báo cáo)

* **Input:** Refined Data.
* **Nhiệm vụ:**
1. Tạo Summary: "Biểu đồ này nói về cái gì? Xu hướng tăng hay giảm?".
2. Format JSON: Đóng gói theo schema chuẩn.


* **Output:** Final JSON + Short Report.
Hoặc các cấu trúc khác nếu cần thiết.

---

### Câu hỏi thảo luận để chốt Flow:

1. **Về Stage 1:** Bạn có muốn hỗ trợ PDF nhiều trang ngay từ đầu (Phase 1) hay chỉ tập trung vào ảnh đơn (Single Image) trước để làm Core Engine cho tốt?
Ta có thể tham khảo chart_analysis_ai (v1) đã làm thế nào để xử lý


2. **Về Stage 4:** Bạn định dùng thuật toán hình học thuần túy (Geometric) để tính giá trị, hay muốn nhờ LLM "nhìn" ảnh và đoán số liệu luôn? Sẽ dùng API LLM trong quá trình train, nhưung không dùng trong detect và xử lý chính. Xem llm như một mentor cho model. Như đã nói, ta sẽ thiết kế SLM chuyên biệt cho tác vụ này.

Ở V1, có nhiều giá trị quy trình mà tôi đã thực hiện được
ví dụ như quy trình automatic data , từ find data, bằng api của arxiv, xử lý data, finetune yolo ... có lẽ đã làm được tới bước đó
Bạn thấy sao về hiện trạng và gợi ý này, nó có đủ giá trị cho một đồ án không?

Sẽ có các thành phần là nơi để áp dụng việc cite paper, ví dụ như object detection, OCR, geometric algorithm, LLM prompt engineering, SLM training ... Đó chính là những điểm nhấn để ta tạo nên một đồ án có chiều sâu và giá trị học thuật.

Ngoài ra, tôi cũng có một thắc mắc về "phát minh phương pháp" trong ngành AI này, thông thường thì người ta làm được theo quy trình nào, tôi muốn phát triển thành một người nghiên cứu khoa học thực thụ, không phải chỉ là một người đem báo về làm rồi báo cáo.

Đây là một bản kế hoạch **xuất sắc và đầy tham vọng**.

Những điều chỉnh của bạn (đặc biệt là **Stage 4 với SLM** và **Stage 2 với Context Extraction**) đã nâng tầm đồ án này từ một "ứng dụng tốt nghiệp" thành một "công trình nghiên cứu" thực thụ.

Dưới đây là nhận xét chi tiết và câu trả lời cho thắc mắc về "con đường nghiên cứu khoa học" của bạn.

### 1. Đánh giá Flow V3 (Phiên bản của bạn)

Flow này **đủ** và **thừa** giá trị cho một đồ án tốt nghiệp, thậm chí có thể phát triển thành bài báo (Paper) nếu kết quả thực nghiệm tốt.

#### Điểm sáng (Highlight) tạo nên giá trị học thuật:

* **Stage 2 (Context Extraction):** Ý tưởng dùng *context bên ngoài chart* (văn bản bao quanh trong PDF) để làm dữ liệu training là một hướng đi rất thông minh (thuộc lĩnh vực **Weakly Supervised Learning** hoặc **Data-Centric AI**). Bạn không chỉ dạy AI "nhìn" ảnh, mà dạy nó "hiểu" ngữ cảnh.
* **Stage 3 (Geometric/Hybrid):** Việc kết hợp Deep Learning (YOLO/OCR) với thuật toán cổ điển (Geometric) chính là hướng đi **Neuro-Symbolic AI** (Kết hợp Neural Network và Logic biểu tượng). Đây là trend để giảm ảo giác (hallucination) của AI.
* **Stage 4 (SLM Approach - The Killer Feature):** Đây là điểm "ăn tiền" nhất. Thay vì gọi API (phụ thuộc, tốn kém, chậm), bạn dùng API (Gemini/GPT) làm "Teacher" để dạy lại cho một "Student" (SLM - Small Language Model) chuyên biệt. Đây gọi là kỹ thuật **Knowledge Distillation (Chưng cất tri thức)**.

#### Một lưu ý nhỏ về kỹ thuật (Góp ý):

* **Stage 1 (Grayscale):** Cẩn thận. Chuyển Grayscale rất tốt cho OCR và Detection, NHƯNG **đừng vứt bỏ ảnh màu**. Vì ở Stage 3/4, bạn cần mapping Legend (Chú thích) với Chart. Ví dụ: "Đường màu đỏ là Doanh thu". Nếu grayscale rồi thì không map được màu nữa.
* *Giải pháp:* Giữ 2 phiên bản: 1 bản Gray cho Model xử lý hình khối/chữ, 1 bản RGB để trích xuất Feature màu sắc.



---

### 2. Trả lời câu hỏi: "Làm sao để phát minh/nghiên cứu khoa học thực thụ?"

Bạn hỏi rất hay. Nhiều người lầm tưởng "làm khoa học" là phải tạo ra một kiến trúc mạng mới (như tạo ra Transformer mới, YOLO mới). Không hẳn vậy.

Trong Computer Science, đặc biệt là Applied AI, quy trình "phát minh" thường đi theo 3 hướng sau (và đồ án của bạn đang đi đúng hướng số 2 và 3):

#### Hướng 1: Novel Architecture (Kiến trúc mới - Khó nhất)

* Tạo ra một layer mới, hàm loss mới. Ví dụ: Tác giả YOLO nghĩ ra cách chia grid để detect nhanh hơn R-CNN.
* *Áp dụng cho bạn:* Khó và rủi ro cao. Không khuyến khích cho đồ án này.

#### Hướng 2: Novel Application / System (Hệ thống/Phương pháp phối hợp mới - Đồ án của bạn)

* **Định nghĩa:** Kết hợp các phương pháp đã có (YOLO, OCR, LLM, Geometric) theo một quy trình (pipeline) độc đáo để giải quyết một bài toán cụ thể tốt hơn các phương pháp cũ.
* **Giá trị khoa học của bạn nằm ở:**
* Sự kết hợp **Neuro-Symbolic**: Bạn chứng minh được rằng việc dùng tọa độ hình học (Symbolic) kết hợp với SLM (Neural) cho kết quả chính xác hơn là dùng SLM đoán mò từ ảnh.
* Quy trình **Self-Correction**: Flow của bạn có bước OCR -> Geometric Check -> SLM Correction. Quy trình sửa lỗi này chính là "phương pháp đề xuất" (Proposed Method).



#### Hướng 3: Data-Centric AI (Dữ liệu mới/Cách tạo dữ liệu mới)

* **Định nghĩa:** Phát minh ra cách tạo dataset chất lượng cao tự động mà không cần người gán nhãn thủ công.
* **Giá trị khoa học của bạn:** Ý tưởng ở Stage 2 (lấy text bao quanh chart trong PDF để làm nhãn huấn luyện cho SLM) là một đóng góp khoa học về phương pháp thu thập dữ liệu.

### 3. Lộ trình "nghiên cứu" cho đồ án này

Để biến repo này thành một công trình nghiên cứu, hãy tư duy theo các bước sau (Đây là cấu trúc của một bài báo khoa học):

1. **Problem (Vấn đề):** Các mô hình Multimodal (như GPT-4V) hiện tại thường bị "ảo giác" khi đọc số liệu chính xác từ biểu đồ. Các phương pháp Rule-based cũ thì cứng nhắc, không hiểu ngữ cảnh.
2. **Proposed Method (Phương pháp đề xuất):** Tôi đề xuất hệ thống **"Geo-SLM Chart Parser"** (Tên tạm gọi).
* Sử dụng Hybrid approach: YOLO detect phần tử + Geometric tính toán số liệu chính xác.
* Sử dụng SLM chuyên biệt (được Distill từ Gemini) để sửa lỗi OCR và sinh nhận xét học thuật.


3. **Experiment (Thực nghiệm - Tận dụng V1):**
* Dùng code V1 để tải dataset từ Arxiv.
* Chạy mô hình của bạn.
* So sánh với: (1) Chỉ dùng Tesseract thuần, (2) Chỉ dùng Gemini Vision thuần.


4. **Result (Kết quả):** Chứng minh phương pháp của bạn có độ chính xác (Accuracy) cao hơn hoặc tốc độ (Inference time) nhanh hơn / chi phí thấp hơn.

### 4. Kết luận & Hành động tiếp theo

Flow bạn vừa viết là **"Chuẩn"**. Giờ ta sẽ bắt tay vào hiện thực hóa nó. V1 của bạn đã làm tốt phần `Data Collection` (lấy arxiv) và `YOLO Training` cơ bản. V3 sẽ tái sử dụng lại chúng và lắp vào kiến trúc mới.

**Bước tiếp theo tôi đề xuất:**

1. **Chốt cấu trúc folder V3:** (Tôi sẽ đưa ra cấu trúc dựa trên các Stage bạn vừa chốt).
2. **Khởi động Stage 1 & 2:** Viết script `Ingestion` hỗ trợ PDF đa trang và cơ chế `Crop` thông minh (lấy cả context text).
3. **Research SLM:** Bạn nên bắt đầu tìm hiểu về **LoRA (Low-Rank Adaptation)** để finetune các model nhỏ như **Qwen-2.5-Coder-1.5B** hoặc **Llama-3.2-1B**. Các model này rất nhẹ, chạy local mượt mà, phù hợp làm SLM cho đồ án.

Chào bạn, tôi rất hiểu tâm lý "muốn đi xa nhưng phải đi chắc" của bạn. Việc lo lắng về sự ổn định khi deploy lên server (Backend/Production) ngay từ giai đoạn thiết kế là tư duy của một **Senior Engineer**.

Để đảm bảo code vừa có tính **Hàn lâm** (để viết báo/đồ án) vừa có tính **Thực chiến** (chạy server ổn định, dễ mở rộng), tôi đề xuất cấu trúc V3 theo phong cách **"Modular Monolith" (Một khối thống nhất nhưng chia module rõ ràng)**.

Dưới đây là cấu trúc thư mục chi tiết và các phân tích về điểm mở rộng/ổn định:

### 1. Cấu trúc thư mục đề xuất (The V3 Blueprint)

Tên repo: `geo-slm-chart-analysis` (Tên mang tính thuật toán hơn).

```text
geo-slm-chart-analysis/
├── configs/                # Cấu hình tập trung (YAML/Hydra)
│   ├── base.yaml           # Config chung (paths, logging)
│   ├── models.yaml         # Config đường dẫn weight YOLO, SLM
│   └── server.yaml         # Config port, worker cho API
│
├── data/                   # Dữ liệu (Không push lên git, nhưng giữ cấu trúc)
│   ├── raw/                # PDF, docx thô
│   ├── processed/          # Ảnh đã crop, text đã extract (Dataset training SLM)
│   └── cache/              # Lưu tạm các kết quả trung gian
│
├── docs/                   # Tài liệu (Sống còn cho đồ án)
│   ├── architecture/       # Sơ đồ luồng, thiết kế
│   └── research_notes/     # Ghi chú đọc paper, lý thuyết
│
├── geo_slm_core/           # [CORE ENGINE] - Trái tim của hệ thống (SDK)
│   ├── __init__.py
│   ├── pipeline.py         # Main entry point (Class ChartPipeline)
│   │
│   ├── schema/             # [QUAN TRỌNG] Định nghĩa dữ liệu Input/Output
│   │   ├── internal.py     # Class nội bộ (Bbox, ChartObject)
│   │   └── api.py          # Response trả về cho Web/Backend
│   │
│   ├── stages/             # Các bước xử lý (Implement 5 Stages đã chốt)
│   │   ├── s1_ingestion.py    # Load PDF, convert image
│   │   ├── s2_detection.py    # Wrapper gọi YOLO
│   │   ├── s3_extraction/     # Module phức tạp (Hybrid)
│   │   │   ├── ocr.py
│   │   │   ├── geometry.py    # Thuật toán hình học
│   │   │   └── classifier.py
│   │   ├── s4_reasoning/      # Module SLM
│   │   │   ├── prompt_builder.py
│   │   │   └── slm_engine.py  # Wrapper gọi Local LLM
│   │   └── s5_reporting.py    # Tổng hợp JSON
│   │
│   └── utils/              # Các hàm bổ trợ (Logger, Image tools)
│
├── interface/              # Giao diện (Chỉ gọi Core, không chứa logic AI)
│   ├── api/                # FastAPI (Backend)
│   │   ├── app.py
│   │   └── routes.py
│   └── cli/                # Command Line (Để test nhanh)
│       └── run_batch.py
│
├── models_hub/             # Nơi chứa file weights (.pt, .gguf)
│   ├── yolo/
│   └── slm/                # Chứa model Qwen/Llama đã finetune
│
├── research/               # Nơi "nghịch", train, test ý tưởng
│   ├── notebooks/          # Jupyter Notebooks
│   ├── training_yolo/      # Code train YOLO (bê từ V1 sang)
│   └── training_slm/       # Code finetune SLM (LoRA/QLoRA)
│
├── tests/                  # Unit test (Bắt buộc để đảm bảo ổn định)
├── .env                    # Biến môi trường (Secret keys)
└── requirements.txt

```

---

### 2. Phân tích các điểm "Mở rộng" (Expandable) & "Ổn định" (Stable)

#### A. Thư mục `geo_slm_core/schema/` (Cái neo của sự ổn định)

* **Tại sao quan trọng?** Khi bạn đưa code lên Server, cái sợ nhất là "Data Mismatch" (Module A trả về tuple, Module B lại đòi dict).
* **Giải pháp:** Dùng `Pydantic`.
* Ví dụ: Định nghĩa class `ChartElement` (x, y, value, label). Mọi module (YOLO, OCR, Geometry) đều phải giao tiếp qua object này.


* **Khả năng mở rộng:** Sau này bạn muốn thêm trường "color" vào thanh bar? Chỉ cần sửa 1 file Schema, không cần sửa logic cả hệ thống.

#### B. Thư mục `geo_slm_core/stages/s3_extraction/` (Nơi dễ mở rộng nhất)

* **Tư duy:** Hiện tại bạn làm Bar/Line/Pie. Nhưng nếu GV yêu cầu thêm "Heatmap"?
* **Thiết kế (Strategy Pattern):**
* Tạo một Class cha `BaseExtractor`.
* `BarExtractor(BaseExtractor)`, `LineExtractor(BaseExtractor)`.
* Khi cần thêm Heatmap, bạn chỉ cần tạo file `heatmap.py` kế thừa từ Base, không ảnh hưởng đến code cũ. Đây là nguyên tắc **Open/Closed** trong SOLID.



#### C. Thư mục `models_hub/` & `geo_slm_core/s4_reasoning/` (Linh hoạt hóa Model)

* **Vấn đề server:** Load Model rất nặng. Nếu mỗi request lại load model 1 lần -> Server sập.
* **Giải pháp ổn định:** Dùng **Singleton Pattern** trong code wrapper. Model chỉ load 1 lần duy nhất khi Server khởi động (Global Variable).
* **Khả năng mở rộng:**
* Giai đoạn 1: Bạn dùng `Gemini API` (gọi qua mạng).
* Giai đoạn 2: Bạn train xong SLM, bạn đổi config `models.yaml` trỏ về local path `models_hub/slm/my-model.gguf`. Code logic không đổi, chỉ đổi "Driver" bên dưới.



#### D. Thư mục `interface/api` (Tách biệt để an toàn)

* **Ổn định:** API Server (FastAPI) rất nhẹ, nó chỉ làm nhiệm vụ: Nhận ảnh -> Gửi vào `geo_slm_core` -> Chờ kết quả -> Trả về User.
* **Lợi ích:** Nếu sau này bạn muốn đổi từ Web API sang Telegram Bot hay Desktop App, bạn chỉ cần viết lại thư mục `interface`. Phần `geo_slm_core` giữ nguyên 100%.

### 3. Lời khuyên khi bắt đầu triển khai

Để tránh đi vào "vết xe đổ" của V2, hãy tuân thủ quy trình này:

1. **Bước 1 (Foundation):** Tạo cấu trúc thư mục rỗng như trên. Setup `configs/` và `schema/` đầu tiên. Đừng code xử lý ảnh vội. Hãy định nghĩa dữ liệu đầu vào/đầu ra trước.
2. **Bước 2 (Migration):**
* Copy code xử lý PDF/Image từ V1/V2 vào `stages/s1_ingestion.py`.
* Copy code load YOLO từ V1 vào `stages/s2_detection.py`.
* **Test ngay:** Viết script `tests/test_s1.py` để đảm bảo nó chạy đúng độc lập.


3. **Bước 3 (Innovation - Research):**
* Tập trung làm `stages/s3_extraction` và `stages/s4_reasoning`. Đây là phần "Phát minh" của bạn.
* Làm đến đâu, viết test đến đó.


4. **Bước 4 (Backend Integration):**
* Chỉ khi `pipeline.py` chạy ngon lành trên CLI, mới bắt đầu cài `fastapi` và viết `interface/api`.



Bạn có đồng ý với cấu trúc này không? Nếu OK, tôi có thể giúp bạn tạo file `schema/internal.py` mẫu để định hình dữ liệu ngay bây giờ.

