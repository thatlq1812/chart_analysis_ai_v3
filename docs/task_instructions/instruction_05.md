Dựa trên việc xem xét kỹ lưỡng mã nguồn trong thư mục `tools/data_factory` của repo `chart_analysis_ai_v3` bạn vừa tải lên, tôi xin đưa ra bản đánh giá chi tiết dưới góc độ kiến trúc sư hệ thống.

**Đánh giá tổng quan: 9/10 (Xuất sắc)**
Bạn (và AI Agent) đã thực hiện một bước nhảy vọt so với các phiên bản cũ. Code không chỉ "chạy được" mà còn được thiết kế để "mở rộng" (scalable). Đây chính xác là những gì cần thiết cho một hệ thống thu thập dữ liệu cấp độ nghiên cứu (Research-grade Data Pipeline).

Dưới đây là chi tiết phân tích:

### 1. Những điểm sáng giá (The Highlights)

* **Kiến trúc "Plug-and-Play" cho Nguồn dữ liệu:**
* Việc bạn tách `services/` thành các file riêng biệt (`hunter.py`, `hf_hunter.py`, `pmc_hunter.py`, `acl_hunter.py`) là thiết kế chuẩn **Strategy Pattern**.
* Nếu sau này muốn thêm nguồn *IEEE* hay *Springer*, bạn chỉ cần thêm 1 file mới mà không làm vỡ logic của `main.py`.
* **HuggingFace Integration (`hf_hunter.py`):** Đây là điểm cộng lớn. Việc stream dữ liệu trực tiếp từ HF `ChartQA` thay vì phải tự đi cào giúp bạn có ngay dữ liệu gán nhãn chuẩn (Gold standard) để validate model.


* **Tuân thủ nghiêm ngặt Schema (Pydantic Power):**
* File `schemas.py` đóng vai trò trung tâm. Tất cả các Hunter dù lấy từ nguồn nào (HTML, API, Dataset) đều phải trả về object `ArxivPaper` (hoặc tương đương) và `ChartImage` chuẩn.
* Điều này đảm bảo tính nhất quán dữ liệu cho các Stage sau (Training/Detection).


* **Logic "Miner" thông minh (`miner.py`):**
* Sử dụng `pymupdf` (fitz) là lựa chọn tối ưu nhất hiện nay cho PDF.
* Tôi thấy có logic `_extract_caption` cố gắng tìm văn bản bắt đầu bằng "Figure" hoặc "Fig" nằm gần ảnh. Đây là logic đơn giản nhưng hiệu quả cho 80% các bài báo khoa học.


* **Tooling chuyên nghiệp:**
* Sử dụng `Typer` cho CLI giúp câu lệnh rất rõ ràng (`python -m tools.data_factory.main hunt ...`).
* Logging bằng `loguru` giúp trace lỗi tốt hơn `print`.



### 2. Các điểm cần lưu ý và cải thiện (Critical Review)

Mặc dù code rất tốt, nhưng để "chắc chắn" hơn khi chạy thực tế với số lượng lớn (Bulk Processing), bạn cần lưu ý:

#### A. Vấn đề Rate Limit & API Keys (Trong `pmc_hunter.py`)

* **Hiện trạng:** Code dùng `Bio.Entrez`.
* **Rủi ro:** NCBI (PubMed) rất gắt gao về rate limit. Nếu không set email và API Key, bạn sẽ bị chặn IP rất nhanh.
* **Giải pháp:** Đảm bảo file `.env` đã có biến `NCBI_EMAIL` và `NCBI_API_KEY`, và code phải load nó vào `Entrez.email` và `Entrez.api_key`.

#### B. Logic trích xuất Context trong PDF (`miner.py`)

* **Hiện trạng:** Code đang tìm Caption (nằm ngay dưới ảnh).
* **Nâng cao:** Để train SLM (Reasoning) tốt, bạn cần cả đoạn văn bản trong bài viết *nhắc đến* hình đó (ví dụ: "As shown in Figure 1, the accuracy increases...").
* **Gợi ý:** Logic hiện tại có vẻ chỉ lấy Caption. Bạn nên mở rộng hàm `_extract_caption` hoặc viết thêm `_extract_context` để quét toàn bộ text trang đó tìm từ khóa `Figure X` tương ứng.

#### C. Xử lý ảnh trùng lặp (Deduplication)

* **Vấn đề:** Dataset `ChartQA` trên HuggingFace thực chất cũng lấy từ Arxiv. Nếu bạn chạy cả `ArxivHunter` và `HFHunter`, khả năng cao bạn sẽ lưu 2 bản copy của cùng một biểu đồ.
* **Giải pháp:** Thêm một bước check MD5 hash của ảnh hoặc check `paper_id` trước khi lưu vào database/folder.

### 3. Đánh giá file `generator.py` (Synthetic Data)

* Tôi thấy bạn có file này. Việc tạo dữ liệu giả lập (dùng `matplotlib` để vẽ chart rồi tự sinh label) là một chiến lược tuyệt vời để Pre-train model (Curriculum Learning) trước khi cho học dữ liệu thật khó hơn. Rất khen ngợi tầm nhìn này.

### 4. Kết luận và Hành động

Nội dung này đạt chuẩn **Production-ready** cho giai đoạn Data Collection.

Chào bạn, **Kiến trúc sư trưởng**. Việc sở hữu 20,000 mẫu từ Hugging Face (ChartQA) và 800 file PDF gốc từ ArXiv là một khởi đầu **cực kỳ mạnh mẽ** cho Phase 1.

Hiện tại, trạng thái dữ liệu của bạn đang phân hóa:

1. **Hugging Face (20k files):** Là dữ liệu "bán thành phẩm" (Semi-processed). Nó đã là ảnh và có label, nhưng cần chuẩn hóa về Schema của dự án V3.
2. **ArXiv (800 PDFs):** Là dữ liệu "thô" (Raw ore). Nó chứa thông tin quý giá nhất (Context, Caption chuẩn học thuật) nhưng đang nằm chết trong file PDF.

Dưới đây là **Chiến lược tiếp theo** để biến kho dữ liệu này thành "Nhiên liệu hạt nhân" cho mô hình AI:

### BƯỚC 1: "Khai thác" (Mining) ArXiv PDFs

800 file PDF này có thể sinh ra khoảng **4,000 - 8,000 biểu đồ** chất lượng cao kèm ngữ cảnh. Chúng ta cần chạy `miner` để bóc tách chúng.

* **Logic:** `PDFMiner` sẽ quét từng trang, tìm ảnh, crop ảnh, và quan trọng nhất là **cắt đoạn text (caption + context)** xung quanh ảnh đó để lưu vào JSON.
* **Hành động:** Chạy lệnh sau trong terminal (nhớ active `.venv`):

```bash
# Lệnh này sẽ quét folder data/raw_pdfs và nhả kết quả vào data/academic_dataset
python -m tools.data_factory.main mine

```

### BƯỚC 2: "Hợp nhất & Sàng lọc" (Unify & Sanitize)

Sau khi Mine xong, bạn sẽ có dữ liệu từ 2 nguồn (HF và ArXiv) nằm rải rác. Chúng ta cần một quy trình **Quality Control (QC)** nghiêm ngặt.

Tôi đề xuất bạn yêu cầu AI Agent viết thêm script hoặc sử dụng `sanitizer` có sẵn để thực hiện:

1. **Deduplication (Khử trùng lặp):**
* Rất nhiều ảnh trong ChartQA (HF) thực chất lấy từ ArXiv. Nếu không lọc, mô hình sẽ bị "data leakage" (học thuộc lòng) khi train/test.
* *Giải pháp:* Tính mã băm (MD5 Hash) của ảnh. Nếu ảnh từ ArXiv trùng với ảnh từ HF -> Ưu tiên giữ ảnh ArXiv (vì nó có context đầy đủ hơn từ PDF gốc), bỏ ảnh HF.


2. **Filter Non-Charts (Lọc rác):**
* Trong PDF khoa học có rất nhiều sơ đồ khối (Block diagrams), ảnh chụp thiết bị, logo... không phải biểu đồ.
* *Giải pháp:* Dùng script `sanitizer.py` để lọc theo tỷ lệ khung hình (Aspect Ratio) và độ phức tạp màu sắc. (Hoặc xịn hơn là dùng model YOLO mini để phân loại sơ bộ).



### BƯỚC 3: Exploratory Data Analysis (EDA) - Kiểm tra kho báu

Trước khi train, bạn phải biết mình có gì trong tay. Hãy tạo một Notebook `notebooks/01_data_exploration.ipynb` để trả lời các câu hỏi:

* Bao nhiêu % là Bar chart, Line chart, Pie chart? (Dựa trên metadata có sẵn hoặc chạy classifier sơ bộ).
* Phân bố kích thước ảnh thế nào? (Để chọn size input cho YOLO/Transformer).
* Độ dài trung bình của Caption?

---

### Đề xuất hành động ngay lập tức

Để giữ mạch làm việc "Kỷ luật thép", tôi đề xuất bạn giao 2 nhiệm vụ này cho AI Agent ngay bây giờ:

**Nhiệm vụ 1: Chạy Mining & Verify**

> "Hãy chạy lệnh `mine` cho toàn bộ 800 file PDF ArXiv. Sau đó viết một script nhỏ `scripts/verify_dataset.py` đếm xem tổng cộng chúng ta có bao nhiêu ảnh hợp lệ trong `data/academic_dataset`, và in ra 5 mẫu JSON metadata để kiểm tra xem field `caption` có bị rỗng không."

**Nhiệm vụ 2: Chuẩn bị Dataset Split**

> "Dựa trên `data/academic_dataset` đã gộp (HF + ArXiv), hãy viết script chia dataset thành 3 tập: `train` (80%), `val` (10%), `test` (10%) và di chuyển chúng vào cấu trúc thư mục `data/training/` theo chuẩn YOLO (images/labels)."

Bạn muốn bắt đầu với **Nhiệm vụ 1** (Xử lý PDF) hay nhảy cóc sang chuẩn bị data training luôn? (Lời khuyên: Nên làm Nhiệm vụ 1 thật kỹ, vì *Garbage In, Garbage Out*).

kèm theo là file env được bóc ra từ chart_ai_analysis v1, bạn xem thử xem có những gì có thể dùng được nhé