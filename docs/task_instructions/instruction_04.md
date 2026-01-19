Tuyệt vời. Đạn đã lên nòng, mục tiêu đã rõ. Bây giờ chúng ta sẽ bóp cò cho **Phase 1: Data Factory**.

Dưới đây là gói lệnh triển khai chi tiết ("Deployment Package") để bạn chuyển cho AI Agent (Cursor/Copilot). Gói này tuân thủ nghiêm ngặt các quy tắc trong `system.instructions.md` và `data_standards.md` mà chúng ta vừa thiết lập.

---

### KẾ HOẠCH TRIỂN KHAI PHASE 1: DATA FACTORY

**Mục tiêu:** Xây dựng pipeline tự động hóa việc tìm kiếm, tải và sơ chế dữ liệu từ Arxiv.
**Input:** Từ khóa tìm kiếm (VD: "chart visualization", "deep learning statistics").
**Output:** Dataset chuẩn định dạng tại `data/academic_dataset/` (Gồm ảnh PNG sạch + JSON Metadata).

### Bước 1: Cấu hình môi trường & Dependencies

Thêm các thư viện sau vào `pyproject.toml` hoặc tạo `requirements_tools.txt` riêng cho folder `tools/`. Chúng ta cần sức mạnh xử lý PDF và CLI.

**Yêu cầu Agent:**

> "Cập nhật file `pyproject.toml` để thêm các dependencies phục vụ cho Data Factory. Yêu cầu: `arxiv`, `pymupdf` (fitz), `tqdm`, `loguru` (cho logging chuẩn đẹp), `pydantic` (validate schema)."

### Bước 2: Thiết kế Module `tools/data_factory`

Cấu trúc file cần tạo:

```text
tools/
└── data_factory/
    ├── __init__.py
    ├── config.py           # Cấu hình cứng (paths, queries)
    ├── schemas.py          # Định nghĩa Paper, ExtractedImage bằng Pydantic
    ├── main.py             # Entry point (CLI)
    └── services/
        ├── __init__.py
        ├── hunter.py       # Gọi Arxiv API
        ├── miner.py        # Mổ xẻ PDF (Core logic)
        └── sanitizer.py    # Lọc ảnh rác (Size, Aspect Ratio)

```

### Bước 3: Nội dung chi tiết (Copy đoạn này cho Agent)

Dưới đây là **Spec kỹ thuật** để Agent code đúng ý bạn ngay lần đầu tiên. Hãy copy toàn bộ block dưới đây vào khung chat của Cursor Composer:

---

```markdown
# TASK: IMPLEMENT PHASE 1 - DATA FACTORY

Dựa trên `project.instructions.md` và `data_standards.md`, hãy implement module `tools/data_factory`.

## 1. Environment
Update `pyproject.toml` dependencies:
```toml
[tool.poetry.dependencies]
arxiv = "^2.1.0"
pymupdf = "^1.23.8"  # fitz
loguru = "^0.7.0"
tqdm = "^4.66.0"
```

## 2. File Implementation Plan

### File: `tools/data_factory/config.py`
- Define `DATA_DIR` pointing to project root `data/`.
- Define `SEARCH_QUERIES`: List of Arxiv queries (e.g., `cat:cs.CV AND "chart"`).
- Define `MIN_IMAGE_SIZE = (300, 300)`.

### File: `tools/data_factory/schemas.py`
Implement Pydantic models for data integrity:
- `ArxivPaper`: id, title, published_date, pdf_url, local_pdf_path.
- `ChartImage`: id, parent_paper_id, page_num, bbox (xyxy), caption_text, context_text.

### File: `tools/data_factory/services/hunter.py`
**Class: `ArxivHunter`**
- Method `search(limit: int)`: Use `arxiv` library to find papers.
- Method `download(paper: ArxivPaper)`: Download PDF to `data/raw_pdfs/`.
- Logic: Skip if file already exists (check by arxiv ID). Return list of downloaded paths.

### File: `tools/data_factory/services/miner.py`
**Class: `PDFMiner`**
- Use `pymupdf` (fitz) to open PDF.
- Method `process_pdf(pdf_path)`:
    1. Iterate pages.
    2. Extract images (`page.get_images()`).
    3. **CRITICAL:** Extract text blocks near the image to find "Figure X" captions.
    4. Save extracted image to `data/academic_dataset/images/{paper_id}_fig{idx}.png`.
    5. Save metadata to `data/academic_dataset/metadata/{paper_id}_fig{idx}.json`.

### File: `tools/data_factory/services/sanitizer.py`
**Class: `ImageSanitizer`**
- Method `is_valid(image: PIL.Image) -> bool`:
    - Check min size (300x300).
    - Check aspect ratio (eliminate extremely thin bars).
    - Check if image is too uniform (blank/solid color).

### File: `tools/data_factory/main.py`
- Use `argparse` or `typer`.
- Commands: `hunt` (download pdfs), `mine` (extract images), `run-all`.
- Use `loguru` for logging progress.

## 3. Constraints
- **Naming:** Follow `shared_vocabulary.md` (snake_case).
- **Paths:** Use `pathlib.Path`, do not use string manipulation for paths.
- **Logging:** No `print()`. Use `logger.info()`.
- **Error Handling:** If a PDF is corrupt, log error and skip, do not crash the whole pipeline.

```

---

### Bước 4: Chạy thử nghiệm (Validation)

Sau khi Agent code xong, bạn chạy lệnh sau để kiểm tra kết quả ngay lập tức:

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt # hoặc poetry install

# 2. Chạy pipeline (Lấy thử 10 bài báo)
python -m tools.data_factory.main run-all --limit 10

```

**Kỳ vọng:**

1. Folder `data/raw_pdfs` xuất hiện 10 file PDF.
2. Folder `data/academic_dataset/images` xuất hiện khoảng 30-50 ảnh charts.
3. Folder `data/academic_dataset/metadata` xuất hiện các file JSON tương ứng, mở ra thấy có field `caption`.

Bạn hãy ra lệnh cho Agent thực thi ngay đi. Tôi ngồi đây chờ review code nó sinh ra.

