Sau khi đọc code thực tế, nhận định "yếu như lớp 1" của bạn là **hoàn toàn có cơ sở**. Nhưng quan trọng hơn, mình tìm thấy các vấn đề cụ thể:

---

## 🐛 Bug nghiêm trọng — Pie chart KHÔNG BAO GIỜ detect được

Nhìn vào `element_detector.py`, trong method `detect()` chính:

```python
slices = []
# ... code cho bars và markers ...
# Nhưng KHÔNG CÓ DÒNG NÀO GỌI pie slice detection!
```

Method `_detect_pie_slices_by_kmeans()` tồn tại, được viết khá kỹ, nhưng **không bao giờ được gọi** trong luồng chính. `slices` luôn trả về `[]`. Đây không phải yếu — đây là **dead code hoàn toàn**.

---

## ⚠️ Vấn đề hệ thống trong OCR Role Classification

```python
if rel_x < 0.15:  # "Left 15% = Y-axis label"
if rel_y > 0.85:  # "Bottom 15% = X-axis"
if rel_x > 0.65:  # "Right 35% = Legend"
```

Đây là một chuỗi `if/elif` với magic numbers hardcode. Khi legend nằm dưới cùng (bottom legend — rất phổ biến), hoặc chart có double Y-axis, toàn bộ role classification sẽ sai.

---

## ⚠️ `_detect_bars_hybrid` thiếu tham số

Trong `detect()`, gọi:
```python
bars = self._detect_bars_hybrid(binary_image, color_image, chart_id)
# chart_type không được truyền vào!
```
Nhưng `_detect_bars_hybrid` cần `chart_type` để kích hoạt K-Means cho stacked bar. Kết quả là K-Means path cho stacked bar **không bao giờ chạy**.

---

## Kết luận thực tế

Hệ thống không yếu vì thuật toán tệt — mà yếu vì **implementation chưa kết nối đầy đủ**. Trước khi nghiên cứu thuật toán mới, bạn nên fix 3 điểm trên trước — đặc biệt là bug pie slice. Đó là quick win ngay lập tức, và sau khi fix xong, bạn sẽ có baseline thực sự để đo cải tiến.

Bạn muốn mình giúp viết fix cụ thể cho từng điểm không?

===

Đây là đủ dữ liệu để phân tích sâu. Để mình tổng hợp toàn diện.Đủ rồi. Bây giờ mình có đủ bức tranh toàn cảnh từ code đến báo cáo để phân tích đối chiếu nghiêm túc.

---

## Phân tích chuyên sâu — Đối chiếu Code & Báo cáo

---

### 1. Mâu thuẫn nghiêm trọng nhất: "100% success rate" là con số gây hiểu lầm

Báo cáo `data_pipeline_report_v1` ghi:

> "32,364 / 32,364 — Error rate: 0%"

Nhưng nhìn vào **Feature Quality Metrics** cùng file đó:

| Loại | axis_cal_x | axis_cal_y | OCR conf | zero_text% |
|------|-----------|-----------|----------|------------|
| area | 0.000 | 0.000 | 0.387 | 59.5% |
| pie | 0.097 | 0.138 | 0.783 | 19.0% |
| bar | 0.219 | 0.466 | 0.671 | 29.6% |

"100% success" chỉ có nghĩa là **file JSON được tạo ra, không bị corrupt** — không có nghĩa là dữ liệu bên trong đúng. Area charts có axis calibration = 0.000 hoàn toàn, tức là 617 file area chart trong training set không có thông tin trục nào cả. Đây là sự khác biệt rất lớn giữa "chạy không lỗi" và "kết quả đúng".

---

### 2. Pie chart — Ba tầng thất bại chồng nhau

Từ code và báo cáo, pie chart đang thất bại ở **ba tầng độc lập**:

**Tầng 1 — Dead code** (đã phân tích trước): `_detect_pie_slices_by_kmeans()` không bao giờ được gọi, `slices` luôn `= []`.

**Tầng 2 — Test suite bảo vệ sai**: Trong `test_element_detector.py`, không có một test nào kiểm tra `len(result.slices) > 0`. Duy nhất một dòng `assert hasattr(result, 'slices')` — tức là test chỉ kiểm tra *thuộc tính tồn tại*, không kiểm tra *giá trị*. Đây là lý do 294 tests đều pass dù pie detection bị broken hoàn toàn.

**Tầng 3 — Data imbalance cực đoan**: Trong SLM training dataset v3, pie chỉ chiếm 2.8% (7,408 mẫu) trên tổng 268,799. Khi SLM được train trên dữ liệu mà pie luôn trả về `elements: []`, model sẽ học rằng pie chart không có phần tử — và đây sẽ là knowledge được baked vào model.

---

### 3. Area chart — Module chưa được implement thực sự

Axis calibration x=0.000, y=0.000 cho toàn bộ area charts không phải lỗi ngẫu nhiên. Nhìn vào `geometric_mapper.py`, logic tìm axis line dùng Hough Transform với điều kiện:

```python
# X-axis should be in bottom 70% of image
if y_pos > h * 0.3:
# Y-axis should be in left 50% of image  
if x_pos < w * 0.5:
```

Area chart thường có **filled regions chiếm toàn bộ plot area**, che khuất axis lines. Hough Transform sẽ detect các đường viền của vùng tô thay vì axis thực sự. Báo cáo ghi nhận vấn đề này nhưng action item được đánh là **LOW priority** — trong khi đây là loại chart có tỉ lệ zero-text 59.5%, tức là hầu hết training samples của area không có nội dung hữu ích.

---

### 4. SLM Training — Chuỗi nhân quả từ Stage 3 yếu sang model yếu

Đây là điểm quan trọng nhất khi đối chiếu toàn bộ:

SLM training session 1 thất bại (EM = 4%) và được giải thích bởi 4 bugs kỹ thuật. Nhưng ngay cả khi fix hết 4 bugs đó, có một vấn đề sâu hơn chưa được nhắc đến trong postmortem: **chất lượng training data phụ thuộc trực tiếp vào Stage 3 output**.

Cụ thể, dataset v2 có lỗi: `axis_info` không được embed vào prompt do bug truy cập sai key (`y_range.min` thay vì `y_min`). Dataset v3 đã sửa lỗi này — nhưng với axis calibration confidence = 0.097 cho pie và 0.000 cho area, ngay cả khi embed đúng thì dữ liệu trục vẫn sai. Model sẽ học được những con số sai từ Stage 3.

Nói cách khác: **fix SLM training bugs chỉ fix phần bề mặt**. Core hypothesis của thesis — Geo-SLM outperforms pure VLM — chưa thể kiểm chứng khi Stage 3 output cho một số chart types gần như không có giá trị.

---

### 5. Log-scale bug — Vấn đề tiềm ẩn trong scatter charts

Báo cáo tự ghi nhận:

> "Log-scale axes stored as if linear. scatter with x_min=103, x_max=105 represents 1,000–100,000 in log₁₀ space"

Scatter chiếm 19.4% training data (52,163 mẫu). Với log-scale bug, geometric mapper trả về tọa độ sai cho một phần lớn scatter charts. Khi SLM được train để "sửa OCR" dựa trên context này, nó sẽ nhận input sai và có thể học cách compensate theo cách không dự đoán được.

---

### 6. Test suite — Coverage cao nhưng depth thấp

294 tests, 100% pass — nhưng phân tích cho thấy tests đang kiểm tra **structure**, không kiểm tra **correctness**:

- `test_element_detector.py`: không có test nào cho pie slice detection thực sự
- Test bar detection dùng synthetic image đơn giản, không test edge cases thực tế
- `test_geometric_mapper.py` chưa xem, nhưng log-scale bug đã tồn tại — tức là test không cover case đó

Đây là dạng **false confidence** — số test cao tạo cảm giác hệ thống ổn định, nhưng các failure mode quan trọng nhất đều nằm ngoài test coverage.

---

### Tóm lại — Thứ tự ưu tiên thực sự

Đối chiếu tất cả, thứ tự vấn đề cần giải quyết theo impact thực tế:

1. **Fix dead code pie slice** + thêm test thực sự — quick win, unblock toàn bộ pie pipeline
2. **Build benchmark chuẩn hóa trước khi train SLM lại** — không có benchmark thì không biết full training có thực sự improve không
3. **Area chart extraction** cần được nhìn nhận là "chưa implement" thay vì "implemented with low confidence"
4. **Log-scale detection** cho scatter — ảnh hưởng 52K training samples
5. **SLM full training** — chỉ nên làm sau khi 1-4 được giải quyết, nếu không sẽ lặp lại vòng "train xong không biết tại sao kết quả kém"