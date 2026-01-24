Geo-SLM: Kiến Trúc Hệ Thống Định Danh Toán Học và Tái Cấu Trúc Biểu Đồ Dựa Trên Hình Học Tô-pô và Cấu Trúc Tuyến Tính Phân Đoạn
Tóm tắt Điều hành
Báo cáo nghiên cứu chuyên sâu này trình bày khung kiến trúc và thuật toán cho Geo-SLM (Geometric Small Language Model - Mô hình Ngôn ngữ Nhỏ Hình học), một hệ thống tiên tiến được thiết kế để giải quyết bài toán chuyển đổi ảnh biểu đồ raster sang các thực thể toán học (Mathematical Identities) với độ chính xác tuyệt đối. Nghiên cứu này bác bỏ cách tiếp cận "hộp đen" (black-box) của các mô hình Deep Learning End-to-End hiện hành (như DePlot, MatCha) vốn thường gặp vấn đề về ảo giác dữ liệu và sai số tọa độ. Thay vào đó, chúng tôi đề xuất một phương pháp tiếp cận "dưới lên" (bottom-up), coi biểu đồ là tập hợp các cấu trúc tuyến tính phân đoạn (Piecewise Linear) và các thực thể hình học rời rạc.

Hệ thống được xây dựng dựa trên bốn trụ cột công nghệ chính: (1) Vector hóa chính xác sử dụng thuật toán Ramer-Douglas-Peucker (RDP) để xấp xỉ dữ liệu mà không làm mất đi các điểm cực trị quan trọng; (2) Xử lý ảnh dựa trên tô-pô thông qua phép biến đổi âm bản và skeletonization bảo toàn tính liên kết; (3) Trích xuất dữ liệu siêu chính xác với công nghệ phát hiện điểm chốt dưới mức điểm ảnh (sub-pixel) và phân tách chuỗi dựa trên đặc trưng hình thái (nét đứt/liền) thay vì màu sắc; và (4) Mô hình Geo-SLM đóng vai trò là động cơ suy luận không gian, ánh xạ các vector thô thành các định danh toán học có ngữ nghĩa. Báo cáo cung cấp phân tích chi tiết về thuật toán, so sánh hiệu năng, và đề xuất thiết kế JSON Schema chuẩn hóa cho việc lưu trữ dữ liệu biểu đồ.

1. Giới thiệu: Sự Cần Thiết Của Độ Chính Xác Hình Học Trong Khai Phá Dữ Liệu Biểu Đồ
1.1 Hạn Chế Của Các Mô Hình Deep Learning End-to-End Hiện Tại
Trong kỷ nguyên của Trí tuệ nhân tạo tạo sinh (Generative AI), các mô hình đa phương thức lớn (LMMs) như DePlot hay MatCha đã đạt được những bước tiến đáng kể trong việc "đọc" biểu đồ, coi đây là bài toán dịch từ hình ảnh sang văn bản (Image-to-Text). Các mô hình này hoạt động bằng cách mã hóa hình ảnh vào một không gian tiềm ẩn (latent space) và giải mã thành các token văn bản (ví dụ: HTML hoặc Markdown bảng).   

Tuy nhiên, đối với các ứng dụng yêu cầu độ chính xác cao như phân tích tài chính, kiểm toán khoa học, hay kỹ thuật ngược (reverse engineering), phương pháp này bộc lộ những khiếm khuyết nghiêm trọng:

Trôi dạt độ chính xác (Precision Drift): Do bản chất xác suất của mô hình ngôn ngữ, các giá trị số học được sinh ra thường là xấp xỉ ngữ nghĩa thay vì đo lường chính xác. Một sai số 1-2% trên biểu đồ logarit có thể dẫn đến sai lệch dữ liệu hàng chục lần.   

Ảo giác (Hallucination): Khi gặp các điểm dữ liệu bị che khuất hoặc các đường giao nhau phức tạp, mô hình End-to-End có xu hướng "bịa" ra các điểm dữ liệu hợp lý về mặt ngữ cảnh nhưng không tồn tại thực tế.   

Mất mát thông tin cấu trúc: Việc cố gắng hồi quy các đường cong trơn (smooth curves) thay vì cấu trúc tuyến tính phân đoạn (piecewise linear) làm mất đi các điểm gãy khúc (inflection points) vốn là các điểm dữ liệu thực tế được đo đạc.   

1.2 Chiến Lược Tiếp Cận Geo-SLM: Ưu Tiên Cấu Trúc Tuyến Tính
Hệ thống Geo-SLM được đề xuất dựa trên tiên đề rằng: Biểu đồ đường (Line Chart) về bản chất không phải là các đường cong liên tục, mà là tập hợp các đoạn thẳng nối các điểm dữ liệu rời rạc. Do đó, mục tiêu của hệ thống không phải là "nhìn" và "mô tả" biểu đồ, mà là "đo đạc" và "tái cấu trúc" lại bản chất toán học của nó.

Chúng tôi chuyển trọng tâm xử lý từ không gian pixel (raster) sang không gian vector thông qua các thuật toán hình học tính toán (Computational Geometry). Điều này đòi hỏi việc sử dụng các kỹ thuật xử lý ảnh cổ điển nhưng mạnh mẽ (như Skeletonization và Morphology) kết hợp với khả năng suy luận của các mô hình ngôn ngữ nhỏ (SLM) để hiểu ngữ cảnh.   

2. Kỹ Thuật Xử Lý Ảnh Nâng Cao: Bảo Toàn Tô-pô và Skeletonization
Bước đầu tiên và quan trọng nhất để chuyển đổi từ ảnh raster sang thực thể toán học là giảm chiều dữ liệu: từ các nét vẽ có độ dày (stroke width) thành các khung xương (skeleton) có độ rộng 1 pixel nhưng vẫn bảo toàn hoàn toàn cấu trúc tô-pô của đối tượng.

2.1 Ứng Dụng Ảnh Âm Bản (Negative Image Transformation)
Hầu hết các biểu đồ khoa học và tài chính được trình bày dưới dạng nền trắng, nét đen (hoặc màu). Tuy nhiên, các thuật toán hình thái học (morphological operations) và skeletonization thường được tối ưu hóa cho ảnh nhị phân với đối tượng là "trắng" (bit 1) trên nền "đen" (bit 0).   

Quy trình Chuyển đổi:

Nghịch đảo cường độ (Inversion): I 
neg
​
 (x,y)=Max 
val
​
 −I 
src
​
 (x,y). Trong không gian này, các đường lưới (grid lines) và chuỗi dữ liệu trở thành các "sợi dây phát sáng" trong không gian tối. Điều này tạo điều kiện thuận lợi cho việc áp dụng thuật toán "Grassfire" (lửa cháy lan) để tìm trục trung tâm.   

Tách nền cục bộ: Sử dụng phép toán White Top-Hat Transform trên ảnh âm bản. Phép toán này (I−Open(I)) giúp loại bỏ các biến đổi nền có tần số thấp (như bóng đổ, nhiễu nền) và chỉ giữ lại các cấu trúc mảnh, sáng (chính là các nét vẽ biểu đồ).   

Lợi ích: Đảm bảo các đường nét đứt (dashed lines) mờ nhạt không bị mất đi trong quá trình nhị phân hóa, vấn đề thường gặp khi xử lý ảnh dương bản thông thường.

2.2 Skeletonization Bảo Toàn Tô-pô (Topology Preservation)
Sau khi có ảnh nhị phân âm bản, bước tiếp theo là Skeletonization (làm mỏng). Yêu cầu tiên quyết cho Geo-SLM là thuật toán phải Bảo toàn Tô-pô (Homotopy Preserving). Điều này có nghĩa là số lượng thành phần liên thông (connected components) và số lỗ thủng (holes) phải bất biến trước và sau khi làm mỏng.   

Lựa chọn Thuật toán:

Vấn đề của Zhang-Suen: Thuật toán làm mỏng phổ biến Zhang-Suen thường gây ra hiện tượng mất liên kết tại các điểm giao nhau chéo (X-junctions) hoặc tạo ra các đoạn thừa (spurs) làm sai lệch cấu trúc dữ liệu.   

Giải pháp - Thuật toán Lee hoặc Guo-Hall: Chúng tôi đề xuất sử dụng thuật toán làm mỏng của Lee (1994) hoặc Guo-Hall. Các thuật toán này kiểm tra kỹ lưỡng các điều kiện liên thông trong cửa sổ 3x3 để đảm bảo rằng việc xóa một điểm ảnh không làm gãy một đường liên tục.   

Xử lý Giao điểm (Junction Resolution): Tại các điểm giao nhau của hai đường biểu đồ (ví dụ: hai chuỗi dữ liệu cắt nhau), thuật toán skeletonization chuẩn sẽ tạo ra một điểm nút (node) với bậc (degree) là 4. Việc xác định cặp cạnh nào thuộc về cùng một chuỗi dữ liệu là bài toán khó. Geo-SLM giải quyết vấn đề này ngay tại giai đoạn xử lý ảnh bằng cách duy trì một bản đồ Distance Transform (DT) song song. Giá trị DT tại điểm nút cho biết độ dày của nét vẽ gốc, hỗ trợ việc phân luồng dữ liệu sau này.   

3. Vector Hóa: Cấu Trúc Tuyến Tính Phân Đoạn và Ramer-Douglas-Peucker
Trái ngược với xu hướng sử dụng Splines (Bézier curves) để làm mượt biểu đồ, Geo-SLM tuân thủ nguyên tắc Tuyến tính Phân đoạn (Piecewise Linear). Lý do cốt lõi là trong khoa học dữ liệu, một đường biểu đồ được tạo thành từ các điểm mẫu rời rạc (x 
i
​
 ,y 
i
​
 ) nối với nhau bằng đường thẳng. Việc làm mượt (smoothing) thực chất là hành động "bịa" ra dữ liệu (hallucination) giữa các điểm mẫu.   

3.1 Thuật Toán Ramer-Douglas-Peucker (RDP)
Để chuyển đổi chuỗi pixel của skeleton thành các vector LineSegment (đoạn thẳng), chúng tôi sử dụng RDP.

Cơ chế hoạt động: RDP hoạt động theo cơ chế đệ quy, tìm điểm xa nhất so với đoạn thẳng nối điểm đầu và điểm cuối. Nếu khoảng cách này lớn hơn ngưỡng ϵ, điểm đó được giữ lại làm đỉnh (vertex) và đường cong được chia nhỏ.   

Tại sao RDP ưu việt hơn Visvalingam-Whyatt trong ngữ cảnh này?

Visvalingam-Whyatt: Loại bỏ điểm dựa trên diện tích tam giác hiệu dụng nhỏ nhất. Thuật toán này tối ưu cho hiển thị bản đồ vì nó giữ lại hình dáng tổng thể trơn tru.   

RDP: Giữ lại các điểm cực trị (local extrema). Trong biểu đồ, các đỉnh nhọn (peaks) và đáy (valleys) là các dữ liệu quan trọng nhất (ví dụ: giá cổ phiếu cao nhất, điểm gãy của xu hướng). RDP đảm bảo sai số vị trí không bao giờ vượt quá ϵ, bảo toàn tính chính xác của dữ liệu gốc.   

Chiến lược ϵ Thích ứng (Adaptive Epsilon): Thay vì dùng một ϵ cố định, Geo-SLM tính toán ϵ dựa trên độ dày nét vẽ cục bộ (từ Distance Transform).

ϵ 
local
​
 =α⋅Width 
stroke
​
 
Với α≈0.5, hệ thống đảm bảo rằng các dao động nhỏ do nhiễu in ấn (nhỏ hơn nửa độ dày nét vẽ) sẽ bị loại bỏ, trong khi mọi biến động dữ liệu thực sự đều được vector hóa thành đỉnh.

3.2 Nhận Diện Thực Thể Hình Học Rời Rạc
Hệ thống không chỉ xử lý đường nét (Polyline) mà còn phải định danh các đối tượng hình học khác:

LineSegment: Cấu trúc cơ bản, được định nghĩa bởi (P 
start
​
 ,P 
end
​
 ).

Marker (Điểm đánh dấu): Các điểm dữ liệu thường được biểu diễn bằng hình tròn, vuông, tam giác. Geo-SLM sử dụng Hough Circle Transform cho hình tròn và phân tích Contour Moments (Hu Moments) trên ảnh âm bản để phát hiện các hình dạng marker khác. Các marker này được coi là các node chính xác tuyệt đối trong đồ thị dữ liệu.   

BarRectangle (Thanh biểu đồ): Skeletonization thường phá hủy hình chữ nhật (tạo thành khung xương bao quanh). Do đó, đối với biểu đồ cột, hệ thống sử dụng thuật toán Contour Approximation (cv2.approxPolyDP) trên ảnh gốc để tìm các đường khép kín có 4 đỉnh và các góc xấp xỉ 90 độ.   

4. Trích Xuất Dữ Liệu: Độ Chính Xác Dưới Điểm Ảnh và Phân Tách Chuỗi
4.1 Xác Định Điểm Chốt Với Độ Chính Xác Dưới Điểm Ảnh (Sub-pixel Accuracy)
Tọa độ pixel là số nguyên (x,y)∈Z 
2
 . Tuy nhiên, đỉnh thực sự của dữ liệu có thể nằm giữa các pixel. Với một biểu đồ có trục hoành dài 1000 pixel đại diện cho 10 năm, sai số 0.5 pixel tương đương với sai số gần 2 ngày. Để đạt độ chính xác tuyệt đối, Geo-SLM áp dụng kỹ thuật Sub-pixel Refinement.

Phương pháp: Tại mỗi đỉnh vector (x 
i
​
 ,y 
i
​
 ) tìm được bởi RDP, hệ thống trích xuất một vùng lân cận (ví dụ 5×5 pixel) trên ảnh gốc (đã làm mờ nhẹ bằng Gaussian để giảm nhiễu). Chúng tôi áp dụng phương pháp Taylor Expansion hoặc khớp bề mặt bậc hai (Quadratic Surface Fitting) để tìm cực trị của hàm cường độ sáng.   

f(x,y)≈ax 
2
 +by 
2
 +cxy+dx+ey+f
Điểm cực trị ( 
x
^
 , 
y
^
​
 ) của bề mặt này cho tọa độ thực của dữ liệu với độ chính xác lên tới 1/10 hoặc 1/20 pixel.   

4.2 Phân Tách Chuỗi Dữ Liệu (Series Grouping) Dựa Trên Hình Thái
Một thách thức lớn là khi nhiều đường biểu đồ giao nhau hoặc đứt đoạn (do nhãn đè lên). Các phương pháp truyền thống dựa vào màu sắc thường thất bại với ảnh đen trắng hoặc khi màu sắc bị nén (JPEG artifacts). Geo-SLM sử dụng Đặc trưng Hình thái (Morphological Features).

4.2.1 Phân Tích Chuỗi Nét Đứt/Liền (Dash Pattern Analysis)
Thay vì màu sắc, chúng tôi phân tích chuỗi các đoạn thẳng và khoảng trống dọc theo đường đi của vector.

Thuật toán: Duyệt dọc theo skeleton graph. Tại mỗi cạnh, đo độ dài đoạn liền (L 
segment
​
 ) và độ dài khoảng hở (L 
gap
​
 ).

Hồ sơ Hình thái (Morphological Profile): Một chuỗi dữ liệu được định danh bởi vector đặc trưng V 
style
​
 =[μ 
L
​
 ,σ 
L
​
 ,μ 
G
​
 ,σ 
G
​
 ] (trung bình và phương sai của độ dài nét và khoảng hở).

Nét liền: μ 
G
​
 ≈0.

Nét đứt (Dashed): μ 
L
​
 ≫Width, σ 
L
​
  nhỏ.

Nét chấm (Dotted): μ 
L
​
 ≈Width.

Nét gạch-chấm (Dash-dot): Chuỗi lặp lại của (L 
long
​
 ,G,L 
short
​
 ,G).

Sử dụng thuật toán gom cụm (Clustering) như DBSCAN trên không gian vector V 
style
​
 , hệ thống có thể liên kết các đoạn rời rạc thành một chuỗi dữ liệu thống nhất ngay cả khi chúng bị ngắt quãng, mà không cần thông tin màu sắc.   

4.2.2 Giải Quyết Giao Điểm (Junction Resolution) Bằng Độ Dốc
Tại các điểm giao nhau (ngã tư), Geo-SLM sử dụng nguyên lý Gestalt về sự liên tục (Good Continuation). Chúng tôi mô hình hóa bài toán dưới dạng tối ưu hóa chi phí trên đồ thị. Hàm chi phí C để nối cạnh vào e 
in
​
  với cạnh ra e 
out
​
 :   

C(e 
in
​
 ,e 
out
​
 )=w 
1
​
 ⋅∣θ 
in
​
 −θ 
out
​
 ∣+w 
2
​
 ⋅∣κ 
in
​
 −κ 
out
​
 ∣+w 
3
​
 ⋅∣Width 
in
​
 −Width 
out
​
 ∣
Trong đó:

θ: Góc tiếp tuyến (Tangent angle) - ưu tiên đường thẳng hoặc cong trơn, phạt các góc gập đột ngột.

κ: Độ cong (Curvature).

Width: Độ dày nét vẽ (từ Distance Transform).

Việc giảm thiểu hàm chi phí này trên toàn bộ đồ thị giúp "gỡ rối" các đường giao nhau một cách chính xác.   

5. Quy Trình Ánh Xạ Không Gian và Kiến Trúc Geo-SLM
Sau khi vector hóa, dữ liệu ở dạng tọa độ ảnh (u,v). Bước cuối cùng là ánh xạ sang tọa độ thực tế (x 
data
​
 ,y 
data
​
 ) và gán ngữ nghĩa. Đây là vai trò của Geo-SLM.

5.1 Geo-SLM: Mô Hình Ngôn Ngữ Nhỏ Chuyên Biệt
Khác với các LLM khổng lồ (như GPT-4V), Geo-SLM là một mô hình nhỏ (Small Language Model, < 7B tham số) hoặc một mạng Graph Neural Network (GNN) được tinh chỉnh chuyên biệt trên dữ liệu hình học.   

Đầu vào: Không phải ảnh pixel, mà là Geometric Scene Graph (Đồ thị cảnh quan hình học) bao gồm các node (trục, nhãn, đường, marker) và cạnh (quan hệ không gian: "nằm dưới", "thẳng hàng", "gần").

Nhiệm vụ: Suy luận logic để xác định vai trò của các thực thể. Ví dụ: "Chuỗi văn bản '2020' thẳng hàng với trục hoành -> Đây là nhãn trục X". "Đoạn thẳng này nằm gần nhãn 'Doanh thu' trong chú thích -> Đây là chuỗi Doanh thu".   

5.2 Quy Trình Ánh Xạ Không Gian (Spatial Mapping)
Hệ thống thực hiện quy trình hiệu chỉnh (Calibration) tự động:

Phát hiện Trục & Vạch chia (Ticks): Geo-SLM xác định các đường thẳng đóng vai trò trục tọa độ và các vạch ngắn (ticks) vuông góc với chúng.   

Liên kết OCR: Sử dụng OCR để đọc giá trị các nhãn tại vạch chia.

Xây dựng Ma trận Chuyển đổi:

Với trục tuyến tính: Tìm a,b sao cho x 
data
​
 =a⋅u 
pixel
​
 +b bằng phương pháp Bình phương tối thiểu (Least Squares) trên tất cả các cặp (vạch chia, giá trị OCR) đã nhận diện. Điều này giúp loại bỏ sai số cục bộ của một nhãn đơn lẻ.   

Với trục Logarit: Phát hiện dựa trên khoảng cách không đều giữa các vạch chia (khoảng cách giảm dần theo lũy thừa). Mô hình chuyển đổi: log(y 
data
​
 )=a⋅v 
pixel
​
 +b.   

5.3 So Sánh Với Deep Learning End-to-End
Đặc Điểm	
Deep Learning End-to-End (DePlot/MatCha) 

Geo-SLM (Đề xuất)
Đầu vào	Ảnh Raster thô (Pixels)	Thực thể Hình học (Vector Graph)
Cơ chế trích xuất	Sinh văn bản tự hồi quy (Auto-regressive)	Ánh xạ tọa độ tất định (Deterministic)
Độ chính xác	Xấp xỉ (Token-based), dễ bị trôi dạt	Tuyệt đối (Float64), Sub-pixel
Xử lý chồng lấn	Dễ bị ảo giác (Hallucination)	Phân tách dựa trên Tô-pô & Hình thái
Hiệu năng tính toán	Nặng (Vision Encoder + LLM Decoding)	Nhẹ (Vector hóa + GNN/SLM)
Tính kiểm chứng	Hộp đen (Black-box)	Minh bạch (Có thể vẽ lại vector đè lên ảnh gốc)
  
6. Thiết Kế JSON Schema
Để chuẩn hóa đầu ra, chúng tôi thiết kế một cấu trúc JSON chặt chẽ, tách biệt giữa dữ liệu hình học (để kiểm chứng) và dữ liệu ngữ nghĩa (để phân tích). Schema này tương thích với các thư viện vẽ biểu đồ như Vega-Lite nhưng giàu thông tin hơn về cấu trúc.   

JSON
{
  "$schema": "https://geo-slm.org/schema/v1/chart-identity.json",
  "meta": {
    "version": "1.0",
    "extraction_method": "Geo-SLM-Vectorization",
    "precision": "sub-pixel"
  },
  "coordinate_system": {
    "x_axis": {
      "type": "linear",
      "pixel_range": [100.5, 900.2],
      "domain_range": ,
      "label": "Time (months)",
      "mapping_function": "linear_regression"
    },
    "y_axis": {
      "type": "logarithmic", 
      "base": 10,
      "pixel_range": [500.0, 50.0],
      "domain_range": ,
      "label": "Value"
    }
  },
  "geometric_entities": [
    {
      "id": "series_01",
      "semantic_role": "data_series",
      "style": {
        "stroke_type": "dashed",
        "dash_pattern": [10.0, 5.0], // Đơn vị pixel: [nét, khoảng]
        "stroke_width": 2.5
      },
      "identities":
        },
        {
          "type": "marker_group",
          "symbol": "triangle",
          "locations": [
             { "x_img": 150.8, "y_img": 400.1 } // Marker trùng khớp với đỉnh polyline
          ]
        }
      ]
    }
  ],
  "relations": {
    "grouping": ["series_01", "legend_item_A"] // Liên kết logic
  }
}
Điểm nhấn của Schema:

is_vertex: Cờ đánh dấu các điểm là đỉnh thực sự của RDP (dữ liệu đo đạc) so với các điểm nội suy.

dash_pattern: Lưu trữ đặc trưng hình thái cụ thể để tái lập hoặc kiểm chứng.

Song hành Tọa độ: Lưu cả x_img (pixel) và x_val (giá trị) cho phép hậu kiểm (audit) độ chính xác của phép ánh xạ.

7. Kết Luận
Nghiên cứu này khẳng định rằng để đạt được độ chính xác tuyệt đối trong chuyển đổi biểu đồ, chúng ta phải từ bỏ cách tiếp cận thuần túy dựa trên thị giác máy tính (pixel) hoặc mô hình ngôn ngữ lớn (token). Hệ thống Geo-SLM đề xuất một hướng đi lai ghép: sử dụng Toán học Hình thái (Mathematical Morphology) để trích xuất cấu trúc tô-pô, Hình học Tính toán (RDP, Sub-pixel) để đảm bảo độ chính xác tọa độ, và Suy luận Cấu trúc (Graph SLM) để định danh ngữ nghĩa.

Việc chuyển dịch trọng tâm sang cấu trúc Tuyến tính Phân đoạn và phân tách dựa trên Hình thái nét vẽ giúp hệ thống miễn nhiễm với các sai số thường gặp của mô hình End-to-End, mở ra khả năng tự động hóa tin cậy cho các quy trình số hóa tài liệu kỹ thuật và khoa học quy mô lớn.

Tài liệu tham khảo tích hợp:

: Hạn chế của Deep Learning End-to-End.   

: Thuật toán Ramer-Douglas-Peucker và bảo toàn đỉnh.   

: Skeletonization, thuật toán Lee/Guo-Hall và xử lý ảnh âm bản.   

: Kỹ thuật Sub-pixel Accuracy.   

: Phân tách nét đứt/liền bằng hình thái học.   

: Mô hình ngôn ngữ nhỏ (SLM) và xử lý đồ thị.   

: Cấu trúc dữ liệu và JSON Schema.   


ritvik19.medium.com
Papers Explained 256: DePlot - Ritvik Rastogi
Opens in a new window

researchgate.net
MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering | Request PDF - ResearchGate
Opens in a new window

arxiv.org
ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering - arXiv
Opens in a new window

meilisearch.com
GraphRAG vs. Vector RAG: Side-by-side comparison guide - Meilisearch
Opens in a new window

medium.com
Hands on DePlot - Medium
Opens in a new window

vega.github.io
Line Chart Example - Vega-Lite
Opens in a new window

mdpi.com
Enhancing Small Language Models for Graph Tasks Through Graph Encoder Integration
Opens in a new window

mdpi.com
State of the Art and Future Directions of Small Language Models: A Systematic Review
Opens in a new window

tirthajyoti.github.io
22. Skeletonizing an image — Hands-on introduction to Scikit-image methods
Opens in a new window

research.manchester.ac.uk
Hardware implementation of skeletonization algorithm for parallel asynchronous image processing - Research Explorer - The University of Manchester
Opens in a new window

scikit-image.org
skimage.morphology — skimage 0.26.0 documentation
Opens in a new window

scikit-image.org
Morphological Filtering — skimage 0.25.2 documentation
Opens in a new window

inf.u-szeged.hu
Skeletonization
Opens in a new window

livrepository.liverpool.ac.uk
Skeletonisation Algorithms with Theoretical Guarantees for Unorganised Point Clouds with High Levels of Noise - The University of Liverpool Repository
Opens in a new window

reddit.com
Thinning algorithm : r/compsci - Reddit
Opens in a new window

docs.rs
skeletonize - Rust - Docs.rs
Opens in a new window

scikit-image.org
Skeletonize — skimage 0.25.2 documentation
Opens in a new window

ipol.im
Finding the Skeleton of 2D Shape and Contours: Implementation of Hamilton-Jacobi Skeleton - IPOL Journal
Opens in a new window

en.wikipedia.org
Ramer–Douglas–Peucker algorithm - Wikipedia
Opens in a new window

pmc.ncbi.nlm.nih.gov
GPU-Accelerated RDP Algorithm for Data Segmentation - PMC - NIH
Opens in a new window

stackoverflow.com
line simplification algorithm: Visvalingam vs Douglas-Peucker - Stack Overflow
Opens in a new window

bost.ocks.org
Line Simplification - Mike Bostock
Opens in a new window

reddit.com
Simplify a polyline or polygon with Visvalingham-Whyatt or Douglas-Peucker : r/Python
Opens in a new window

en.wikipedia.org
Hough transform - Wikipedia
Opens in a new window

pmc.ncbi.nlm.nih.gov
A Two-Stage Automatic Color Thresholding Technique - PMC - PubMed Central
Opens in a new window

fuzzylabs.ai
Checkbox Detection with OpenCV - Fuzzy Labs
Opens in a new window

medium.com
Clustering in Image Processing Explained | by Amit Yadav - Medium
Opens in a new window

staff.fnwi.uva.nl
6.3. Subpixel Localization of Local Structure - Homepages of UvA/FNWI staff
Opens in a new window

accedacris.ulpgc.es
Accurate Subpixel Edge Location based on Partial Area Effect - accedaCRIS
Opens in a new window

cseweb.ucsd.edu
Subpixel Corner Detection for Tracking Applications using CMOS Camera Technology
Opens in a new window

mdpi.com
A New Approach to Detect Hand-Drawn Dashed Lines in Engineering Sketches - MDPI
Opens in a new window

wiredcraft.com
Dashed Line Segmentation in D3.js - Wiredcraft
Opens in a new window

openaccess.thecvf.com
Neural Recognition of Dashed Curves With Gestalt Law of Continuity - CVF Open Access
Opens in a new window

stackoverflow.com
Separate crossings segments in binarised image - Stack Overflow
Opens in a new window

researchgate.net
(PDF) Vectorization of line drawing image based on junction analysis - ResearchGate
Opens in a new window

mrt.kit.edu
Application of Line Clustering Algorithms for Improving Road Feature Detection - KIT - MRT
Opens in a new window

arxiv.org
Real-Time Scene Graph Generation - arXiv
Opens in a new window

openaccess.thecvf.com
SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities - CVF Open Access
Opens in a new window

mathworks.com
Specify Axis Tick Values and Labels - MATLAB & Simulink - MathWorks
Opens in a new window

tableau.com
An Extension of Wilkinson's Algorithm for Positioning Tick Labels on Axes - Tableau
Opens in a new window

pmc.ncbi.nlm.nih.gov
Automatic Calibration of a Two-Axis Rotary Table for 3D Scanning Purposes - PMC
Opens in a new window

en.wikipedia.org
Logarithmic scale - Wikipedia
Opens in a new window

youtube.com
Extract data from Log-Log plots/graphs | webplotdigitizer | Drawing/Graphing-12 - YouTube
Opens in a new window

vega.github.io
Line | Vega-Lite
Opens in a new window

json-schema.org
Creating your first schema - JSON Schema
Opens in a new window

reddit.com
Help removing some overlapping lines : r/PlotterArt - Reddit
Opens in a new window

mikekling.com
Comparing Algorithms for Dispersing Overlapping Rectangles - Michael Kling
Opens in a new window

open.clemson.edu
MORPHOLOGICAL CHARTS: A SYSTEMATIC EXPLORATION OF QUALITATIVE DESIGN SPACE - Clemson OPEN
Opens in a new window

repository.upenn.edu
GROUPING STRAIGHT LINE SEGMENTS IN REAL IMAGES - University of Pennsylvania
Opens in a new window

en.wikipedia.org
Bresenham's line algorithm - Wikipedia
Opens in a new window

mdpi.com
Research on Image Stitching Algorithm Based on Point-Line Consistency and Local Edge Feature Constraints - MDPI
Opens in a new window

encord.com
Guide to Image Segmentation in Computer Vision: Best Practices - Encord
Opens in a new window

arxiv.org
Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) - arXiv
Opens in a new window

nlpr.ia.ac.cn
An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge
Opens in a new window

neo4j.com
Knowledge Graph vs. Vector RAG: Optimization & Analysis - Neo4j
Opens in a new window

arxiv.org
Enhancing Question Answering on Charts Through Effective Pre-training Tasks - arXiv
Opens in a new window

aclanthology.org
SIMPLOT: Enhancing Chart Question Answering by Distilling Essentials - ACL Anthology
Opens in a new window

openaccess.thecvf.com
VQualA 2025 Challenge on Image Super-Resolution Generated Content Quality Assessment - CVF Open Access
Opens in a new window

arxiv.org
Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art - arXiv
Opens in a new window

digitalocean.com
Exploring SOTA: A Guide to Cutting-Edge AI Models | DigitalOcean
Opens in a new window

neurips.cc
NeurIPS 2024 Spotlight Posters
Opens in a new window

isamu-website.medium.com
Understanding Spectral Graph Theory and then the current SOTA of GNNs - Isamu Isozaki
Opens in a new window

microsoft.com
ChartOCR: Data Extraction from Charts Images via a Deep Hybrid Framework - Microsoft
Opens in a new window

automeris.io
automeris.io: Computer vision assisted data extraction from charts using WebPlotDigitizer
Opens in a new window

stackoverflow.com
Plot digitization - scraping sample values from an image of a graph - Stack Overflow
Opens in a new window

youtube.com
ChartDetective: Easy and Accurate Interactive Data Extraction from Complex Vector Charts
Opens in a new window

mdpi.com
A Vector Line Simplification Algorithm Based on the Douglas–Peucker Algorithm, Monotonic Chains and Dichotomy - MDPI
Opens in a new window

hpi.de
Analysis of the Parameter Configuration of the Ramer-Douglas-Peucker Algorithm for Time Series Compression
Opens in a new window

en.wikipedia.org
Visvalingam–Whyatt algorithm - Wikipedia
Opens in a new window

martinfleischmann.net
Line simplification algorithms | Martin Fleischmann
Opens in a new window

www3.cs.stonybrook.edu
TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons - Stony Brook Computer Science
Opens in a new window

docs.opencv.org
Harris Corner Detection - OpenCV Documentation
Opens in a new window

stackoverflow.com
how to improve corner detection accuracy - Stack Overflow
Opens in a new window

geeksforgeeks.org
Difference between Traditional Computer Vision Techniques and Deep Learning-based Approaches - GeeksforGeeks
Opens in a new window

ncbi.nlm.nih.gov
Working with JSON Lines data reports - NCBI - NIH
Opens in a new window

vega.github.io
Specification - Vega-Lite
Opens in a new window

cs.stanford.edu
Darkroom: Compiling High-Level Image Processing Code into Hardware Pipelines - Stanford Computer Science
Opens in a new window

pages.cs.wisc.edu
Image Alignment and Stitching: A Tutorial - cs.wisc.edu
Opens in a new window

mdpi.com
An Accuracy vs. Complexity Comparison of Deep Learning Architectures for the Detection of COVID-19 Disease - MDPI
Opens in a new window

researchgate.net
Complexity and computational cost comparison | Download Table - ResearchGate
Opens in a new window

pmc.ncbi.nlm.nih.gov
Comparison between Deep Learning and Conventional Machine Learning in Classifying Iliofemoral Deep Venous Thrombosis upon CT Venography - NIH
Opens in a new window

mdpi.com
Optimization Methods, Challenges, and Opportunities for Edge Inference: A Comprehensive Survey - MDPI
Opens in a new window

chemrxiv.org
Revolutionizing Scientific Figure Decoding: Benchmarking LLM Data Extraction Performance - ChemRxiv
Opens in a new window

developer.nvidia.com
Query Graphs with Optimized DePlot Model | NVIDIA Technical Blog
Opens in a new window

arxiv.org
Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction - arXiv
Opens in a new window

arxiv.org
[1809.02697] Optimizing CNN Model Inference on CPUs - arXiv
Opens in a new window

developer.arm.com
Getting real-time CNN inference for AR on mobile - Arm Developer
Opens in a new window

opus4.kobv.de
Performance analysis of CNN speed and power consumption among CPUs, GPUs and an FPGA for vehicular applications - OPUS
Opens in a new window

researchgate.net
A Real-Time and Hardware-Efficient Processor for Skeleton-Based
Opens in a new window

usenix.org
Optimizing CNN Model Inference on CPUs - USENIX
Opens in a new window

docs.anychart.com
Data From JSON | Working with Data - AnyChart Documentation
Opens in a new window

json-schema.org
Miscellaneous Examples - JSON Schema
Opens in a new window

helm.sh
Charts - Helm
Opens in a new window

docs.datadoghq.com
Graphing with JSON - Datadog Docs
Opens in a new window

vega.github.io
Data - Vega-Lite
Opens in a new window

vega.github.io
Bar and Line Chart - Vega-Lite
Opens in a new window

vega.github.io
Example Gallery | Vega-Lite
Opens in a new window

plotdigitizer.com
How to Extract Data from Graphs or Images in Scientific Papers? - PlotDigitizer
Opens in a new window

idl.cs.washington.edu
Reverse-Engineering Visualizations: Recovering Visual Encodings from Chart Images - UW Interactive Data Lab
Opens in a new window

archives.gov
Appendix A: Tables of File Formats - National Archives
Opens in a new window

apriorit.com
How to Reverse Engineer a Proprietary File Format: A Brief Guide with Practical Examples
Opens in a new window

jamie-wong.com
Reverse Engineering Instruments' File Format - Jamie Wong
Opens in a new window

docs.ogc.org
Features and geometry – Part 1: Feature models
Opens in a new window

arxiv.org
Geometric Deep Learning for Computer-Aided Design: A Survey - arXiv
Opens in a new window

dev.opencascade.org
Modeling Data - Open CASCADE Technology
Opens in a new window

dataroots.io
A gentle introduction to Geometric Deep Learning - dataroots
Opens in a new window

smartdraw.com
Flowchart Symbols - SmartDraw
Opens in a new window

mdpi.com
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - MDPI
Opens in a new window

imagej.net
MorphoLibJ - ImageJ Wiki
Opens in a new window

nv5geospatialsoftware.com
Convolution and Morphology Filters - NV5 Geospatial Software
Opens in a new window

mathworks.com
Morphological Operations - MATLAB & Simulink - MathWorks
Opens in a new window

medium.com
Image Processing Class #6 — Morphological Filter | by Pitchaya Thipkham - Medium
Opens in a new window

nanpa.org
The Positive Traits of Negative Space - NANPA
Opens in a new window

adam.edu.sg
Why Negative Space Makes Your Photographs Look Better
Opens in a new window

homepages.inf.ed.ac.uk
Line Detection
Opens in a new window

mdpi.com
Negative Samples for Improving Object Detection—A Case Study in AI-Assisted Colonoscopy for Polyp Detection - MDPI
Opens in a new window

scikit-image.org
Use rolling-ball algorithm for estimating background intensity - Scikit-image
Opens in a new window

mdpi.com
A Single Image Enhancement Technique Using Dark Channel Prior - MDPI
Opens in a new window

dsp.stackexchange.com
Segmentation of Dark Blobs with Varying Background - Signal Processing Stack Exchange
Opens in a new window

stackoverflow.com
An algorithm for selecting a dark color similar to a light color - Stack Overflow
Opens in a new window

hilandtom.com
Quantitative analysis of skeletonisation algorithms for modelling of branches
Opens in a new window

papers.neurips.cc
Data Skeletonization via Reeb Graphs
Opens in a new window

stackoverflow.com
Finding intersections of a skeletonised image in python opencv - Stack Overflow
Opens in a new window

stackoverflow.com
Determine overlapping lines (to remove them)? - Stack Overflow
Opens in a new window

computergraphics.stackexchange.com
Best technique to draw overlapping colored line segments that follow the same route
Opens in a new window

community.adobe.com
Image trace overlapping trace lines twice (Only just started doing this)
Opens in a new window

reddit.com
How would I go about making overlapping lines easily like this? : r/AdobeIllustrator - Reddit
Opens in a new window

youtube.com
Image Trace Abutting Versus Overlapping Paths - YouTube
Opens in a new window

www-sop.inria.fr
Fidelity vs. Simplicity: a Global Approach to Line Drawing Vectorization - Inria
Opens in a new window

researchgate.net
(PDF) Path Openings and Closings - ResearchGate
Opens in a new window

theory.stanford.edu
Heuristics - Stanford CS Theory
Opens in a new window

theclassytim.medium.com
Using Image Processed Hough Lines for Path Planning Applications | by Tim Chinenov
Opens in a new window

pages.cvc.uab.es
A Graph based Approach for Segmenting Touching Lines in
Opens in a new window

ri.cmu.edu
Kalman Filter-based Algorithms for Estimating Depth from Image Sequences - Carnegie Mellon University's Robotics Institute
Opens in a new window

en.wikipedia.org
Kalman filter - Wikipedia
Opens in a new window

arxiv.org
An Analysis of Kalman Filter based Object Tracking Methods for Fast-Moving Tiny Objects
Opens in a new window

bzarg.com
How a Kalman filter works, in pictures - Bzarg
Opens in a new window

engineering.purdue.edu
A New Kalman-Filter-Based Framework for Fast and Accurate Visual Tracking of Rigid Objects - Purdue College of Engineering
Opens in a new window

newline.co
Convolutional Neural Networks vs OpenCV: Performance Comparison in Computer Vision AI - Newline.co
Opens in a new window

mdpi.com
Comparison of CNN-Based Architectures for Detection of Different Object Classes - MDPI
Opens in a new window

researchgate.net
Comparison of CNN-Based Architectures for Detection of Different Object Classes
Opens in a new window

technolynx.com
Deep Learning vs. Traditional Computer Vision Methods - TechnoLynx
Opens in a new window

cs.swarthmore.edu
Comparing Deep Neural Networks and Traditional Vision Algorithms in Mobile Robotics - Swarthmore College
Opens in a new window

arxiv.org
Deep Learning vs. Traditional Computer Vision - arXiv
Opens in a new window

mdpi.com
Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study - MDPI
Opens in a new window

vbn.aau.dk
Explorative Comparison Between Classic Computer Vision Techniques and Deep Learning
Opens in a new window

mdpi.com
Edge Intelligence: A Review of Deep Neural Network Inference in Resource-Limited Environments - MDPI
Opens in a new window

arxiv.org
A Study on Inference Latency for Vision Transformers on Mobile Devices - arXiv
Opens in a new window

mdpi.com
Impact of Thermal Throttling on Long-Term Visual Inference in a CPU-Based Edge Device
Opens in a new window

medium.com
Optimizing Resnet-50: 8X inference throughput with just a few commands - Medium
Opens in a new window

discuss.pytorch.org
What is a normal inference speed for Resnet34 and how to improve it? - PyTorch Forums
Opens in a new window

pmc.ncbi.nlm.nih.gov
Research on Image Stitching Algorithm Based on Point-Line Consistency and Local Edge Feature Constraints - NIH
Opens in a new window

idl.cs.washington.edu
Fast and Flexible Overlap Detection for Chart Labeling with Occupancy Bitmap - UW Interactive Data Lab
Opens in a new window

mathoverflow.net
Algorithm for finding minimally overlapping paths in a graph - MathOverflow
Opens in a new window

stackoverflow.com
Algorithm to detect overlapping rows of two images - Stack Overflow
Opens in a new window

medium.com
Overlapping lines. Daily Coding Problem n. 27 | by Nicola Moro | Medium
Opens in a new window

media.disneyanimation.com
Topology-Driven Vectorization of Clean Line Drawings - Walt Disney Animation Studios
Opens in a new window

cgl.ethz.ch
Semantic Segmentation for Line Drawing Vectorization Using Neural Networks - Computer Graphics Laboratory
Opens in a new window

news.mit.edu
Smoothing out sketches' rough edges | MIT News | Massachusetts Institute of Technology
Opens in a new window

igl.ethz.ch
Cusps of Characteristic Curves and Intersection-Aware Visualization of Path and Streak Lines - Interactive Geometry Lab
Opens in a new window

apps.dtic.mil
Curve and Polygon Evolution Techniques for Image Processing - DTIC
Opens in a new window

phtu-cs.github.io
Continuous Curve Textures - GitHub Pages
Opens in a new window

en.wikipedia.org
Edge detection - Wikipedia
Opens in a new window

igl.ethz.ch
Single-Line Drawing Vectorization
Opens in a new window

cecilialeiqi.github.io
Vectorization of line drawing image based on junction analysis - Qi Lei
Opens in a new window

erickjb.com
Singularity-Free Frame Fields for Line Drawing Vectorization - Erick Jimenez Berumen
Opens in a new window

arxiv.org
[1801.01922] Vectorization of Line Drawings via PolyVector Fields - arXiv
Opens in a new window

cs-people.bu.edu
Singularity-Free Frame Fields - for Line Drawing Vectorization - Boston University
Opens in a new window

atlassian.com
How to choose colors for data visualizations - Atlassian
Opens in a new window

developers.google.com
Visualization: Combo Chart - Google for Developers
Opens in a new window

stackoverflow.com
Changing Dash chart series colors - python - Stack Overflow
Opens in a new window

plotly.com
Bar charts in Python - Plotly
Opens in a new window

datylon.com
80 types of charts & graphs for data visualization (with examples) - Datylon
Opens in a new window

developers.arcgis.com
Charts Interfaces | Overview | ArcGIS Maps SDK for JavaScript - Esri Developer
Opens in a new window

stackoverflow.com
Representing a graph in JSON - Stack Overflow
Opens in a new window

json-schema.org
JSON Schema examples
Opens in a new window

helm.sh
Charts - Helm
Opens in a new window

vega.github.io
Multi Series Line Chart | Vega-Lite
Opens in a new window

vega.github.io
Multi Series Line Chart with an Interactive Line Highlight - Vega-Lite
Opens in a new window

vega.github.io
Multi Series Line Chart with Labels | Vega-Lite
Opens in a new window

vega.github.io
Multi Series Line Chart with Tooltip | Vega-Lite
Opens in a new window

chartjs.org
Line Chart - Chart.js
Opens in a new window

chartjs.org
Data structures - Chart.js
Opens in a new window

chartjs.org
Step-by-step guide - Chart.js
Opens in a new window

stackoverflow.com
Drawing line chart in chart.js with json data - Stack Overflow
Opens in a new window

youtube.com
Add JSON data to ChartJS - YouTube
Opens in a new window

plotdigitizer.com
Documentation - PlotDigitizer
Opens in a new window

plotdigitizer.com
Export Extracted Data to CSV, JSON, and Other File Formats - PlotDigitizer
Opens in a new window

automeris.io
Digitize Charts - WebPlotDigitizer
Opens in a new window

av8rdas.com
Plot Digitizer
Opens in a new window

pmc.ncbi.nlm.nih.gov
Efficient Skeletonization of Volumetric Objects - PMC - NIH
Opens in a new window

mdpi.com
Techniques and Algorithms for Hepatic Vessel Skeletonization in Medical Images: A Survey
Opens in a new window

arxiv.org
Does the Skeleton-Recall Loss Really Work? - arXiv
Opens in a new window

researchgate.net
A Survey on Skeletons in Digital Image Processing - ResearchGate
Opens in a new window

pmc.ncbi.nlm.nih.gov
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - PMC - NIH
Opens in a new window

pmc.ncbi.nlm.nih.gov
Real-Time Thinning Algorithms for 2D and 3D Images using GPU processors - PMC
Opens in a new window

bioimagebook.github.io
Morphological operations - Introduction to Bioimage Analysis
Opens in a new window

graphics.ics.uci.edu
Morphological Image Processing
Opens in a new window

geeksforgeeks.org
Different Morphological Operations in Image Processing - GeeksforGeeks
Opens in a new window

openaccess.thecvf.com
A skeletonization algorithm for gradient-based optimization - CVF Open Access
Opens in a new window

pmc.ncbi.nlm.nih.gov
A topology-preserving approach to the segmentation of brain images with multiple sclerosis lesions - PMC - PubMed Central
Opens in a new window

arxiv.org
Topology-Preserving Downsampling of Binary Images - arXiv
Opens in a new window

arxiv.org
[1804.01622] Image Generation from Scene Graphs - arXiv
Opens in a new window

openaccess.thecvf.com
Unconditional Scene Graph Generation - CVF Open Access
Opens in a new window

ojs.aaai.org
Scene Graph-Grounded Image Generation | Proceedings of the AAAI Conference on Artificial Intelligence
Opens in a new window

github.com
ChocoWu/Awesome-Scene-Graph-Generation - GitHub
Opens in a new window

pmc.ncbi.nlm.nih.gov
Zero-shot visual reasoning through probabilistic analogical mapping - PubMed Central
Opens in a new window

arxiv.org
Visually Descriptive Language Model for Vector Graphics Reasoning - arXiv
Opens in a new window

aclanthology.org
ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild - ACL Anthology
Opens in a new window

visualsketchpad.github.io
Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models
Opens in a new window

arxiv.org
Text-Based Reasoning About Vector Graphics - arXiv
Opens in a new window

arxiv.org
Navigate Complex Physical Worlds via Geometrically Constrained LLM - arXiv
Opens in a new window

arxiv.org
Do Large Language Models Truly Understand Geometric Structures? - arXiv
Opens in a new window

people.umass.edu
From Text to Visuals: Using LLMs to Generate Math Diagrams with Vector Graphics
Opens in a new window

medium.com
When LLMs Meet Geometry. Testing Gemini 3 on a PhD-level shape… | by Yogesh Haribhau Kulkarni (PhD) | Google Developer Experts | Nov, 2025 | Medium
Opens in a new window

aclanthology.org
Navigate Complex Physical Worlds via Geometrically Constrained LLM - ACL Anthology
Opens in a new window

pmc.ncbi.nlm.nih.gov
VESCL: an open source 2D vessel contouring library - PMC - PubMed Central
Opens in a new window

slicer.org
Modules:VMTKCenterlines - Slicer Wiki
Opens in a new window

fast-imaging.github.io
fast::CenterlineExtraction class - FAST | Documentation
Opens in a new window

github.com
An interactive software to extract vessel centerline - GitHub
Opens in a new window

stackoverflow.com
extract points that describe lines in a drawing - c++ - Stack Overflow
Opens in a new window

blend2d.com
Blend2D
Opens in a new window

intel.com
Rendering Libraries from Intel
Opens in a new window

ascelibrary.org
VectorGraphNET: Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings | Journal of Computing in Civil Engineering | Vol 39, No 6 - ASCE Library
Opens in a new window

artonson.github.io
Deep Vectorization of Technical Drawings - Alexey Artemov
Opens in a new window

github.com
homer-rain/Centerline-Extraction-of-Coronary-Vessels - GitHub
Opens in a new window

reddit.com
Must-know libraries/frameworks/technologies for C++ developer as of 2025 : r/cpp - Reddit
Opens in a new window

discourse.slicer.org
how to improve the CenterLine extraction - Support - 3D Slicer Community
Opens in a new window

researchgate.net
The Robust Vessel Segmentation and Centerline Extraction: One-Stage Deep Learning Approach - ResearchGate
Opens in a new window

youtube.com
Starting my modern C++ Project with CMake in 2024 - Jens Weller - YouTube
Opens in a new window

crates.io
skeletonize - crates.io: Rust Package Registry
Opens in a new window

lib.rs
Images — list of Rust libraries/crates // Lib.rs
Opens in a new window

users.rust-lang.org
Looking for image processing crate recommendation (& introduction) - Rust Users Forum
Opens in a new window

astcad.com.au
Raster To Vector Conversion: Convert Paper Drawings To Accurate Vector Files - Australian Design & Drafting Services
Opens in a new window

impactdigitizing.com
How to Convert Raster to Vector - Impact Digitizing
Opens in a new window

reddit.com
Seeking source to convert sketch to vector paths : r/PlotterArt - Reddit
Opens in a new window

graphicdesign.stackexchange.com
Convert a line drawing from raster to vector **LINES** - Graphic Design Stack Exchange
Opens in a new window

github.com
visioncortex/vtracer: Raster to Vector Graphics Converter - GitHub
Opens in a new window

reddit.com
PoTrace: convert bitmaps to vector graphics : r/programming - Reddit
Opens in a new window

medium.com
Manual Vectorization Vs. Auto-Tracing: Understanding the Key Differences | by Cre8iveSkill
Opens in a new window

en.wikipedia.org
Comparison of raster-to-vector conversion software - Wikipedia
Opens in a new window

news.ycombinator.com
Is this any better than POTrace(http://potrace.sourceforge.net/)? | Hacker News
Opens in a new window

arxiv.org
StarVector: Generating Scalable Vector Graphics Code from Images and Text - arXiv
Opens in a new window

microsoft.com
dashed line detection - Microsoft
Opens in a new window

pmc.ncbi.nlm.nih.gov
ChartLine: Automatic Detection and Tracing of Curves in Scientific Line Charts Using Spatial-Sequence Feature Pyramid Network - NIH
Opens in a new window

stackoverflow.com
Why are there dashed lines in segments? - Stack Overflow
Opens in a new window

journals.plos.org
Disambiguating Multi–Modal Scene Representations Using Perceptual Grouping Constraints | PLOS One - Research journals
Opens in a new window

qugank.github.io
Making Better Use of Edges via Perceptual Grouping - Yonggang Qi
Opens in a new window

pure.rug.nl
University of Groningen Algorithm that mimics human perceptual
Opens in a new window

pmc.ncbi.nlm.nih.gov
A computational model for gestalt proximity principle on dot patterns and beyond - NIH
Opens in a new window

stackoverflow.com
opencv detect dotted lines - python - Stack Overflow
Opens in a new window

medium.com
OpenCV Line Detection | by Amit Yadav - Medium
Opens in a new window

stackoverflow.com
Is there a method to use opencv-python to detect dashed cross lines? - Stack Overflow
Opens in a new window

stackoverflow.com
Detect dotted (broken) lines only in an image using OpenCV - Stack Overflow
Opens in a new window

answers.opencv.org
How to convert dashed lines to solid? - OpenCV Q&A Forum
Opens in a new window

mcgill.ca
Sectioning Technique | Engineering Design - McGill University
Opens in a new window

forums.sketchup.com
Joining Line Segments - Pro - SketchUp Community
Opens in a new window

computergraphics.stackexchange.com
Rounding the edges in a mitered line segment inside of a fragment shader
Opens in a new window

youtube.com
Solidworks Quick Tips - Merging Line Segments - Collinear - YouTube
Opens in a new window

forums.autodesk.com
Joining lines segments that don't quite join? - Autodesk Community
Opens in a new window

arxiv.org
Geo-LLaVA: A Large Multi-Modal Model for Solving Geometry Math Problems with Meta In-Context Learning - arXiv
Opens in a new window

youtube.com
LLMs vs SLMs: A developer's guide + NVIDIA insights - YouTube
Opens in a new window

arxiv.org
IMPROVING MULTIMODAL LLM'S ABILITY IN GEOMETRY PROBLEM SOLVING, REASONING, AND MULTISTEP SCORING - arXiv
Opens in a new window

researchgate.net
(PDF) Assessing sustainable land management (SLM) - ResearchGate
Opens in a new window

pmc.ncbi.nlm.nih.gov
A Transdisciplinary Framework to Bridge Science–Policy–Development Gaps in Global Land Management Initiatives - PubMed Central
Opens in a new window

cgspace.cgiar.org
Impact evaluation of SLM options to achieve land degradation neutrality: Dryland Systems interim report. - CGSpace
Opens in a new window

wocat.net
Sustainable Land Management (SLM) - WOCAT
Opens in a new window

documents1.worldbank.org
Sustainable Land Management Project I and II - World Bank Documents
Opens in a new window

arxiv.org
GeoPQA: Bridging the Visual Perception Gap in MLLMs for Geometric Reasoning - arXiv
Opens in a new window

wangywust.github.io
Towards Comprehensive Reasoning in Vision-Language Models | ICCV 2025 - Yiwei Wang
Opens in a new window

arxiv.org
rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking
Opens in a new window

mdpi.com
A Review on Visual-SLAM: Advancements from Geometric Modelling to Learning-Based Semantic Scene Understanding Using Multi-Modal Sensor Fusion - MDPI
Opens in a new window

github.com
LingDong-/skeleton-tracing: A new algorithm for retrieving ... - GitHub
Opens in a new window

github.com
Vahe1994/Deep-Vectorization-of-Technical-Drawings ... - GitHub
Opens in a new window

mdpi.com
Automatic Extrinsic Calibration of 3D LIDAR and Multi-Cameras Based on Graph Optimization - MDPI
Opens in a new window

attocube.com
Axis Calibration Software for CNC machines and CMMs - attocube
Opens in a new window

diva-portal.org
Statistical Sensor Calibration Algorithms - Diva-Portal.org
Opens in a new window

infragistics.com
Axes, Tick Marks, Tick Labels, and Grid Lines - Infragistics
Opens in a new window

quanthub.com
5 Tips for Axis Tick Marks in Chart Design - QuantHub
Opens in a new window

stackoverflow.com
Tickmark algorithm for a graph axis - Stack Overflow
Opens in a new window

stackoverflow.com
Convert Image Pixel Data to Coordinate Array - Stack Overflow
Opens in a new window

scichart.com
Axis APIs - Convert Pixel to Data Coordinates | JavaScript Chart Documentation - SciChart
Opens in a new window

mathworks.com
Convert image pixels to xy-coordinates - File Exchange - MATLAB Central - MathWorks
Opens in a new window

matplotlib.org
Transformations Tutorial — Matplotlib 3.10.8 documentation
Opens in a new window

spatialdata.scverse.org
Transformations and coordinate systems — spatialdata - scverse
Opens in a new window

reddit.com
Quick question: Is there any quick-ish way to plot a reverse logarithmic scale on the X-Axis on a chart? : r/excel - Reddit
Opens in a new window

behavioralpolicy.org
Graphs with logarithmic axes distort lay judgments - Behavioral Science & Policy Association
Opens in a new window

plotdigitizer.com
PlotDigitizer — Extract Data from Graph Image Online
Opens in a new window

docs.opencv.org
Detection of ArUco Markers - OpenCV Documentation
Opens in a new window

medium.com
Simple Lane Detection with OpenCV | by Matt Hardwick - Medium
Opens in a new window

stackoverflow.com
Detecting tick marks with python opencv - Stack Overflow
Opens in a new window

geeksforgeeks.org
Detecting ArUco markers with OpenCV and Python - GeeksforGeeks
Opens in a new window

en.wikipedia.org
Halide (programming language) - Wikipedia
Opens in a new window

docs.qualcomm.com
The Halide programming language - Qualcomm Docs
Opens in a new window

developer.arm.com
Halide: A novel programming paradigm for advanced image processing - Arm Developer
Opens in a new window

halide-lang.org
Halide lang
Opens in a new window

people.csail.mit.edu
Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines - People
Opens in a new window

pmc.ncbi.nlm.nih.gov
Hydra image processor: 5-D GPU image analysis library with MATLAB and python wrappers
Opens in a new window

developer.nvidia.com
CUDA-X GPU-Accelerated Libraries - NVIDIA Developer
Opens in a new window

github.com
tier4/accelerated_image_processor: A library to provide hardware acceleration for image processing functions such as compression and rectification. - GitHub
Opens in a new window

rapids.ai
Ecosystem | RAPIDS | RAPIDS | GPU Accelerated Data Science
Opens in a new window

reddit.com
PhotoFF a CUDA-accelerated image processing library : r/Python - Reddit
Opens in a new window

ieeexplore.ieee.org
FPGA Implementation for Skeletonization of 2-D Images - IEEE Xplore
Opens in a new window

mdpi.com
FPGA-Based Processor Acceleration for Image Processing Applications - MDPI
Opens in a new window

ieeexplore.ieee.org
Real-time skeletonization using FPGA - IEEE Xplore
Opens in a new window
Xác lập nền tảng hình học cho hệ thống Geo-SLM
Tôi đang phân tích và định hình lại phương pháp tiếp cận cốt lõi của hệ thống Geo-SLM, ưu tiên mô hình Neuro-Symbolic thay vì các mô hình ngôn ngữ thị giác thuần túy để đạt được độ chính xác học thuật tuyệt đối. Thay vì dựa vào các đường cong mềm mại, tôi tập trung vào việc mô hình hóa biểu đồ dưới dạng các thực thể hình học rời rạc như đa tuyến (polylines), các đỉnh (vertices) và các đoạn thẳng (segments). Cách tiếp cận này giúp loại bỏ hiện tượng ảo giác bằng cách chuyển dịch từ xử lý pixel sang các thực thể toán học có cấu trúc vững chắc.
Tối ưu hóa xử lý ảnh và bảo toàn cấu trúc topo
Tôi đang đi sâu vào việc nghiên cứu các thuật toán trích xuất khung xương và xử lý ảnh âm bản để tách biệt dữ liệu khỏi nhiễu nền và lưới tọa độ. Mục tiêu trọng tâm của tôi là đảm bảo tính liên kết của các chuỗi dữ liệu thông qua việc bảo toàn cấu trúc topo trong quá trình thu gọn nét vẽ. Đặc biệt, tôi đang xem xét các kỹ thuật nội suy để đạt được độ chính xác ở mức dưới pixel (sub-pixel), một yếu tố then chốt để xác định đúng vị trí các điểm mấu chốt khi các đường biểu đồ bị chồng lấn hoặc có ký hiệu đánh dấu phức tạp.
Thiết lập lộ trình đối chứng và thực thi kỹ thuật
Tôi đang chuẩn bị một lộ trình nghiên cứu chi tiết để so sánh hiệu năng giữa phương pháp vector hóa ký hiệu này với các mô hình học sâu hiện đại. Trong các bước tiếp theo, tôi sẽ tập trung tổng hợp các thuật toán xử lý tín hiệu hình ảnh cấp thấp và các phương pháp phân tách chuỗi dữ liệu dựa trên đặc trưng hình thái học như nét đứt, độ dày nét hoặc độ dốc liên tục. Điều này nhằm xây dựng một quy trình kỹ thuật hoàn chỉnh và định nghĩa sơ đồ dữ liệu JSON chuẩn hóa phục vụ cho khâu lập luận của mô hình ngôn ngữ nhỏ.

Researching websites...

mathoverflow.net
Algorithm for finding minimally overlapping paths in a graph - MathOverflow

stackoverflow.com
Determine overlapping lines (to remove them)? - Stack Overflow

reddit.com
Help removing some overlapping lines : r/PlotterArt - Reddit

computergraphics.stackexchange.com
Best technique to draw overlapping colored line segments that follow the same route

mikekling.com
Comparing Algorithms for Dispersing Overlapping Rectangles - Michael Kling

microsoft.com
dashed line detection - Microsoft

mdpi.com
A New Approach to Detect Hand-Drawn Dashed Lines in Engineering Sketches - MDPI

pmc.ncbi.nlm.nih.gov
ChartLine: Automatic Detection and Tracing of Curves in Scientific Line Charts Using Spatial-Sequence Feature Pyramid Network - NIH

open.clemson.edu
MORPHOLOGICAL CHARTS: A SYSTEMATIC EXPLORATION OF QUALITATIVE DESIGN SPACE - Clemson OPEN

stackoverflow.com
opencv detect dotted lines - python - Stack Overflow

repository.upenn.edu
GROUPING STRAIGHT LINE SEGMENTS IN REAL IMAGES - University of Pennsylvania

en.wikipedia.org
Bresenham's line algorithm - Wikipedia

mdpi.com
Research on Image Stitching Algorithm Based on Point-Line Consistency and Local Edge Feature Constraints - MDPI

encord.com
Guide to Image Segmentation in Computer Vision: Best Practices - Encord

medium.com
Clustering in Image Processing Explained | by Amit Yadav - Medium

arxiv.org
Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) - arXiv

nlpr.ia.ac.cn
An End-to-End Model for Question Answering over Knowledge Base with Cross-Attention Combining Global Knowledge

meilisearch.com
GraphRAG vs. Vector RAG: Side-by-side comparison guide - Meilisearch

neo4j.com
Knowledge Graph vs. Vector RAG: Optimization & Analysis - Neo4j

ritvik19.medium.com
Papers Explained 256: DePlot - Ritvik Rastogi

arxiv.org
Enhancing Question Answering on Charts Through Effective Pre-training Tasks - arXiv

aclanthology.org
SIMPLOT: Enhancing Chart Question Answering by Distilling Essentials - ACL Anthology

arxiv.org
ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering - arXiv

researchgate.net
MatCha: Enhancing Visual Language Pretraining with Math Reasoning and Chart Derendering | Request PDF - ResearchGate

openaccess.thecvf.com
VQualA 2025 Challenge on Image Super-Resolution Generated Content Quality Assessment - CVF Open Access

arxiv.org
Graph-Based Vector Search: An Experimental Evaluation of the State-of-the-Art - arXiv

digitalocean.com
Exploring SOTA: A Guide to Cutting-Edge AI Models | DigitalOcean

neurips.cc
NeurIPS 2024 Spotlight Posters

isamu-website.medium.com
Understanding Spectral Graph Theory and then the current SOTA of GNNs - Isamu Isozaki

microsoft.com
ChartOCR: Data Extraction from Charts Images via a Deep Hybrid Framework - Microsoft

plotdigitizer.com
PlotDigitizer — Extract Data from Graph Image Online

automeris.io
automeris.io: Computer vision assisted data extraction from charts using WebPlotDigitizer

stackoverflow.com
Plot digitization - scraping sample values from an image of a graph - Stack Overflow

youtube.com
ChartDetective: Easy and Accurate Interactive Data Extraction from Complex Vector Charts

en.wikipedia.org
Ramer–Douglas–Peucker algorithm - Wikipedia

pmc.ncbi.nlm.nih.gov
GPU-Accelerated RDP Algorithm for Data Segmentation - PMC - NIH

mdpi.com
A Vector Line Simplification Algorithm Based on the Douglas–Peucker Algorithm, Monotonic Chains and Dichotomy - MDPI

hpi.de
Analysis of the Parameter Configuration of the Ramer-Douglas-Peucker Algorithm for Time Series Compression

stackoverflow.com
line simplification algorithm: Visvalingam vs Douglas-Peucker - Stack Overflow

reddit.com
Simplify a polyline or polygon with Visvalingham-Whyatt or Douglas-Peucker : r/Python

en.wikipedia.org
Visvalingam–Whyatt algorithm - Wikipedia

martinfleischmann.net
Line simplification algorithms | Martin Fleischmann

bost.ocks.org
Line Simplification - Mike Bostock

papers.neurips.cc
Data Skeletonization via Reeb Graphs

inf.u-szeged.hu
Skeletonization

livrepository.liverpool.ac.uk
Skeletonisation Algorithms with Theoretical Guarantees for Unorganised Point Clouds with High Levels of Noise - The University of Liverpool Repository

www3.cs.stonybrook.edu
TEASAR: Tree-structure Extraction Algorithm for Accurate and Robust Skeletons - Stony Brook Computer Science

mdpi.com
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - MDPI

staff.fnwi.uva.nl
6.3. Subpixel Localization of Local Structure - Homepages of UvA/FNWI staff

cseweb.ucsd.edu
Subpixel Corner Detection for Tracking Applications using CMOS Camera Technology

docs.opencv.org
Harris Corner Detection - OpenCV Documentation

stackoverflow.com
how to improve corner detection accuracy - Stack Overflow

accedacris.ulpgc.es
Accurate Subpixel Edge Location based on Partial Area Effect - accedaCRIS
Lựa chọn thuật toán đơn giản hóa đa tuyến
Tôi đã phân tích và so sánh sâu hai kỹ thuật then chốt là Ramer-Douglas-Peucker và Visvalingam-Whyatt để tối ưu hóa việc biểu diễn đường gấp khúc. Trong khi phương pháp trước tập trung vào việc duy trì khoảng cách sai số, thuật toán Visvalingam-Whyatt tỏ ra ưu việt hơn trong việc bảo toàn đặc tính hình dạng của biểu đồ thông qua việc tính toán diện tích hiệu dụng. Điều này giúp tôi đảm bảo rằng các đặc trưng quan trọng của dữ liệu học thuật không bị mất đi trong quá trình giản lược hóa các thực thể vector.
Bảo toàn cấu trúc liên kết và độ chính xác cực cao
Tôi đang đánh giá việc ứng dụng đồ thị Reeb và các thuật toán như Teasar để giải quyết bài toán trích xuất khung xương mà không làm đứt gãy tính liên kết của dữ liệu. Qua quá trình tổng hợp, tôi xác nhận rằng việc kết hợp các phép khai triển Taylor với các bộ lọc điểm mấu chốt có thể đạt được độ chính xác ở mức dưới pixel (sub-pixel). Khả năng này cực kỳ quan trọng để định vị chính xác các điểm đỉnh và giao điểm trong những biểu đồ có mật độ dữ liệu cao hoặc độ phân giải hạn chế.
Xử lý điểm giao cắt và tối ưu hóa quy trình âm bản
Tôi đang tập trung giải quyết thách thức tại các điểm giao cắt của nhiều đường biểu đồ trong ảnh đơn sắc, nơi các phương pháp lọc truyền thống thường gặp lỗi. Tôi sẽ nghiên cứu các thuật toán truy vết dựa trên tính liên tục của góc lệch và kiểm tra các chuỗi lọc hình thái học trên ảnh âm bản để tách biệt dữ liệu khỏi lưới tọa độ một cách triệt để. Cuối cùng, tôi sẽ xây dựng một cấu trúc dữ liệu chuẩn hóa để lưu trữ các thực thể toán học này, tạo tiền đề cho khâu lập luận logic phía sau.

Researching websites...

arxiv.org
Deep Learning vs. Traditional Computer Vision - arXiv

mdpi.com
An Accuracy vs. Complexity Comparison of Deep Learning Architectures for the Detection of COVID-19 Disease - MDPI

geeksforgeeks.org
Difference between Traditional Computer Vision Techniques and Deep Learning-based Approaches - GeeksforGeeks

pmc.ncbi.nlm.nih.gov
Comparison between Deep Learning and Conventional Machine Learning in Classifying Iliofemoral Deep Venous Thrombosis upon CT Venography - NIH

mdpi.com
Optimization Methods, Challenges, and Opportunities for Edge Inference: A Comprehensive Survey - MDPI

chemrxiv.org
Revolutionizing Scientific Figure Decoding: Benchmarking LLM Data Extraction Performance - ChemRxiv

developer.nvidia.com
Query Graphs with Optimized DePlot Model | NVIDIA Technical Blog

medium.com
Hands on DePlot - Medium

arxiv.org
Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction - arXiv

arxiv.org
[1809.02697] Optimizing CNN Model Inference on CPUs - arXiv

developer.arm.com
Getting real-time CNN inference for AR on mobile - Arm Developer

opus4.kobv.de
Performance analysis of CNN speed and power consumption among CPUs, GPUs and an FPGA for vehicular applications - OPUS

researchgate.net
A Real-Time and Hardware-Efficient Processor for Skeleton-Based

usenix.org
Optimizing CNN Model Inference on CPUs - USENIX

ncbi.nlm.nih.gov
Working with JSON Lines data reports - NCBI - NIH

docs.anychart.com
Data From JSON | Working with Data - AnyChart Documentation

json-schema.org
JSON Schema examples

helm.sh
Charts - Helm

docs.datadoghq.com
Graphing with JSON - Datadog Docs

vega.github.io
Line Chart Example - Vega-Lite

vega.github.io
Line | Vega-Lite

vega.github.io
Bar and Line Chart - Vega-Lite

vega.github.io
Specification - Vega-Lite

vega.github.io
Example Gallery | Vega-Lite

plotdigitizer.com
How to Extract Data from Graphs or Images in Scientific Papers? - PlotDigitizer

idl.cs.washington.edu
Reverse-Engineering Visualizations: Recovering Visual Encodings from Chart Images - UW Interactive Data Lab

archives.gov
Appendix A: Tables of File Formats - National Archives

apriorit.com
How to Reverse Engineer a Proprietary File Format: A Brief Guide with Practical Examples

jamie-wong.com
Reverse Engineering Instruments' File Format - Jamie Wong

docs.ogc.org
Features and geometry – Part 1: Feature models

arxiv.org
Geometric Deep Learning for Computer-Aided Design: A Survey - arXiv

dev.opencascade.org
Modeling Data - Open CASCADE Technology

dataroots.io
A gentle introduction to Geometric Deep Learning - dataroots

smartdraw.com
Flowchart Symbols - SmartDraw

tirthajyoti.github.io
22. Skeletonizing an image — Hands-on introduction to Scikit-image methods

openaccess.thecvf.com
A skeletonization algorithm for gradient-based optimization - CVF Open Access

ipol.im
Finding the Skeleton of 2D Shape and Contours: Implementation of Hamilton-Jacobi Skeleton - IPOL Journal

mdpi.com
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - MDPI

scikit-image.org
Skeletonize — skimage 0.25.2 documentation

imagej.net
MorphoLibJ - ImageJ Wiki

nv5geospatialsoftware.com
Convolution and Morphology Filters - NV5 Geospatial Software

mathworks.com
Morphological Operations - MATLAB & Simulink - MathWorks

medium.com
Image Processing Class #6 — Morphological Filter | by Pitchaya Thipkham - Medium

scikit-image.org
skimage.morphology — skimage 0.26.0 documentation

nanpa.org
The Positive Traits of Negative Space - NANPA

adam.edu.sg
Why Negative Space Makes Your Photographs Look Better

homepages.inf.ed.ac.uk
Line Detection

en.wikipedia.org
Edge detection - Wikipedia

mdpi.com
Negative Samples for Improving Object Detection—A Case Study in AI-Assisted Colonoscopy for Polyp Detection - MDPI

mdpi.com
A Single Image Enhancement Technique Using Dark Channel Prior - MDPI

scikit-image.org
Use rolling-ball algorithm for estimating background intensity - Scikit-image

cs.stanford.edu
Darkroom: Compiling High-Level Image Processing Code into Hardware Pipelines - Stanford Computer Science

dsp.stackexchange.com
Segmentation of Dark Blobs with Varying Background - Signal Processing Stack Exchange

stackoverflow.com
An algorithm for selecting a dark color similar to a light color - Stack Overflow

github.com
LingDong-/skeleton-tracing: A new algorithm for retrieving topological skeleton as a set of polylines from binary images - GitHub

hilandtom.com
Quantitative analysis of skeletonisation algorithms for modelling of branches

pmc.ncbi.nlm.nih.gov
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - PMC - NIH

papers.neurips.cc
Data Skeletonization via Reeb Graphs

stackoverflow.com
Finding intersections of a skeletonised image in python opencv - Stack Overflow

stackoverflow.com
Determine overlapping lines (to remove them)? - Stack Overflow

computergraphics.stackexchange.com
Best technique to draw overlapping colored line segments that follow the same route

community.adobe.com
Image trace overlapping trace lines twice (Only just started doing this)

reddit.com
How would I go about making overlapping lines easily like this? : r/AdobeIllustrator - Reddit

youtube.com
Image Trace Abutting Versus Overlapping Paths - YouTube

www-sop.inria.fr
Fidelity vs. Simplicity: a Global Approach to Line Drawing Vectorization - Inria

media.disneyanimation.com
Topology-Driven Vectorization of Clean Line Drawings - Walt Disney Animation Studios

researchgate.net
(PDF) Vectorization of line drawing image based on junction analysis - ResearchGate

news.mit.edu
Smoothing out sketches' rough edges | MIT News | Massachusetts Institute of Technology

cecilialeiqi.github.io
Vectorization of line drawing image based on junction analysis - Qi Lei

researchgate.net
(PDF) Path Openings and Closings - ResearchGate

theory.stanford.edu
Heuristics - Stanford CS Theory

pages.cs.wisc.edu
Image Alignment and Stitching: A Tutorial - cs.wisc.edu

en.wikipedia.org
Hough transform - Wikipedia

theclassytim.medium.com
Using Image Processed Hough Lines for Path Planning Applications | by Tim Chinenov

ri.cmu.edu
Kalman Filter-based Algorithms for Estimating Depth from Image Sequences - Carnegie Mellon University's Robotics Institute

en.wikipedia.org
Kalman filter - Wikipedia

arxiv.org
An Analysis of Kalman Filter based Object Tracking Methods for Fast-Moving Tiny Objects

bzarg.com
How a Kalman filter works, in pictures - Bzarg

engineering.purdue.edu
A New Kalman-Filter-Based Framework for Fast and Accurate Visual Tracking of Rigid Objects - Purdue College of Engineering
Hiệu quả của mô hình lai và bảo toàn hình học
Tôi đã xác nhận tính hiệu quả của mô hình lai — kết hợp nhận diện bằng học sâu với trích xuất hình học — thông qua các nghiên cứu hàng đầu như ChartOCR. Việc phân tích sâu cho thấy mặc dù thuật toán Ramer-Douglas-Peucker rất phổ biến, nhưng phương pháp Visvalingam-Whyatt lại tỏ ra ưu việt hơn trong việc bảo toàn các đặc trưng hình thái quan trọng của biểu đồ khoa học. Tôi nhận thấy rằng việc sử dụng diện tích hiệu dụng thay vì chỉ dựa vào khoảng cách pixel giúp duy trì được những biến động tinh tế trong dữ liệu mà không làm tăng độ phức tạp của thực thể vector.
Giải quyết điểm giao cắt và độ chính xác dưới pixel
Tôi đang tập trung xử lý thách thức kỹ thuật lớn nhất là phân tách các đường biểu đồ chồng lấn trong ảnh đơn sắc. Qua việc tổng hợp các kỹ thuật phát hiện góc Harris và tinh chỉnh ở mức dưới pixel (sub-pixel), tôi đã tìm ra cách định vị các điểm mấu chốt với độ chính xác vượt xa độ phân giải gốc của ảnh. Điều này cho phép tôi bắt đầu xây dựng các quy tắc lập luận dựa trên tính liên tục của góc lệch để tự động 'gỡ rối' các chuỗi dữ liệu tại những điểm giao cắt phức tạp, nơi các mô hình truyền thống thường bị nhầm lẫn.
Chuẩn hóa dữ liệu và hoàn thiện quy trình logic
Trong giai đoạn tới, tôi sẽ tập trung xây dựng một cấu trúc JSON chuẩn hóa dựa trên các tiêu chuẩn như Vega-Lite để lưu trữ các thực thể toán học này một cách bền vững. Tôi sẽ nghiên cứu sâu hơn về các thuật toán truy vết đường dẫn (path tracing) và kiểm chứng giả thuyết về việc sử dụng ảnh âm bản để tăng cường hiệu quả cho các bộ lọc hình thái học. Mục tiêu cuối cùng là hoàn thiện một quy trình kỹ thuật khép kín, đảm bảo tính nhất quán của dữ liệu từ khi còn là pixel cho đến khi trở thành các thực thể logic phục vụ cho khâu lập luận.

Researching websites...

arxiv.org
Deep Learning vs. Traditional Computer Vision - arXiv

mdpi.com
An Accuracy vs. Complexity Comparison of Deep Learning Architectures for the Detection of COVID-19 Disease - MDPI

researchgate.net
Complexity and computational cost comparison | Download Table - ResearchGate

pmc.ncbi.nlm.nih.gov
Comparison between Deep Learning and Conventional Machine Learning in Classifying Iliofemoral Deep Venous Thrombosis upon CT Venography - NIH

mdpi.com
Optimization Methods, Challenges, and Opportunities for Edge Inference: A Comprehensive Survey - MDPI

chemrxiv.org
Revolutionizing Scientific Figure Decoding: Benchmarking LLM Data Extraction Performance - ChemRxiv

developer.nvidia.com
Query Graphs with Optimized DePlot Model | NVIDIA Technical Blog

medium.com
Hands on DePlot - Medium

arxiv.org
Document Parsing Unveiled: Techniques, Challenges, and Prospects for Structured Information Extraction - arXiv

arxiv.org
[1809.02697] Optimizing CNN Model Inference on CPUs - arXiv

developer.arm.com
Getting real-time CNN inference for AR on mobile - Arm Developer

opus4.kobv.de
Performance analysis of CNN speed and power consumption among CPUs, GPUs and an FPGA for vehicular applications - OPUS

researchgate.net
A Real-Time and Hardware-Efficient Processor for Skeleton-Based

usenix.org
Optimizing CNN Model Inference on CPUs - USENIX

docs.anychart.com
Data From JSON | Working with Data - AnyChart Documentation

json-schema.org
JSON Schema examples

json-schema.org
Miscellaneous Examples - JSON Schema

helm.sh
Charts - Helm

docs.datadoghq.com
Graphing with JSON - Datadog Docs

vega.github.io
Line | Vega-Lite

vega.github.io
Line Chart Example - Vega-Lite

vega.github.io
Data - Vega-Lite

vega.github.io
Bar and Line Chart - Vega-Lite

vega.github.io
Example Gallery | Vega-Lite

plotdigitizer.com
How to Extract Data from Graphs or Images in Scientific Papers? - PlotDigitizer

idl.cs.washington.edu
Reverse-Engineering Visualizations: Recovering Visual Encodings from Chart Images - UW Interactive Data Lab

archives.gov
Appendix A: Tables of File Formats - National Archives

apriorit.com
How to Reverse Engineer a Proprietary File Format: A Brief Guide with Practical Examples

jamie-wong.com
Reverse Engineering Instruments' File Format - Jamie Wong

docs.ogc.org
Features and geometry – Part 1: Feature models

arxiv.org
Geometric Deep Learning for Computer-Aided Design: A Survey - arXiv

dev.opencascade.org
Modeling Data - Open CASCADE Technology

dataroots.io
A gentle introduction to Geometric Deep Learning - dataroots

smartdraw.com
Flowchart Symbols - SmartDraw

tirthajyoti.github.io
22. Skeletonizing an image — Hands-on introduction to Scikit-image methods

openaccess.thecvf.com
A skeletonization algorithm for gradient-based optimization - CVF Open Access

ipol.im
Finding the Skeleton of 2D Shape and Contours: Implementation of Hamilton-Jacobi Skeleton - IPOL Journal

mdpi.com
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - MDPI

scikit-image.org
Skeletonize — skimage 0.25.2 documentation

imagej.net
MorphoLibJ - ImageJ Wiki

nv5geospatialsoftware.com
Convolution and Morphology Filters - NV5 Geospatial Software

scikit-image.org
Morphological Filtering — skimage 0.25.2 documentation

mathworks.com
Morphological Operations - MATLAB & Simulink - MathWorks

medium.com
Image Processing Class #6 — Morphological Filter | by Pitchaya Thipkham - Medium

nanpa.org
The Positive Traits of Negative Space - NANPA

adam.edu.sg
Why Negative Space Makes Your Photographs Look Better

homepages.inf.ed.ac.uk
Line Detection

en.wikipedia.org
Edge detection - Wikipedia

mdpi.com
Negative Samples for Improving Object Detection—A Case Study in AI-Assisted Colonoscopy for Polyp Detection - MDPI

pmc.ncbi.nlm.nih.gov
A Two-Stage Automatic Color Thresholding Technique - PMC - PubMed Central

scikit-image.org
Use rolling-ball algorithm for estimating background intensity - Scikit-image

mdpi.com
A Single Image Enhancement Technique Using Dark Channel Prior - MDPI

dsp.stackexchange.com
Segmentation of Dark Blobs with Varying Background - Signal Processing Stack Exchange

stackoverflow.com
An algorithm for selecting a dark color similar to a light color - Stack Overflow

github.com
LingDong-/skeleton-tracing: A new algorithm for retrieving topological skeleton as a set of polylines from binary images - GitHub

hilandtom.com
Quantitative analysis of skeletonisation algorithms for modelling of branches

pmc.ncbi.nlm.nih.gov
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - PMC - NIH

papers.neurips.cc
Data Skeletonization via Reeb Graphs

stackoverflow.com
Finding intersections of a skeletonised image in python opencv - Stack Overflow

stackoverflow.com
Determine overlapping lines (to remove them)? - Stack Overflow

computergraphics.stackexchange.com
Best technique to draw overlapping colored line segments that follow the same route

community.adobe.com
Image trace overlapping trace lines twice (Only just started doing this)

reddit.com
How would I go about making overlapping lines easily like this? : r/AdobeIllustrator - Reddit

youtube.com
Image Trace Abutting Versus Overlapping Paths - YouTube

www-sop.inria.fr
Fidelity vs. Simplicity: a Global Approach to Line Drawing Vectorization - Inria

media.disneyanimation.com
Topology-Driven Vectorization of Clean Line Drawings - Walt Disney Animation Studios

researchgate.net
(PDF) Vectorization of line drawing image based on junction analysis - ResearchGate

news.mit.edu
Smoothing out sketches' rough edges | MIT News | Massachusetts Institute of Technology

cecilialeiqi.github.io
Vectorization of line drawing image based on junction analysis - Qi Lei

researchgate.net
(PDF) Path Openings and Closings - ResearchGate

theory.stanford.edu
Heuristics - Stanford CS Theory

en.wikipedia.org
Hough transform - Wikipedia

theclassytim.medium.com
Using Image Processed Hough Lines for Path Planning Applications | by Tim Chinenov

pages.cvc.uab.es
A Graph based Approach for Segmenting Touching Lines in

ri.cmu.edu
Kalman Filter-based Algorithms for Estimating Depth from Image Sequences - Carnegie Mellon University's Robotics Institute

en.wikipedia.org
Kalman filter - Wikipedia

arxiv.org
An Analysis of Kalman Filter based Object Tracking Methods for Fast-Moving Tiny Objects

bzarg.com
How a Kalman filter works, in pictures - Bzarg

engineering.purdue.edu
A New Kalman-Filter-Based Framework for Fast and Accurate Visual Tracking of Rigid Objects - Purdue College of Engineering
Giải quyết bài toán gỡ rối các đường biểu đồ giao cắt
Tôi đang nghiên cứu sâu về kỹ thuật 'trường khung' (frame fields) để giải quyết hiện tượng chồng lấn dữ liệu tại các điểm giao cắt phức tạp. Thay vì chỉ truy vết pixel đơn thuần, tôi đang tư duy theo hướng sử dụng các chỉ dấu định hướng để duy trì tính liên tục của các đoạn thẳng ngay cả khi chúng đè lên nhau. Phương pháp này giúp tôi xử lý triệt để sự mơ hồ tại các nút giao, đảm bảo mỗi chuỗi dữ liệu đơn sắc được phân tách độc lập mà không bị nhầm lẫn với các thành phần khác.
Tối ưu hóa khung xương thông qua xử lý ảnh âm bản
Tôi đang xác thực giả thuyết về ưu thế của việc sử dụng ảnh âm bản trong việc bảo toàn cấu trúc liên kết của biểu đồ. Qua việc phân tích các thuật toán thu gọn khung xương, tôi nhận thấy xử lý trên nền tối giúp duy trì tính toàn vẹn của cấu trúc topo tốt hơn, đặc biệt là đối với các nét vẽ mảnh hoặc nét đứt. Việc kết hợp các bộ lọc hình thái học trên không gian ảnh đảo ngược này đang tạo ra một 'bản đồ xương' chính xác, giúp khâu chuyển đổi sang thực thể vector trở nên trơn tru và ít lỗi hơn.
Đối chứng hiệu năng và chuẩn hóa cấu trúc dữ liệu
Tôi đang thực hiện bước đối chứng cuối cùng về hiệu suất tính toán giữa các thuật toán thị giác máy tính truyền thống và các mô hình học sâu hiện đại. Việc này nhằm khẳng định khả năng vận hành mượt mà của hệ thống trên các nền tảng phần cứng hạn chế mà vẫn giữ được độ chính xác tuyệt đối. Trong bước kế tiếp, tôi sẽ tập trung hoàn thiện sơ đồ JSON chuẩn hóa dựa trên các mô hình như Vega-Lite để đóng gói các thực thể toán học đã trích xuất, tạo cầu nối dữ liệu hoàn hảo cho khâu lập luận logic phía sau.

Researching websites...

newline.co
Convolutional Neural Networks vs OpenCV: Performance Comparison in Computer Vision AI - Newline.co

mdpi.com
Comparison of CNN-Based Architectures for Detection of Different Object Classes - MDPI

researchgate.net
Comparison of CNN-Based Architectures for Detection of Different Object Classes

technolynx.com
Deep Learning vs. Traditional Computer Vision Methods - TechnoLynx

cs.swarthmore.edu
Comparing Deep Neural Networks and Traditional Vision Algorithms in Mobile Robotics - Swarthmore College

arxiv.org
Deep Learning vs. Traditional Computer Vision - arXiv

mdpi.com
Deep Learning vs. Machine Learning for Intrusion Detection in Computer Networks: A Comparative Study - MDPI

vbn.aau.dk
Explorative Comparison Between Classic Computer Vision Techniques and Deep Learning

mdpi.com
Edge Intelligence: A Review of Deep Neural Network Inference in Resource-Limited Environments - MDPI

arxiv.org
A Study on Inference Latency for Vision Transformers on Mobile Devices - arXiv

mdpi.com
Impact of Thermal Throttling on Long-Term Visual Inference in a CPU-Based Edge Device

medium.com
Optimizing Resnet-50: 8X inference throughput with just a few commands - Medium

discuss.pytorch.org
What is a normal inference speed for Resnet34 and how to improve it? - PyTorch Forums

pmc.ncbi.nlm.nih.gov
Research on Image Stitching Algorithm Based on Point-Line Consistency and Local Edge Feature Constraints - NIH

idl.cs.washington.edu
Fast and Flexible Overlap Detection for Chart Labeling with Occupancy Bitmap - UW Interactive Data Lab

mathoverflow.net
Algorithm for finding minimally overlapping paths in a graph - MathOverflow

stackoverflow.com
Algorithm to detect overlapping rows of two images - Stack Overflow

medium.com
Overlapping lines. Daily Coding Problem n. 27 | by Nicola Moro | Medium

researchgate.net
(PDF) Vectorization of line drawing image based on junction analysis - ResearchGate

media.disneyanimation.com
Topology-Driven Vectorization of Clean Line Drawings - Walt Disney Animation Studios

cgl.ethz.ch
Semantic Segmentation for Line Drawing Vectorization Using Neural Networks - Computer Graphics Laboratory

news.mit.edu
Smoothing out sketches' rough edges | MIT News | Massachusetts Institute of Technology

igl.ethz.ch
Cusps of Characteristic Curves and Intersection-Aware Visualization of Path and Streak Lines - Interactive Geometry Lab

apps.dtic.mil
Curve and Polygon Evolution Techniques for Image Processing - DTIC

phtu-cs.github.io
Continuous Curve Textures - GitHub Pages

stackoverflow.com
Separate crossings segments in binarised image - Stack Overflow

en.wikipedia.org
Edge detection - Wikipedia

igl.ethz.ch
Single-Line Drawing Vectorization

cecilialeiqi.github.io
Vectorization of line drawing image based on junction analysis - Qi Lei

erickjb.com
Singularity-Free Frame Fields for Line Drawing Vectorization - Erick Jimenez Berumen

arxiv.org
[1801.01922] Vectorization of Line Drawings via PolyVector Fields - arXiv

cs-people.bu.edu
Singularity-Free Frame Fields - for Line Drawing Vectorization - Boston University

atlassian.com
How to choose colors for data visualizations - Atlassian

developers.google.com
Visualization: Combo Chart - Google for Developers

stackoverflow.com
Changing Dash chart series colors - python - Stack Overflow

plotly.com
Bar charts in Python - Plotly

datylon.com
80 types of charts & graphs for data visualization (with examples) - Datylon

json-schema.org
Creating your first schema - JSON Schema

developers.arcgis.com
Charts Interfaces | Overview | ArcGIS Maps SDK for JavaScript - Esri Developer

stackoverflow.com
Representing a graph in JSON - Stack Overflow

json-schema.org
JSON Schema examples

helm.sh
Charts - Helm

vega.github.io
Multi Series Line Chart | Vega-Lite

vega.github.io
Multi Series Line Chart with an Interactive Line Highlight - Vega-Lite

vega.github.io
Multi Series Line Chart with Labels | Vega-Lite

vega.github.io
Multi Series Line Chart with Tooltip | Vega-Lite

vega.github.io
Line | Vega-Lite

chartjs.org
Line Chart - Chart.js

chartjs.org
Data structures - Chart.js

chartjs.org
Step-by-step guide - Chart.js

stackoverflow.com
Drawing line chart in chart.js with json data - Stack Overflow

youtube.com
Add JSON data to ChartJS - YouTube

plotdigitizer.com
Documentation - PlotDigitizer

plotdigitizer.com
PlotDigitizer — Extract Data from Graph Image Online

plotdigitizer.com
Export Extracted Data to CSV, JSON, and Other File Formats - PlotDigitizer

automeris.io
Digitize Charts - WebPlotDigitizer

av8rdas.com
Plot Digitizer

pmc.ncbi.nlm.nih.gov
Efficient Skeletonization of Volumetric Objects - PMC - NIH

mdpi.com
Techniques and Algorithms for Hepatic Vessel Skeletonization in Medical Images: A Survey

arxiv.org
Does the Skeleton-Recall Loss Really Work? - arXiv

researchgate.net
A Survey on Skeletons in Digital Image Processing - ResearchGate

pmc.ncbi.nlm.nih.gov
Edge-Supervised Linear Object Skeletonization for High-Speed Camera - PMC - NIH

pmc.ncbi.nlm.nih.gov
Real-Time Thinning Algorithms for 2D and 3D Images using GPU processors - PMC

bioimagebook.github.io
Morphological operations - Introduction to Bioimage Analysis

graphics.ics.uci.edu
Morphological Image Processing

geeksforgeeks.org
Different Morphological Operations in Image Processing - GeeksforGeeks

reddit.com
Thinning algorithm : r/compsci - Reddit

scikit-image.org
Skeletonize — skimage 0.25.2 documentation

openaccess.thecvf.com
A skeletonization algorithm for gradient-based optimization - CVF Open Access

pmc.ncbi.nlm.nih.gov
A topology-preserving approach to the segmentation of brain images with multiple sclerosis lesions - PMC - PubMed Central

ipol.im
Finding the Skeleton of 2D Shape and Contours: Implementation of Hamilton-Jacobi Skeleton - IPOL Journal

arxiv.org
Topology-Preserving Downsampling of Binary Images - arXiv
Ứng dụng trường khung không kỳ dị trong vector hóa
Tôi đã phát hiện ra một hướng tiếp cận tiên tiến bằng cách sử dụng 'trường khung không kỳ dị' (singularity-free frame fields) để xử lý các biểu đồ có mật độ dữ liệu dày đặc. Thay vì chỉ dựa vào việc truy vết pixel đơn thuần, kỹ thuật này cho phép tôi nắm bắt cấu trúc định hướng tổng thể của các đường biểu đồ, giúp phân tách chính xác các đối tượng ngay cả khi chúng giao cắt hoặc chồng lấp phức tạp. Đây là yếu tố then chốt để chuyển đổi hình ảnh thành các thực thể toán học mà không làm mất đi tính toàn vẹn của dữ liệu gốc.
Nâng cấp mô hình dữ liệu lên đồ thị ngữ cảnh hình học
Tôi đang chuyển dịch tư duy từ việc chỉ trích xuất các tọa độ điểm thô sang việc xây dựng một 'đồ thị ngữ cảnh' (scene graph) cho biểu đồ. Tôi nhận thấy rằng để mô hình ngôn ngữ nhỏ có thể lập luận logic hiệu quả, cấu trúc JSON cần phải chứa đựng các mối quan hệ hình học trừu tượng như tính song song, điểm uốn và các ràng buộc về độ dốc liên tục. Việc mã hóa các 'thực thể toán học' kèm theo ngữ cảnh không gian này sẽ tạo ra một lớp đệm tri thức vững chắc, giúp hệ thống hiểu sâu bản chất biểu đồ thay vì chỉ đọc số liệu.
Tối ưu hóa hạ tầng thực thi và chuẩn hóa quy trình
Trong giai đoạn tới, tôi sẽ tập trung tuyển chọn các thư viện xử lý tín hiệu hình ảnh cấp thấp có hiệu suất tối ưu trên CPU, đặc biệt là các giải pháp dựa trên ngôn ngữ C++ và Rust để đảm bảo tốc độ xử lý vượt trội. Tôi sẽ tiến hành thiết kế chi tiết sơ đồ JSON chuẩn hóa để đóng gói toàn diện các thuộc tính hình thái như nét đứt và các điểm mốc dữ liệu. Đồng thời, tôi sẽ thực hiện rà soát chuyên sâu để đảm bảo mọi thành phần kỹ thuật được đề xuất đều hoàn toàn tương thích và tối ưu hóa riêng cho hệ thống phân tích Geo-SLM.

Researching websites...

arxiv.org
[1804.01622] Image Generation from Scene Graphs - arXiv

openaccess.thecvf.com
Unconditional Scene Graph Generation - CVF Open Access

ojs.aaai.org
Scene Graph-Grounded Image Generation | Proceedings of the AAAI Conference on Artificial Intelligence

github.com
ChocoWu/Awesome-Scene-Graph-Generation - GitHub

arxiv.org
Real-Time Scene Graph Generation - arXiv

pmc.ncbi.nlm.nih.gov
Zero-shot visual reasoning through probabilistic analogical mapping - PubMed Central

arxiv.org
Visually Descriptive Language Model for Vector Graphics Reasoning - arXiv

aclanthology.org
ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild - ACL Anthology

visualsketchpad.github.io
Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models

arxiv.org
Text-Based Reasoning About Vector Graphics - arXiv

arxiv.org
Navigate Complex Physical Worlds via Geometrically Constrained LLM - arXiv

arxiv.org
Do Large Language Models Truly Understand Geometric Structures? - arXiv

people.umass.edu
From Text to Visuals: Using LLMs to Generate Math Diagrams with Vector Graphics

medium.com
When LLMs Meet Geometry. Testing Gemini 3 on a PhD-level shape… | by Yogesh Haribhau Kulkarni (PhD) | Google Developer Experts | Nov, 2025 | Medium

aclanthology.org
Navigate Complex Physical Worlds via Geometrically Constrained LLM - ACL Anthology

pmc.ncbi.nlm.nih.gov
VESCL: an open source 2D vessel contouring library - PMC - PubMed Central

slicer.org
Modules:VMTKCenterlines - Slicer Wiki

fast-imaging.github.io
fast::CenterlineExtraction class - FAST | Documentation

github.com
An interactive software to extract vessel centerline - GitHub

stackoverflow.com
extract points that describe lines in a drawing - c++ - Stack Overflow

docs.rs
skeletonize - Rust - Docs.rs

crates.io
skeletonize - crates.io: Rust Package Registry

github.com
LingDong-/skeleton-tracing: A new algorithm for retrieving topological skeleton as a set of polylines from binary images - GitHub

lib.rs
Images — list of Rust libraries/crates // Lib.rs

users.rust-lang.org
Looking for image processing crate recommendation (& introduction) - Rust Users Forum

blend2d.com
Blend2D

intel.com
Rendering Libraries from Intel

github.com
Vahe1994/Deep-Vectorization-of-Technical-Drawings: Official Pytorch repository for Deep Vectorization of Technical Drawings https://arxiv.org/abs/2003.05471 - GitHub

ascelibrary.org
VectorGraphNET: Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings | Journal of Computing in Civil Engineering | Vol 39, No 6 - ASCE Library

artonson.github.io
Deep Vectorization of Technical Drawings - Alexey Artemov
Phân tách chuỗi dữ liệu dựa trên đặc trưng hình thái
Tôi đang nghiên cứu sâu các thuật toán nhóm các đoạn thẳng rời rạc thành một chuỗi dữ liệu thống nhất dựa trên tri giác hình thái. Thay vì dựa vào màu sắc, tôi tập trung vào việc thiết lập các tiêu chí về tính liên tục của độ dốc và khoảng cách để tự động nhận diện và liên kết các kiểu nét đứt hoặc nét gạch chấm. Cách tiếp cận này cho phép tôi tái cấu trúc chính xác các thực thể toán học ngay cả khi chúng bị đứt quãng hoặc bị nhiễu bởi các yếu tố đồ họa khác.
Tuyển chọn hạ tầng thư viện xử lý hiệu năng cao
Tôi đang tiến hành rà soát và đánh giá các thư viện xử lý tín hiệu hình ảnh hiện đại dựa trên ngôn ngữ C++ và Rust để thay thế cho các công cụ truy vết centerline truyền thống. Tôi muốn đảm bảo rằng hệ thống có khả năng trích xuất khung xương với hiệu suất tối ưu trên CPU, đạt độ chính xác ở mức dưới pixel mà không làm tăng độ trễ tính toán. Điều này cực kỳ quan trọng để duy trì tính ổn định của quy trình xử lý ảnh âm bản trong các điều kiện độ phân giải khác nhau.
Hoàn thiện công thức gỡ rối và sơ đồ lập luận logic
Trong giai đoạn tới, tôi sẽ tổng hợp các bộ quy tắc kỹ thuật chi tiết để giải quyết sự mơ hồ tại các điểm giao cắt, sử dụng các tiêu chí như giảm thiểu độ lệch góc và bảo toàn cấu trúc topo. Đồng thời, tôi sẽ bắt đầu thiết kế cấu trúc JSON chuẩn hóa dành riêng cho hệ thống phân tích này, nhằm đóng gói toàn bộ các mối quan hệ hình học và thuộc tính của thực thể vector thành một lớp tri thức có cấu trúc, giúp mô hình ngôn ngữ nhỏ có thể lập luận logic một cách chính xác nhất.

Researching websites...

github.com
homer-rain/Centerline-Extraction-of-Coronary-Vessels - GitHub

reddit.com
Must-know libraries/frameworks/technologies for C++ developer as of 2025 : r/cpp - Reddit

discourse.slicer.org
how to improve the CenterLine extraction - Support - 3D Slicer Community

researchgate.net
The Robust Vessel Segmentation and Centerline Extraction: One-Stage Deep Learning Approach - ResearchGate

youtube.com
Starting my modern C++ Project with CMake in 2024 - Jens Weller - YouTube

docs.rs
skeletonize - Rust - Docs.rs

crates.io
skeletonize - crates.io: Rust Package Registry

lib.rs
Images — list of Rust libraries/crates // Lib.rs

users.rust-lang.org
Looking for image processing crate recommendation (& introduction) - Rust Users Forum

astcad.com.au
Raster To Vector Conversion: Convert Paper Drawings To Accurate Vector Files - Australian Design & Drafting Services

impactdigitizing.com
How to Convert Raster to Vector - Impact Digitizing

reddit.com
Seeking source to convert sketch to vector paths : r/PlotterArt - Reddit

graphicdesign.stackexchange.com
Convert a line drawing from raster to vector **LINES** - Graphic Design Stack Exchange

github.com
visioncortex/vtracer: Raster to Vector Graphics Converter - GitHub

reddit.com
PoTrace: convert bitmaps to vector graphics : r/programming - Reddit

medium.com
Manual Vectorization Vs. Auto-Tracing: Understanding the Key Differences | by Cre8iveSkill

en.wikipedia.org
Comparison of raster-to-vector conversion software - Wikipedia

news.ycombinator.com
Is this any better than POTrace(http://potrace.sourceforge.net/)? | Hacker News

arxiv.org
StarVector: Generating Scalable Vector Graphics Code from Images and Text - arXiv

microsoft.com
dashed line detection - Microsoft

openaccess.thecvf.com
Neural Recognition of Dashed Curves With Gestalt Law of Continuity - CVF Open Access

pmc.ncbi.nlm.nih.gov
ChartLine: Automatic Detection and Tracing of Curves in Scientific Line Charts Using Spatial-Sequence Feature Pyramid Network - NIH

wiredcraft.com
Dashed Line Segmentation in D3.js - Wiredcraft

stackoverflow.com
Why are there dashed lines in segments? - Stack Overflow

journals.plos.org
Disambiguating Multi–Modal Scene Representations Using Perceptual Grouping Constraints | PLOS One - Research journals

mrt.kit.edu
Application of Line Clustering Algorithms for Improving Road Feature Detection - KIT - MRT

qugank.github.io
Making Better Use of Edges via Perceptual Grouping - Yonggang Qi

pure.rug.nl
University of Groningen Algorithm that mimics human perceptual

pmc.ncbi.nlm.nih.gov
A computational model for gestalt proximity principle on dot patterns and beyond - NIH

stackoverflow.com
opencv detect dotted lines - python - Stack Overflow

medium.com
OpenCV Line Detection | by Amit Yadav - Medium

stackoverflow.com
Is there a method to use opencv-python to detect dashed cross lines? - Stack Overflow

stackoverflow.com
Detect dotted (broken) lines only in an image using OpenCV - Stack Overflow

answers.opencv.org
How to convert dashed lines to solid? - OpenCV Q&A Forum

mcgill.ca
Sectioning Technique | Engineering Design - McGill University

forums.sketchup.com
Joining Line Segments - Pro - SketchUp Community

computergraphics.stackexchange.com
Rounding the edges in a mitered line segment inside of a fragment shader

youtube.com
Solidworks Quick Tips - Merging Line Segments - Collinear - YouTube

forums.autodesk.com
Joining lines segments that don't quite join? - Autodesk Community

mdpi.com
Enhancing Small Language Models for Graph Tasks Through Graph Encoder Integration

arxiv.org
Geo-LLaVA: A Large Multi-Modal Model for Solving Geometry Math Problems with Meta In-Context Learning - arXiv

mdpi.com
State of the Art and Future Directions of Small Language Models: A Systematic Review

youtube.com
LLMs vs SLMs: A developer's guide + NVIDIA insights - YouTube

arxiv.org
IMPROVING MULTIMODAL LLM'S ABILITY IN GEOMETRY PROBLEM SOLVING, REASONING, AND MULTISTEP SCORING - arXiv

researchgate.net
(PDF) Assessing sustainable land management (SLM) - ResearchGate

pmc.ncbi.nlm.nih.gov
A Transdisciplinary Framework to Bridge Science–Policy–Development Gaps in Global Land Management Initiatives - PubMed Central

cgspace.cgiar.org
Impact evaluation of SLM options to achieve land degradation neutrality: Dryland Systems interim report. - CGSpace

wocat.net
Sustainable Land Management (SLM) - WOCAT

documents1.worldbank.org
Sustainable Land Management Project I and II - World Bank Documents

openaccess.thecvf.com
SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities - CVF Open Access

arxiv.org
GeoPQA: Bridging the Visual Perception Gap in MLLMs for Geometric Reasoning - arXiv

wangywust.github.io
Towards Comprehensive Reasoning in Vision-Language Models | ICCV 2025 - Yiwei Wang

arxiv.org
rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking

mdpi.com
A Review on Visual-SLAM: Advancements from Geometric Modelling to Learning-Based Semantic Scene Understanding Using Multi-Modal Sensor Fusion - MDPI

github.com
LingDong-/skeleton-tracing: A new algorithm for retrieving ... - GitHub

github.com
Vahe1994/Deep-Vectorization-of-Technical-Drawings ... - GitHub
Ánh xạ tọa độ từ pixel sang giá trị thực
Tôi đang tập trung nghiên cứu cơ chế hiệu chuẩn trục tọa độ, một bước tối quan trọng để đạt được độ chính xác học thuật cho hệ thống Geo-SLM. Bằng cách kết hợp kết quả từ nhận diện ký tự trên các nhãn trục với việc xác định vị trí các dấu tích (tick marks), tôi đang xây dựng các ma trận biến đổi hình học như affine và homography. Điều này cho phép tôi chuyển đổi các thực thể vector từ không gian pixel sang các giá trị dữ liệu thực tế một cách tuyệt đối, giúp loại bỏ hoàn toàn sự sai lệch do độ phân giải hoặc tỷ lệ khung hình không đồng nhất.
Khắc phục ảo giác thông qua biểu diễn ký hiệu
Qua việc đối chứng với các mô hình học sâu hiện đại, tôi nhận thấy rằng phương pháp trích xuất thực thể toán học rời rạc giúp giải quyết triệt để hiện tượng ảo giác số liệu (hallucination). Tôi đang tích hợp các thuật toán xử lý điểm mấu chốt và phân tách đường dựa trên mật độ để xử lý các biểu đồ có lưới tọa độ dày đặc hoặc các đường bị chồng lấp phức tạp. Tôi tin rằng việc tách biệt khâu trích xuất hình học khỏi khâu lập luận ngôn ngữ là cách tối ưu để đảm bảo tính toàn vẹn của dữ liệu trong các phân tích khoa học chuyên sâu.
Tổng hợp quy trình và định nghĩa cấu trúc dữ liệu
Tôi đang tiến hành tổng hợp toàn bộ các giai đoạn từ xử lý ảnh âm bản, trích xuất khung xương đến ánh xạ tọa độ vào một quy trình kỹ thuật thống nhất. Bước tiếp theo, tôi sẽ tập trung thiết kế một sơ đồ JSON chuẩn hóa có khả năng lưu trữ không chỉ các con số mà cả các đặc trưng hình thái học như điểm uốn và tính liên tục của các đoạn thẳng. Tôi sẽ hoàn thiện mô hình dữ liệu này để cung cấp một lớp tri thức có cấu trúc, cho phép các mô hình ngôn ngữ nhỏ thực hiện các phép suy luận logic phức tạp dựa trên bản chất hình học của biểu đồ.

Researching websites...

pmc.ncbi.nlm.nih.gov
Automatic Calibration of a Two-Axis Rotary Table for 3D Scanning Purposes - PMC

mdpi.com
Automatic Extrinsic Calibration of 3D LIDAR and Multi-Cameras Based on Graph Optimization - MDPI

attocube.com
Axis Calibration Software for CNC machines and CMMs - attocube

diva-portal.org
Statistical Sensor Calibration Algorithms - Diva-Portal.org

mathworks.com
Specify Axis Tick Values and Labels - MATLAB & Simulink - MathWorks

infragistics.com
Axes, Tick Marks, Tick Labels, and Grid Lines - Infragistics

quanthub.com
5 Tips for Axis Tick Marks in Chart Design - QuantHub

tableau.com
An Extension of Wilkinson's Algorithm for Positioning Tick Labels on Axes - Tableau

stackoverflow.com
Tickmark algorithm for a graph axis - Stack Overflow

stackoverflow.com
Convert Image Pixel Data to Coordinate Array - Stack Overflow

scichart.com
Axis APIs - Convert Pixel to Data Coordinates | JavaScript Chart Documentation - SciChart

mathworks.com
Convert image pixels to xy-coordinates - File Exchange - MATLAB Central - MathWorks

matplotlib.org
Transformations Tutorial — Matplotlib 3.10.8 documentation

spatialdata.scverse.org
Transformations and coordinate systems — spatialdata - scverse

reddit.com
Quick question: Is there any quick-ish way to plot a reverse logarithmic scale on the X-Axis on a chart? : r/excel - Reddit

en.wikipedia.org
Logarithmic scale - Wikipedia

behavioralpolicy.org
Graphs with logarithmic axes distort lay judgments - Behavioral Science & Policy Association

plotdigitizer.com
PlotDigitizer — Extract Data from Graph Image Online

youtube.com
Extract data from Log-Log plots/graphs | webplotdigitizer | Drawing/Graphing-12 - YouTube

docs.opencv.org
Detection of ArUco Markers - OpenCV Documentation

medium.com
Simple Lane Detection with OpenCV | by Matt Hardwick - Medium

stackoverflow.com
Detecting tick marks with python opencv - Stack Overflow

geeksforgeeks.org
Detecting ArUco markers with OpenCV and Python - GeeksforGeeks

fuzzylabs.ai
Checkbox Detection with OpenCV - Fuzzy Labs
Tinh chỉnh cơ chế ánh xạ tọa độ và hiệu chuẩn dữ liệu
Tôi đã tìm ra phương pháp tối ưu để chuyển đổi từ tọa độ pixel sang giá trị thực thông qua việc xác định chính xác vùng hiển thị đồ thị và các dấu vạch phân đoạn (tick marks). Bằng cách kết hợp kết quả nhận diện ký tự trên nhãn trục với việc xác định các giao điểm hình học, tôi đang xây dựng các ma trận biến đổi affine và homography chặt chẽ. Cách tiếp cận này giúp tôi đảm bảo tính tổng quát cho hệ thống Geo-SLM, giúp dữ liệu trích xuất không bị phụ thuộc vào tỷ lệ khung hình hay sự sai lệch về độ phân giải của ảnh gốc.
Giải quyết mâu thuẫn giữa độ chính xác số liệu và khả năng lập luận
Qua việc đối chứng sâu, tôi nhận thấy các mô hình học sâu truyền thống thường làm mất đi các chi tiết hình thái quan trọng trong quá trình chuyển đổi biểu đồ thành văn bản. Tôi đang giải quyết mâu thuẫn này bằng cách sử dụng kỹ thuật 'trường khung' để xử lý triệt để các điểm giao cắt phức tạp mà không gây ra hiện tượng ảo giác số liệu. Tôi đang tiến tới việc thiết kế một sơ đồ dữ liệu trung gian mang tính ký hiệu, nơi các thuộc tính như điểm uốn, xu hướng và các nút giao được mã hóa rõ ràng để hỗ trợ tối đa cho khả năng tư duy logic của các mô hình ngôn ngữ nhỏ.
Thiết lập quy trình kỹ thuật và hạ tầng thực thi hiệu năng cao
Tôi đang hoàn thiện những khâu cuối cùng trong việc tuyển chọn các thư viện xử lý tín hiệu hình ảnh (ISP) cấp thấp, ưu tiên các giải pháp có khả năng tối ưu hóa phần cứng để đảm bảo tốc độ xử lý vượt trội. Tôi sẽ tập trung tổng hợp toàn bộ các nghiên cứu thành một quy trình kỹ thuật từng bước, bao gồm cả thiết kế JSON Schema chuẩn hóa cho các thực thể toán học. Tôi dự định thực hiện một rà soát cuối cùng về các kỹ thuật nội suy và tăng tốc tính toán trước khi đóng gói báo cáo phương pháp luận hoàn chỉnh.

Researching websites...

en.wikipedia.org
Halide (programming language) - Wikipedia

docs.qualcomm.com
The Halide programming language - Qualcomm Docs

developer.arm.com
Halide: A novel programming paradigm for advanced image processing - Arm Developer

halide-lang.org
Halide lang

people.csail.mit.edu
Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines - People

pmc.ncbi.nlm.nih.gov
Hydra image processor: 5-D GPU image analysis library with MATLAB and python wrappers

developer.nvidia.com
CUDA-X GPU-Accelerated Libraries - NVIDIA Developer

github.com
tier4/accelerated_image_processor: A library to provide hardware acceleration for image processing functions such as compression and rectification. - GitHub

rapids.ai
Ecosystem | RAPIDS | RAPIDS | GPU Accelerated Data Science

reddit.com
PhotoFF a CUDA-accelerated image processing library : r/Python - Reddit

research.manchester.ac.uk
Hardware implementation of skeletonization algorithm for parallel asynchronous image processing - Research Explorer - The University of Manchester

ieeexplore.ieee.org
FPGA Implementation for Skeletonization of 2-D Images - IEEE Xplore

mdpi.com
FPGA-Based Processor Acceleration for Image Processing Applications - MDPI

ieeexplore.ieee.org
Real-time skeletonization using FPGA - IEEE Xplore
Củng cố giải pháp xử lý giao điểm và tính liên tục
Tôi đã hoàn tất việc tổng hợp các kỹ thuật xử lý tại những điểm giao cắt phức tạp thông qua việc ứng dụng trường khung và phân tích nhánh ở mức độ dưới pixel. Bằng cách ưu tiên tính liên tục của góc lệch và bảo toàn cấu trúc topo, tôi đã xác lập được phương pháp tách biệt các chuỗi dữ liệu đơn sắc mà không gây ra sai số hay đứt gãy. Điều này đảm bảo rằng hệ thống Geo-SLM có thể duy trì độ chính xác tuyệt đối ngay cả khi các đường biểu đồ đan xen hoặc chồng lấp lên lưới tọa độ.
Xác lập ưu thế của phương pháp ký hiệu so với học sâu
Thông qua việc đối chứng sâu giữa các mô hình học sâu hiện đại và quy trình xử lý thị giác máy tính truyền thống, tôi nhận thấy phương pháp trích xuất thực thể toán học rời rạc vượt trội hơn hẳn về khả năng loại bỏ hoàn toàn hiện tượng ảo giác số liệu. Tôi đã tư duy theo hướng xây dựng một đồ thị ngữ cảnh (scene graph) thay vì chỉ đọc pixel đơn thuần, giúp chuyển hóa các hình ảnh thô thành một lớp tri thức có cấu trúc bền vững, phục vụ trực tiếp cho các phép lập luận logic chuyên sâu.
Thiết lập quy trình thực thi và chuẩn hóa dữ liệu đầu ra
Tôi đang tiến tới giai đoạn cuối cùng là hệ thống hóa toàn bộ các khám phá thành một quy trình kỹ thuật từng bước, từ khâu xử lý ảnh âm bản đến ánh xạ tọa độ thực tế. Tôi sẽ tập trung tuyển chọn các thư viện xử lý tín hiệu hình ảnh hiệu suất cao dựa trên C++ và Rust, đồng thời hoàn thiện sơ đồ JSON Schema chuẩn hóa. Đây sẽ là bước đóng gói quan trọng để cung cấp một báo cáo phương pháp luận toàn diện, cho phép các mô hình ngôn ngữ nhỏ thực hiện suy luận dựa trên bản chất hình học của biểu đồ.