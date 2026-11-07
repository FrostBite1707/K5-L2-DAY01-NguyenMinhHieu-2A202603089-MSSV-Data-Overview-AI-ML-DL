# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**2026-11-06

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:**Python 3.10 / PyTorch 2.x / Ultralytics 8.3.x

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=1 ; `class_name`=468 ; `rank`=1 ; `score`= 0.510915; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì record chỉ gán nhãn cho cả bức ảnh chứ không có tọa độ bounding box, segmentation mask, hay nhãn cho vật thể nào khác cả.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp đi kèm từ taxonomy và checkpoint chứ model không tự nghĩ ra mà chỉ đơn giản là neuron ở lớp đầu ra ánh xạ xác suất vào danh sach đã được cấu hình sẵn.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
-Tránh xung đột ngữ nghĩa (Disambiguation): Một class_name có thể mang nhiều nghĩa (ví dụ: crane vừa là loài chim sếu, vừa là cần cẩu). class_id định danh duy nhất chỉ mục đó trong hệ thống.
-Đồng bộ hóa trong lập trình: Code và ma trận đầu ra của mô hình làm việc trực tiếp với số nguyên (class_id), trong khi người gán nhãn và đánh giá cần class_name để đối soát trực quan.
+Đảm bảo khả năng truy xuất nguồn gốc (Traceability): Cùng một tên lớp hoặc cùng một ID có thể mang ý nghĩa khác nhau ở các bộ dữ liệu khác nhau (ví dụ: COCO có 80 lớp, ImageNet có 1.000 lớp). taxonomy_name xác định rõ không gian nhãn cụ thể để tránh việc nhầm lẫn khi chuyển đổi checkpoint hay pipeline xử lý dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
-Quy tắc đối tượng chiếm ưu thế (Dominant Object Rule): Ưu tiên vật thể chiếm diện tích lớn nhất ở tiền cảnh (foreground), nằm ở trọng tâm khung hình hoặc có độ sắc nét/lấy nét chính (in-focus).
-Quy tắc phân cấp ưu tiên (Hierarchy/Tie-breaking Rule): Bảng phân cấp nhãn rõ ràng nếu có hai vật thể nổi bật ngang nhau (ví dụ: ưu tiên hành động/con người > phương tiện > vật thể tĩnh; hoặc đối tượng hành động > bối cảnh nền).
-Quy tắc phân biệt bối cảnh và vật thể: Quy định khi nào nhãn được chọn theo bối cảnh chung (scene-level như street, kitchen) và khi nào chọn theo vật thể đơn lẻ bên trong bối cảnh đó.
6. Vì sao không được lấy `score` của model làm ground truth?
-Bản chất của score: Score chỉ là giá trị xác suất/độ tự tin tính toán từ hàm kích hoạt (như Softmax) dựa trên các trọng số hiện tại của mô hình, không phải là sự thật khách quan (ground truth).
-Sai số và tự tin ảo (Overconfidence): Mô hình thường xuyên gặp lỗi nhận diện sai, thiên kiến dữ liệu (bias) hoặc gán mức độ tự tin rất cao cho một dự đoán không chính xác.
-Tránh vòng lặp suy thoái dữ liệu: Ground truth phải là chuẩn chân lý do con người thẩm định độc lập theo guideline. Nếu lấy score của mô hình làm chuẩn, hệ thống sẽ rơi vào lỗi tự xác thực (confirmation bias), làm mất đi khả năng phát hiện lỗi và đánh giá chính xác chất lượng thực tế của mô hình.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[ 385.33, 69.24, 498.92, 348.92 ],; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
-Góc trên bên trái ở [385.33, 69.24] góc dưới bên phải ở [498.92, 348.92]; rộng 113.58 pixel, cao 279,68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
-Ngưỡng 0.60: Có 5 predictions (gồm: 2 person có score 0.91 và 0.61; 2 oven có score 0.69 và 0.63; 1 bowl có score 0.72 - lưu ý bowl score 0.701 cũng vượt 0.60 nên thực tế có 6 predictions thỏa mãn $\ge 0.60$)
-Ngưỡng 0.20: Có 14 predictions (khi hạ ngưỡng xuống 0.20, notebook giữ thêm các hộp vật thể nhỏ, bị che khuất hoặc có độ tự tin thấp như các bát nhỏ bowl, chén/ly cup, và các vật dụng nền phía xa).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
-Khi hạ ngưỡng tin cậy để không bỏ sót vật thể (tăng Recall), mô hình sẽ sinh ra rất nhiều dự đoán rác và dương tính giả (False Positives - FP). Reviewer phải:
-Rà soát thủ công để xóa bỏ các bounding box rác/nhận diện sai (ví dụ: vết bóng, họa tiết gạch hoặc góc tối bị nhận nhầm thành cup hay bowl).
-Sửa lại nhãn phân loại (class) cho các hộp nhận diện đúng vị trí vật thể nhưng sai nhãn danh mục (ví dụ: nhầm lẫn giữa cup và bowl).
-Điều chỉnh lại ranh giới (box boundary) của các box bị lệch hoặc xóa bớt các box bị trùng lặp/chồng lấn do thuật toán NMS không lọc hết ở ngưỡng thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này). w
Quy tắc viền khít và bao trọn phần nhìn thấy được (Tightness & Visible Boundary Rule):
"Bounding box phải được vẽ ôm sát (tight) toàn bộ phần biên nhìn thấy được (visible pixels) của vật thể mục tiêu, sai số viền trống không vượt quá 2–3 pixel so với mép thực tế. Không cắt cụt các bộ phận nhô ra của vật thể và không cố phỏng đoán/vẽ trùm ra vùng bị vật khác che khuất hoàn toàn."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
-Vật thể quan sát: Chiếc lò / bếp (oven) ở mép trái (bbox_xyxy: [0.11, 187.54, 195.96, 292.81]) bị cắt mép sát góc trái của khung hình ($x_{\min} \approx 0$).
-Quyết định (Viết quy tắc hay Escalation?):
*Trường hợp này nên viết thành quy tắc trong Guideline: Vì vật thể bị cắt mép (truncated) bởi cạnh ảnh hoặc bị che khuất một phần (occluded) bởi vật dụng phía trước là đặc trưng rất phổ biến trong dữ liệu thực tế.
*Nội dung quy tắc: Chỉ gán nhãn nếu phần nhìn thấy của vật thể $\ge 20\%$ diện tích và vẫn nhận dạng rõ ràng; Bounding box chỉ ôm sát phần pixel hiển thị thực tế trong khung ảnh, không vẽ tràn ra ngoài mép ảnh.
*Khi nào mới Escalation (hỏi Lab Coach): Chỉ đưa lên hỏi Mentor/Coach với các trường hợp bất thường không thể phân định được theo taxonomy (ví dụ: vật thể bị che khuất trên 80–90% chỉ còn thấy một mảnh kim loại/nhựa nhỏ không rõ là lò nướng oven, lò vi sóng microwave hay máy nướng bánh mỳ)
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
-Đường viền hình học thực tế (True Contour): Bounding box chỉ cung cấp phạm vi bao ngoài dạng chữ nhật thô ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$), trong khi polygon mô tả chính xác viền đa giác bám sát mép vật thể theo từng pixel.
-Tách biệt vùng nền (Background Exclusion): Hộp chữ nhật luôn chứa cả pixel nền (background noise) ở các góc hoặc khoảng trống; polygon loại bỏ hoàn toàn các pixel không thuộc vật thể.
-Đặc tính hình học chi tiết: Thể hiện trực quan tư thế (pose), độ nghiêng, các đường lượn sóng, góc uốn lượn và phần rỗng/khuyết mà hình hộp chữ nhật không thể hiện được.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
-instance_id là định danh duy nhất dùng để phân biệt từng cá thể/đối tượng riêng lẻ xuất hiện trong cùng một bức ảnh (ví dụ: kitchen-001, kitchen-002):
+Không phải class_id: class_id là mã số đại diện cho cả một loại/danh mục đối tượng trong bộ nhãn (ví dụ: nhãn person có class_id = 0, ảnh có 5 người thì cả 5 người đều chung class_id = 0). Trong khi đó, instance_id gán nhãn riêng cho từng người một.
+Không phải tracking_id: tracking_id dùng trong xử lý video để theo dõi một vật thể duy nhất xuyên suốt qua nhiều khung hình liên tiếp theo thời gian ($t_1, t_2, \dots$). instance_id ở đây chỉ có giá trị cục bộ trong phạm vi một ảnh tĩnh đơn lẻ, không mang thông tin liên kết danh tính qua các ảnh hay khung hình khác.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc bám sát biên điểm ảnh (Pixel-tight Boundary Rule):
    "Đa giác phải được chấm bám sát theo đường viền thực tế nhìn thấy được của vật thể, sai số lệch mép không vượt quá 1–2 pixel. Đặt mật độ điểm dày ở các khúc bo tròn, góc cua gập và thưa hơn ở các cạnh thẳng; tuyệt đối không cắt lẹm vào thân vật thể và không bao trùm các pixel của vật thể khác hoặc nền xung quanh."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
-Vị trí quan sát: Khu vực các chiếc bát/chén (bowl) đặt chạm nhau trên bàn/kệ bếp (ví dụ vị trí của kitchen-002, kitchen-003, kitchen-006), hoặc phần chân/thân của người (person) bị che khuất một phần phía sau quầy bếp/bàn ăn (dining table).
-Lựa chọn xử lý (Guideline hay Escalation?):
*Sử dụng Quy tắc trong Guideline.
-Lý do:
*Hiện tượng các đồ vật gia dụng xếp đè/chạm nhau (touching instances) hoặc người đứng sau bàn ghế (occlusion) là tình huống rất kinh điển trong môi trường thực tế.   
*Guideline hoàn toàn có thể chuẩn hóa trước bằng các điều khoản rõ ràng:
+Khi hai vật cùng loại chạm nhau: Vẽ hai polygon riêng biệt tiếp xúc sát ranh giới, không vẽ lấn đè (overlap) lên nhau.
*Khi vật thể bị che khuất: Chỉ vẽ mask bao quanh phần điểm ảnh thực sự nhìn thấy (visible part); không tự phỏng đoán vùng bị che lấp bên dưới quầy bếp.
-Chỉ cần escalation (hỏi Lab Coach) khi ranh giới giữa hai vật bị mờ/tối đến mức hoàn toàn không thể phân biệt được bằng mắt thường đâu là mép của vật A và đâu là mép của vật B. 

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth (Đơn vị nhãn) | Lỗi / Khó quyết định (Dẫn chứng) | Annotator làm | Reviewer xem |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Nhãn gán toàn ảnh: `class_id`, `class_name`, `taxonomy_name`

 | Dự đoán `cab` (`score: 0.51`) cho ảnh `traffic` chứa nhiều đối tượng (xe buýt, ô tô, người) khiến khó định đoạt nhãn chủ đề đơn lẻ.

 | Chọn **duy nhất một nhãn** bao quát hoặc mang tính áp đảo nhất theo quy tắc ưu tiên.

 | Kiểm tra nhãn có bám sát taxonomy/guideline không; tránh thiên kiến chọn nhãn theo cảm tính.

 |
| **Phát hiện vật thể** | Bounding box: `bbox_xyxy`, `class_name`

 | Mép ảnh `kitchen` cắt cụt `oven` (`bbox_xyxy: [0.11, 187.54, 195.96, 292.81]`); vật nhỏ phía xa dễ bị bỏ sót khi nâng ngưỡng lọc. | Vẽ box ôm sát phần nhìn thấy của vật thể; gắn đúng nhãn phân loại theo taxonomy. | Kiểm tra độ khít (tightness) của viền box, rà soát vật thể bị bỏ sót và xóa các box đoán nhầm (dương tính giả). |
| **Phân đoạn đối tượng** | Đa giác khép kín: `polygon_xy`, `instance_id`, `class_name` | Các vật thể chạm nhau như bát đĩa `bowl` (`kitchen-002`, `003`, `006`); bề mặt bàn `dining table` (`kitchen-005`, 598 điểm) bị nhiều vật đè lên làm ranh giới zích zắc, khó phân tách. | Chấm các điểm polygon bám khít đường biên pixel thực tế của từng đối tượng riêng biệt; không vẽ đè lên vật khác. | Soi kỹ độ chính xác từng pixel ở ranh giới tiếp xúc, phát hiện lỗi lấn biên (overlap) hoặc gộp nhầm 2 đối tượng vào 1 mask. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ sử dụng ba ảnh COCO công khai có sẵn trong bài lab, tuyệt đối không tải lên ảnh cá nhân hay dữ liệu nội bộ.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Xử lý dữ liệu không đúng phạm vi: Bạn cần dừng lại ngay lập tức và báo cáo cho Lab Coach / Mentor (hoặc Giảng viên phụ trách) để được hướng dẫn xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v] `day1_lab_outputs/detection_predictions.json`
- [v] `day1_lab_outputs/segmentation_predictions.json`
- [v] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v] `day1_lab_outputs/visuals/classification_top5.png`
- [v] `day1_lab_outputs/visuals/detection_predictions.png`
- [v] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp **link repository** trên LMS
