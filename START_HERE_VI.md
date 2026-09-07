# Bắt đầu với Digit Lab

Bạn chưa cần chuẩn bị dữ liệu hay GPU. Đây là bộ khởi đầu đã chạy được, gồm notebook, code và kết quả thật. Phần khởi tạo được làm với ChatGPT; phần bạn tự thử nghiệm sẽ được ghi riêng trong EXPERIMENT_LOG.md.

## 1. Chạy lần đầu trên Colab

1. Giải nén digit-lab.zip trên máy tính.
2. Mở https://colab.research.google.com/ và chọn Upload notebook (Tải notebook lên).
3. Chọn file digit_lab.ipynb trong thư mục vừa giải nén.
4. Chạy lần lượt bằng nút bên trái từng ô code, hoặc chọn Runtime → Run all.
5. Đối chiếu kết quả với bảng trong README. Phiên bản thư viện khác có thể làm kết quả khác đôi chút.

Không cần tải train.py vào Colab: notebook đã có toàn bộ code. Nếu môi trường báo thiếu thư viện, tải requirements.txt vào Colab rồi chạy một ô `%pip install -r requirements.txt` và khởi động lại runtime trước khi chạy notebook. Các kết quả mới nằm trong thư mục results của Colab; tải xuống nếu muốn giữ.

## 2. Hiểu project bằng ngôn ngữ đơn giản

- Đầu vào: mỗi ảnh có 8 × 8 = 64 ô sáng tối, giá trị từ 0 đến 16.
- Đầu ra: một số nguyên từ 0 đến 9.
- Training set: 1.437 ảnh để học và chọn mô hình qua cross-validation.
- Test set: 360 ảnh giữ riêng để kiểm tra mô hình đã chọn.
- Baseline: luôn đoán chữ số xuất hiện nhiều nhất trong tập học. Nó giúp kiểm tra xem mô hình học được gì hơn cách đoán đơn giản.
- Logistic Regression: mô hình phân loại; tên có chữ Regression nhưng ở đây dùng để phân loại chữ số.
- SVM RBF: phân loại với ranh giới phi tuyến.
- StandardScaler: đưa từng đặc trưng về thang đo dựa trên tập học. Pipeline giúp bước này không nhìn trộm dữ liệu validation.
- Accuracy: số ảnh đoán đúng chia tổng ảnh.
- Macro F1: tính F1 từng chữ số rồi lấy trung bình, mỗi chữ số có trọng số như nhau.
- Confusion matrix: bảng xem chữ số thật thường bị đoán thành chữ số nào; đường chéo là dự đoán đúng.

## 3. Đọc kết quả thay vì chỉ nhìn điểm

SVM đạt 353/360 = 98,06% trong lần chạy đi kèm. Nhưng khi thêm nhiễu Gaussian có độ lệch chuẩn 1, điểm trung bình chỉ còn 16,25%. Điều này không có nghĩa mọi SVM đều kém: đây là một pipeline cụ thể và một loại nhiễu cụ thể.

Một hướng giải thích cần kiểm tra: một số pixel nền ít thay đổi nên StandardScaler chia cho độ lệch chuẩn rất nhỏ, làm nhiễu bị phóng đại. Hãy kiểm tra giả thuyết bằng thí nghiệm, không viết như một kết luận đã chứng minh.

## 4. Bài tập đầu tiên để có đóng góp riêng

1. Chạy bản gốc và ghi ngày chạy, phiên bản, kết quả.
2. Mở ảnh mistakes.png, chọn hai lỗi và mô tả bạn nhìn thấy gì.
3. Viết dự đoán: nếu thay StandardScaler bằng chia toàn bộ pixel cho 16, độ bền với nhiễu sẽ thay đổi thế nào?
4. Tạo bản sao notebook để thử thay đổi. Chia 16 là phép biến đổi cố định, không cần học từ test set; phải áp dụng nhất quán cho ảnh sạch và ảnh nhiễu.
5. Giữ nguyên seed, tập dữ liệu và các thành phần khác để biết thay đổi đến từ đâu.
6. Ghi cả trường hợp không cải thiện vào EXPERIMENT_LOG.md.

Sau khi đã xem kết quả test, việc dùng lại tập này để thử cải tiến chỉ mang tính khám phá. Chọn mô hình bằng cross-validation trên tập training; muốn xác nhận một cải tiến mới cần dữ liệu đánh giá thực sự chưa xem.

## 5. Đưa lên GitHub

Tạo repo mới, riêng với repo hồ sơ ltgbao1107:

- Tên: digit-lab
- Description: Handwritten digit classification with baselines, error analysis, and a noise robustness experiment.
- Public nếu bạn muốn người xem hồ sơ mở được.
- Có thể bật Add a README để trang repo có sẵn nút Add file.

Sau khi tạo, chọn Add file → Upload files. Mở thư mục digit-lab đã giải nén và tải các file cùng thư mục results lên gốc repo. Không tải riêng file ZIP: GitHub cần file đã giải nén để hiển thị README và notebook. Thay README mặc định bằng README của project. Không cần tải file .gitignore nếu giao diện của bạn không hiện file đó.

Giữ nguyên các tên file và đường dẫn results để ảnh trong README hiển thị. Bấm Commit changes. Mở README và notebook để kiểm tra lại. Sau đó có thể ghim repo trên trang cá nhân qua Customize your pins.

Kết nối GitHub trong cuộc trò chuyện trước đã bị từ chối quyền ghi, vì vậy bộ này được giao dạng ZIP; chưa đăng lên tài khoản của bạn.

## 6. Dùng trong hồ sơ du học

Lúc này bạn có một project khởi đầu được AI hỗ trợ. Sau khi tự chạy, hiểu và bổ sung thí nghiệm, hãy mô tả cụ thể phần bạn thực sự làm: thay đổi gì, kiểm tra thế nào, học được gì. README đã ghi nguồn dữ liệu và vai trò của AI. Không cần thêm chức danh researcher hay tự nhận tạo bộ dữ liệu.

## 7. Tự kiểm tra xem đã hiểu chưa

- Vì sao cần baseline?
- Tại sao không chọn mô hình bằng điểm test?
- Vì sao scaler phải nằm trong pipeline?
- 98,06% trên bộ dữ liệu này có nghĩa là nhận đúng 98,06% ảnh điện thoại không?
- Kết quả nhiễu làm bạn thay đổi nhận xét về mô hình thế nào?

Gợi ý: baseline là mốc so sánh; dùng test để chọn sẽ làm đánh giá lạc quan; pipeline giới hạn scaler trong dữ liệu học; ảnh điện thoại khác phân phối dữ liệu; accuracy ảnh sạch không đủ mô tả độ bền của mô hình.
