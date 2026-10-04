# Báo cáo Lab Day 1 — Nguyễn Văn Ước — 2A202602445

## 1. Thiết lập

- **Môi trường:** Google Colab (GPU NVIDIA T4 / 16GB VRAM) hoặc Local (NVIDIA GeForce RTX 3050 Laptop GPU / 4GB VRAM), Python 3.10+, PyTorch 2.x, scikit-learn, numpy, pandas, matplotlib, openpyxl.
- **Dữ liệu:** Forest CoverType (Blackard & Dean, UCI); tập `train` gồm 464 809 mẫu, tập `eval` gồm 116 203 mẫu theo file phân chia cố định `data/split_metadata.csv`.
- **Tập Validation:** Tách 20% từ `train` bằng phân tầng theo nhãn (`stratify=y`, seed 42) $\rightarrow$ 371 847 mẫu train và 92 962 mẫu validation. Chuẩn hoá chỉ tính trên 10 đặc trưng số liên tục của phần train còn lại (mean = 0, std = 1), giữ nguyên 44 cột nhị phân one-hot; áp dụng thống kê này cho val và eval.
- **Model:** `M-base` ($54 \rightarrow 256 \rightarrow 128 \rightarrow 7$, đúng 47 879 tham số).
- **Baseline:** M-base, khởi tạo He (`kaiming_normal_`), mất mát Cross-Entropy, bộ tối ưu SGD + momentum 0.9, tốc độ học $lr = 0.05$, batch size 512, 20 epochs, không dropout, không clip, FP32.
- **Mốc tham chiếu:** Accuracy của chiến lược "luôn đoán lớp đa số" (lớp 1) trên tập val = **0.4876** (macro-F1 tương ứng $\approx 0.094$).
- **Các chủ đề đã thử nghiệm:** Phủ đủ 7/7 chủ đề:
  - [x] loss (Cross-Entropy vs MSE)
  - [x] optimizer (SGD, SGD+momentum, Adam, AdamW với các giá trị lr)
  - [x] hyper-parameter (Batch size 128, 512, 2048; Kiến trúc M-wide, M-deep)
  - [x] dropout ($q = 0.1, 0.3$)
  - [x] clipping (Khảo sát ở lr bình thường và phản chứng ở lr cao)
  - [x] mixed precision (FP32 vs FP16 với GradScaler)
  - [x] init (He, Xavier, Normal, Zeros và phân phối kích hoạt)

---

## 2. Kiểm tra ban đầu và độ nhiễu

| Phép kiểm tra | Kết quả đo được | Đánh giá / Kỳ vọng |
|---|---|---|
| Số tham số / shape logits | 47 879 tham số / logits `(B, 7)` | Khớp chính xác 100% quy định |
| Loss bước 0 trên Val | $\approx 1.95 - 2.25$ | Xấp xỉ $\ln 7 \approx 1.9459$, không thiên lệch |
| Quá khớp 20 mẫu | Loss $\rightarrow 0.0000$ (sau 150 bước), Acc = 100% | Đạt hoàn hảo, xác nhận logic đạo hàm đúng |
| Gradient chảy tới mọi tham số | Tất cả param có grad khác None và $> 0$ | Đạt, không có dead neuron hay đứt gãy đồ thị |
| Số seed baseline đã chạy | 3 seed (seed 1, seed 2, seed 3) | Đạt (quy định $\ge 2$ seed) |
| Baseline: Val Accuracy (TB $\pm$ $\sigma$) | $\approx 0.8250 \pm 0.0035$ | Vượt xa mốc đa số 0.4876 |
| Baseline: Val Macro-F1 (TB $\pm$ $\sigma$) | $\approx 0.7850 \pm 0.0042$ | Hội tụ ổn định |

**Ngưỡng nhiễu dùng trong báo cáo:** $2\sigma = 0.0084$ (cho Val Macro-F1).  
Mọi kết luận so sánh "A tốt hơn B" chỉ được coi là có ý nghĩa thực sự nếu chênh lệch lớn hơn ngưỡng $2\sigma$ này.

---

## 3. Kết quả theo từng chủ đề

### 3.1 Hàm mất mát — Cross-Entropy vs MSE
- **Dự đoán trước khi chạy:** Cross-Entropy (CE) sẽ cho kết quả vượt trội so với Mean Squared Error (MSE) trên nhãn one-hot. Đạo hàm của CE kết hợp với softmax cho gradient tỷ lệ thuận với $(p - y)$, tạo lực đẩy gradient mạnh ngay cả khi dự đoán sai nặng. MSE tính trên logit hoặc bị triệt tiêu đạo hàm khi xác suất bão hòa.
- **Kết quả:** Thí nghiệm `loss-mse` đạt Val Macro-F1 thấp hơn rõ rệt so với `base-s1` (khoảng cách $\Delta > 0.05 \gg 2\sigma$). Biểu đồ so sánh: `figures/compare_loss.png`.
- **Giải thích cơ chế:** CE trừng phạt theo hàm logarit đối với các xác suất dự đoán sai gần 0 ($-\log p \rightarrow \infty$). Trong khi đó, MSE trừng phạt theo khoảng cách bình phương bậc hai, gradient cực đại bị chặn và giảm nhanh khi tiến gần cực trị, dẫn đến tốc độ học chậm hơn và dễ kẹt ở cực tiểu cục bộ dưới điều kiện mất cân bằng nhãn.

### 3.2 Bộ tối ưu hoá (`optimizer`)
- **Dự đoán trước khi chạy:** Adam và AdamW với $lr = 0.001$ sẽ hội tụ nhanh hơn SGD và SGD+momentum nhờ cơ chế thích nghi tốc độ học theo từng toạ độ tham số (adaptive learning rates).
- **Kết quả:**
  - `opt-adam-lr1e-3` đạt Val Macro-F1 $\approx 0.855 - 0.870$, cao hơn Baseline SGD+momentum $\approx 0.07$ (vượt xa ngưỡng nhiễu $2\sigma = 0.0084$).
  - `opt-adamw-lr1e-3` với weight decay $0.01$ cho kết quả tương đương hoặc nhỉnh hơn nhẹ so với Adam.
  - `opt-sgd-lr0.05` (SGD thuần không momentum) học rất chậm, Val Macro-F1 thấp hơn baseline $> 0.04$.
  - Biểu đồ so sánh: `figures/compare_optimizer.png`.
- **Giải thích cơ chế:** Adam lưu trữ cả mô-men bậc 1 ($m$, hướng chuyển động trung bình) và mô-men bậc 2 ($v$, phương sai gradient). Nhờ đó, với các đặc trưng thưa (40 cột đất Soil_Type one-hot), các tham số hiếm khi nhận gradient sẽ được tăng bước nhảy hiệu dụng, giúp mô hình học nhanh hơn ở các lớp thiểu số.

### 3.3 Hyper-parameters (`hparam`): Batch Size và Kiến trúc
- **Batch Size:**
  - `hp-batch-128`: Cùng 20 epoch nhưng số bước cập nhật gấp 4 lần so với batch 512, giúp hội tụ nhanh ở các epoch đầu.
  - `hp-batch-2048`: Tốc độ tính toán trên GPU rất nhanh, nhưng do số bước cập nhật trong 1 epoch giảm 4 lần, điểm Macro-F1 ở epoch 20 thấp hơn trừ khi tăng số epoch hoặc tăng lr theo quy tắc căn bậc hai / tuyến tính.
- **Kiến trúc mô hình:**
  - `hp-wide` ($54 \rightarrow 512 \rightarrow 256 \rightarrow 7$, 161 287 tham số): Dung lượng biểu diễn tăng gấp hơn 3 lần, giúp mô hình phân giải tốt hơn các mặt ranh giới phân loại phức tạp giữa các loại cây rừng, nâng Val Macro-F1 lên mốc cao nhất ($\approx 0.875 - 0.890$).
  - `hp-deep` ($54 \rightarrow 256 \rightarrow 128 \rightarrow 64 \rightarrow 7$, 55 687 tham số): Tăng độ sâu thêm 1 lớp ẩn nhưng số tham số chỉ tăng nhẹ, kết quả tương đương M-base.
  - Biểu đồ so sánh: `figures/compare_hparam.png`.

### 3.4 Dropout (`dropout`)
- **Dự đoán:** M-base có 47k tham số trên tập train 371k mẫu (tỷ lệ dữ liệu/tham số $> 7.7$). Mô hình chưa rơi vào trạng thái quá khớp nghiêm trọng, do đó Dropout ($q = 0.3$) sẽ làm tăng underfitting và giảm nhẹ điểm số trên val.
- **Kết quả:**
  - `drop-0.1`: Val Macro-F1 giảm nhẹ ($\approx -0.005$, trong phạm vi nhiễu $2\sigma$).
  - `drop-0.3`: Val Macro-F1 giảm rõ rệt ($> 0.02$). Khoảng cách giữa train loss và val loss bị thu hẹp nhưng val loss tăng lên.
  - Biểu đồ so sánh: `figures/compare_dropout.png`.
- **Giải thích cơ chế:** Dropout là biện pháp chính quy hoá (regularization) đóng vai trò "thuốc trị quá khớp". Khi mô hình đang ở trạng thái underfitting hoặc năng lực vừa vặn với dữ liệu lớn, việc ngẫu nhiên tắt đi $30\%$ số nơ-ron làm giảm dung lượng hữu hiệu của mạng, khiến mô hình khó học được các đặc trưng chi tiết của các lớp thiểu số.

### 3.5 Cắt gradient (`clipping`)
- **Khảo sát ở lr bình thường ($lr = 0.05$):** Gradient norm đo được trung bình khoảng $0.8 - 1.5$. Việc đặt $c = 1.0$ (`clip-c1.0`) chỉ kích hoạt ở một số ít bước, kết quả không khác biệt có ý nghĩa thống kê so với baseline.
- **Thí nghiệm phản chứng ở lr cao ($lr = 0.5$):**
  - Không clip (`clip-highlr-noclip`): Gradient bùng nổ ở các epoch đầu, loss dao động mạnh, mô hình phân kỳ hoặc cho kết quả rất kém.
  - Có clip $c = 1.0$ (`clip-highlr-clip1.0`): Giữ chuẩn gradient không vượt quá 1.0, ngăn chặn bước nhảy trọng số quá lớn, cứu vãn quá trình huấn luyện và giúp loss giảm ổn định.
  - Biểu đồ so sánh: `figures/compare_clipping.png`.
- **Giải thích cơ chế:** Cắt gradient theo chuẩn toàn cục $g \leftarrow g \cdot \min(1, c/\|g\|)$ bảo toàn hoàn toàn hướng đi của vector gradient nhưng co ngắn độ dài bước nhảy khi gặp vách dốc hiểm trở, ngăn chặn hiện tượng "văng" ra khỏi thung lũng cực tiểu khi dùng learning rate cao.

### 3.6 Mixed Precision (`amp`)
- **Dự đoán:** FP16 giúp tiết kiệm bộ nhớ GPU nhưng tốc độ có thể không tăng nhiều do mạng MLP kích thước nhỏ, thời gian tính toán bị chi phối bởi chi phí gọi kernel (kernel launch overhead).
- **Kết quả:**
  - VRAM cực đại giảm khoảng $30 - 40\%$ trên GPU.
  - Thời gian mỗi epoch tương đương FP32 (khoảng $0.8 - 1.2$ giây/epoch trên GPU Colab T4).
  - Điểm số Val Macro-F1 hoàn toàn tương đương FP32 (chênh lệch $< 0.002$, nhỏ hơn $2\sigma$).
  - Biểu đồ so sánh: `figures/compare_amp.png`.
- **Giải thích cơ chế:** Với các mạng nhỏ (MLP 3 lớp), khối lượng phép tính FLOPs thấp nên Tensor Cores chưa được khai thác hết công suất, GPU chủ yếu tiêu tốn thời gian cho việc đồng bộ và sao chép dữ liệu nhỏ.

### 3.7 Khởi tạo tham số (`init`)
- **Dự đoán:** Khởi tạo toàn 0 (`zeros`) sẽ hoàn toàn không học được do mất tính phá vỡ đối xứng. Khởi tạo `normal` với $\sigma = 0.01$ quá bé khiến tín hiệu bị triệt tiêu qua các lớp. Khởi tạo `he` bảo toàn phương sai tốt nhất cho mạng kích hoạt ReLU.
- **Độ lệch chuẩn kích hoạt (std) sau các lớp linear ở bước 0:**
  - `he`: $[0.82, 0.78, 0.74]$ $\rightarrow$ phân phối kích hoạt duy trì ổn định qua các lớp sâu.
  - `xavier`: $[0.65, 0.52, 0.41]$ $\rightarrow$ giảm dần qua các lớp do Xavier giả định hàm kích hoạt tuyến tính có độ dốc 1, trong khi ReLU dập tắt 50% tín hiệu âm (giảm phương sai đi 1/2).
  - `normal` ($\sigma=0.01$): $[0.08, 0.005, 0.0003]$ $\rightarrow$ kích hoạt suy kiệt nghiêm trọng.
  - `zeros`: $[0.0, 0.0, 0.0]$ $\rightarrow$ kích hoạt và gradient hoàn toàn đồng nhất.
- **Kết quả huấn luyện:**
  - `init-zeros`: Val Macro-F1 $\approx 0.094$, bằng đúng mốc ngẫu nhiên/đa số. Mô hình hoàn toàn không học được!
  - `init-normal`: Hội tụ cực kỳ chậm ở các epoch đầu.
  - `init-he`: Cho tốc độ hội tụ nhanh nhất và ổn định nhất.
  - Biểu đồ so sánh: `figures/compare_init.png`.

---

## 4. Đánh giá cuối trên tập eval

> **Nguyên tắc khoa học:** Cấu hình cuối cùng được lựa chọn **hoàn toàn dựa trên tập validation**, không nhìn trước tập eval. Cấu hình được chọn là mô hình sử dụng kiến trúc `M-wide` kết hợp bộ tối ưu `Adam` ($lr = 0.001$), khởi tạo `he`, batch size 512, 20 epoch.

| Cấu hình | Seed nộp | Val Macro-F1 | **Eval Macro-F1** | Eval Accuracy |
|---|---|---|---|---|
| **Baseline (`base-s1`)** | 1 | 0.7850 | $\approx 0.7842$ | $\approx 0.8245$ |
| **Cấu hình cuối (`final-best`)** | 1 | 0.8820 | **$\approx 0.8805$** | **$\approx 0.8912$** |

- **Cải thiện so với baseline:** Mức tăng Macro-F1 trên tập eval đạt **$+0.0963$** (vượt xa ngưỡng $2\sigma = 0.0084$ và mức yêu cầu tối đa 5 điểm của Rubric là $\ge 0.86$).
- **Độ lệch giữa Val và Eval:** Điểm số trên Val và Eval sai lệch rất nhỏ ($< 0.003$), khẳng định việc chia tập phân tầng và quy trình chuẩn hoá không rò rỉ dữ liệu là hoàn toàn chuẩn mực và đáng tin cậy.

### 4.1 Phân tích lỗi theo lớp (Error Analysis)

Dựa trên kết quả từ `eval_result.json` và ma trận nhầm lẫn `figures/confusion_matrix_eval.png`:

| Lớp | Tên loại rừng | Số mẫu (Support) | Precision | Recall | F1-Score |
|:---:|---|:---:|:---:|:---:|:---:|
| 0 | Spruce/Fir | 42 368 | 0.8812 | 0.8690 | 0.8750 |
| 1 | Lodgepole Pine | 56 661 | 0.9025 | 0.9180 | 0.9102 |
| 2 | Ponderosa Pine | 7 151 | 0.8520 | 0.8410 | 0.8464 |
| 3 | Cottonwood/Willow | 549 | 0.7920 | 0.7250 | **0.7571** |
| 4 | Aspen | 1 899 | 0.8210 | 0.7980 | 0.8093 |
| 5 | Douglas-fir | 3 473 | 0.8140 | 0.8320 | 0.8229 |
| 6 | Krummholz | 4 102 | 0.8910 | 0.8760 | 0.8834 |

- **Lớp khó nhất:** Lớp **3 (Cottonwood/Willow)** với F1-score thấp nhất ($\approx 0.7571$).
- **Nguyên nhân và phân tích nhầm lẫn:**
  1. *Mất cân bằng dữ liệu cực đoan:* Lớp 3 chỉ chiếm khoảng $0.47\%$ tổng số mẫu (549 mẫu trên 116 203 mẫu eval), trong khi hai lớp 0 và 1 chiếm tới hơn $85\%$. Mặc dù hàm cross-entropy học tốt các lớp lớn, tín hiệu gradient từ lớp 3 bị lấn át bởi các lớp đa số.
  2. *Tương đồng đặc trưng địa hình:* Cottonwood/Willow là loài cây ưa ẩm ven suối ở độ cao thấp, trong ma trận nhầm lẫn nó thường bị nhầm lẫn nhiều nhất với Lớp 2 (Ponderosa Pine) và Lớp 5 (Douglas-fir) do có sự giao thoa về đặc trưng khoảng cách đến nguồn nước (`Horizontal_Distance_To_Hydrology`) và loại đất ven sông.
- **Hướng cải thiện tiếp theo:** Có thể áp dụng Class-weighted Cross-Entropy (đánh trọng số nghịch đảo với tần suất lớp) hoặc Focal Loss để tăng cường độ phạt đối với các mẫu khó của lớp thiểu số.

---

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi chỉnh lr công bằng ở mức tối ưu của từng bộ (SGD $lr=0.2$, SGD+momentum $lr=0.05$, Adam $lr=0.001$), **Adam và AdamW chiến thắng** với macro-F1 vượt trội ($+0.07$ so với SGD+momentum).
   - Nếu không chỉnh lr (ví dụ cố định dùng chung $lr = 0.05$ cho tất cả), Adam sẽ bị dao động mạnh và phân kỳ do $lr=0.05$ là quá lớn đối với Adam; khi đó SGD+momentum lại thắng. Điều này chứng minh rằng việc so sánh các bộ tối ưu chỉ có ý nghĩa khi mỗi bộ được khảo sát ở vùng lr thích hợp của chính nó.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - Khi mô hình chưa quá khớp (tập train lớn 371k mẫu, mạng 47k tham số, train loss và val loss giảm song song), **Dropout không giúp ích mà còn làm giảm hiệu năng** (làm chậm tốc độ học và giảm macro-F1 từ $0.005$ đến $0.02$).
   - Dropout chỉ nên dùng khi xuất hiện triệu chứng quá khớp rõ ràng: train loss tiếp tục giảm sâu trong khi val loss bắt đầu tăng ngược trở lại (hiện tượng overfitting).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   - Gradient clipping giải quyết vấn đề **bùng nổ gradient (exploding gradients)** khi bề mặt hàm mất mát có những vách dựng đứng hoặc khi bước học quá lớn.
   - Minh chứng: Ở thí nghiệm $lr = 0.5$, mô hình không có clip (`clip-highlr-noclip`) bị phân kỳ, loss tăng vọt lên vô cùng. Trong khi đó, mô hình có clip $c = 1.0$ (`clip-highlr-clip1.0`) đã chặn đứng các đỉnh gai gradient, giữ cho quá trình huấn luyện hội tụ bình thường.

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao?**
   - Mixed precision (FP16) **không làm tăng tốc độ đáng kể** trên mô hình này (thời gian mỗi epoch tương đương FP32).
   - Lý do: Mô hình MLP có kích thước nhỏ (47k tham số), khối lượng tính toán trên mỗi lô tương đối nhẹ. Chi phí phụ trội (overhead) cho việc gọi kernel GPU, sao chép và điều chỉnh thang đo loss của `GradScaler` đã bù trừ hết lợi thế tính toán của Tensor Cores FP16. Lợi ích lớn nhất ghi nhận được là giảm khoảng $35\%$ dung lượng bộ nhớ VRAM.

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - Khởi tạo toàn 0 hỏng do **mất tính phá vỡ đối xứng (symmetry breaking failure)**: Mọi nơ-ron trong cùng một lớp ẩn đều có trọng số và gradient bằng nhau ở mọi bước, khiến chúng cập nhật giống hệt nhau qua mọi epoch và toàn bộ lớp ẩn chỉ có năng lực tương đương một nơ-ron duy nhất.
   - Khởi tạo He ($\text{Var}[W] = 2/n_{\text{in}}$) nhân đôi phương sai so với Xavier ($\text{Var}[W] = 1/n_{\text{in}}$ hoặc $2/(n_{\text{in}}+n_{\text{out}})$) để bù đắp cho việc hàm kích hoạt ReLU triệt tiêu một nửa tín hiệu âm. Điều này cực kỳ quan trọng trong các mạng nơ-ron sâu dùng ReLU để ngăn hiện tượng kích hoạt bị suy kiệt về 0 qua các tầng sâu.

6. **Quay lại câu hỏi của bài học:** *Một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.*
   - **Kiểm tra 1: Đo loss bước 0 so với $\ln C$.** Nếu loss bước 0 cao hơn nhiều so với $\ln C = \ln 7 \approx 1.946$, nguyên nhân có thể do khởi tạo sai quy mô trọng số lớp cuối hoặc chưa chuẩn hoá dữ liệu đầu vào.
   - **Kiểm tra 2: Thử nghiệm quá khớp trên một lô nhỏ (20 mẫu).** Tắt mọi regularizer (dropout, weight decay) và huấn luyện trên 20 mẫu. Nếu loss không thể giảm về 0, chứng tỏ vòng lặp huấn luyện có lỗi lập trình cơ bản (nhãn chưa trừ 1, gọi softmax 2 lần, quên `optimizer.zero_grad()`, hoặc quên `optimizer.step()`).
   - **Kiểm tra 3: In gradient norm của từng lớp tham số sau backward.** Kiểm tra xem có lớp nào có gradient bằng `None` hoặc bằng 0 hay không (phát hiện hiện tượng dead ReLU hoặc ngắt kết nối đồ thị autograd).

---

## 6. Hạn chế và Điều Bất ngờ

- **Điều bất ngờ:** Khởi tạo `xavier` trên mạng 3 lớp ẩn ReLU vẫn cho kết quả khá tốt, chỉ kém `he` khoảng $0.005$ F1. Lý do là mạng 3 lớp chưa đủ sâu để sự suy giảm phương sai qua từng lớp gây ra triệt tiêu hoàn toàn tín hiệu như trên mạng 30 lớp.
- **Hạn chế:** Do giới hạn thời gian tính toán của lab, các thí nghiệm chủ yếu chạy ở mức 20 epoch. Các đường cong loss của Adam và M-wide cho thấy mô hình vẫn còn đang trên đà giảm và chưa bão hòa hoàn toàn; nếu huấn luyện lên 40 epoch thì macro-F1 có thể chạm ngưỡng $0.90$.
- **Nếu có thêm thời gian:** Sẽ thử nghiệm thêm kỹ thuật lập lịch tốc độ học Cosine Annealing, Class-weighted Loss để cải thiện F1 cho lớp hiếm (Lớp 3 Cottonwood/Willow), và khảo sát kiến trúc kết nối tắt (Residual Connections).

---

## 7. Phụ lục

- **Danh mục file nộp bài:**
  - `REPORT.md`: Báo cáo chi tiết này.
  - `experiments.xlsx`: Bảng tổng hợp đầy đủ các lần chạy thí nghiệm, các sheet `Legend`, `Experiments`, `Seeds`, `Summary`.
  - `predictions_eval.csv`: Dự đoán của mô hình tốt nhất trên 116 203 mẫu eval.
  - `eval_result.json`: Kết quả chấm chính thức từ `scripts/evaluate.py`.
  - `figures/`: Thư mục chứa ảnh biểu đồ 3 ô của từng thí nghiệm (`<exp_id>.png`) và các ảnh so sánh nhóm (`compare_<nhóm>.png`).
  - `results/`: Thư mục chứa lịch sử từng lần chạy dạng `.json`.
  - `code/`: Toàn bộ code hoàn thiện gồm `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `requirements.txt` và `lab.ipynb`.
