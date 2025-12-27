# CHƯƠNG 3. THỰC NGHIỆM VÀ KẾT QUẢ

## 3.1. Môi trường thực nghiệm

### 3.1.1. Cấu hình phần cứng và phần mềm

| Thành phần | Thông số kỹ thuật |
|------------|-------------------|
| CPU | Intel Core (hoặc AMD Ryzen) |
| GPU | NVIDIA GPU với CUDA support |
| RAM | 16GB+ |
| Python | 3.8+ |
| Framework | PyTorch 2.0+ |
| Hệ điều hành | Ubuntu Linux / Windows |

### 3.1.2. Thư viện sử dụng

- **PyTorch**: Framework deep learning chính
- **scikit-learn**: Các metrics đánh giá và preprocessing
- **tree-sitter**: Phân tích cú pháp mã nguồn C
- **NumPy/Pandas**: Xử lý dữ liệu

## 3.2. Bộ dữ liệu thực nghiệm

### 3.2.1. Tập dữ liệu Devign

Nghiên cứu sử dụng tập dữ liệu **Devign** - một bộ dữ liệu chuẩn cho bài toán phát hiện lỗ hổng bảo mật trong mã nguồn C, được chia như sau:

| Tập dữ liệu | Số lượng mẫu | Tỷ lệ |
|-------------|--------------|-------|
| Training | 21,817 | 80% |
| Validation | 2,728 | 10% |
| Test | 2,726 | 10% |
| **Tổng cộng** | **27,271** | 100% |

### 3.2.2. Tiền xử lý dữ liệu

Quá trình tiền xử lý bao gồm:

1. **Tokenization**: Mã nguồn được tokenize với vocabulary size = 238 tokens
2. **Program Slicing**: 
   - Backward slicing với window_size = 15, max_depth = 5
   - Forward slicing với window_size = 15, max_depth = 5
   - Mỗi mẫu tối đa 6 slices, mỗi slice tối đa 256 tokens
3. **Padding/Truncation**: max_len = 512 tokens

### 3.2.3. Đặc trưng lỗ hổng (Vulnerability Features)

Mô hình sử dụng 26 đặc trưng tĩnh được trích xuất từ mã nguồn:

| Nhóm đặc trưng | Các đặc trưng |
|----------------|---------------|
| **Thống kê cơ bản** | loc, stmt_count |
| **Dangerous calls** | dangerous_call_count, dangerous_call_without_check_count, dangerous_call_without_check_ratio |
| **Pointer operations** | pointer_deref_count, pointer_deref_without_null_check_count, pointer_deref_without_null_check_ratio |
| **Array access** | array_access_count, array_access_without_bounds_check_count, array_access_without_bounds_check_ratio |
| **Memory management** | malloc_count, malloc_without_free_count, malloc_without_free_ratio, free_count, free_without_null_check_count, free_without_null_check_ratio |
| **Return value check** | unchecked_return_value_count, unchecked_return_value_ratio |
| **Defense patterns** | null_check_count, bounds_check_count, defense_ratio |
| **Density metrics** | dangerous_call_density, pointer_deref_density, array_access_density, null_check_density |

## 3.3. Cấu hình mô hình

### 3.3.1. Kiến trúc BiGRU

Mô hình sử dụng kiến trúc **Bidirectional GRU** kết hợp với các đặc trưng tĩnh:

- **Embedding layer**: vocab_size = 238
- **BiGRU layers**: Xử lý chuỗi token theo cả hai chiều
- **Attention mechanism**: Tập trung vào các phần quan trọng của code
- **Feature fusion**: Kết hợp token embeddings với 26 vulnerability features
- **Classification head**: Fully-connected layers với sigmoid output

### 3.3.2. Ensemble với Multiple Seeds

Để tăng độ ổn định và chính xác, mô hình được huấn luyện với **3 random seeds** khác nhau:
- Seed 42
- Seed 1042  
- Seed 2042

Kết quả cuối cùng là **ensemble average** của 3 mô hình.

## 3.4. Quá trình huấn luyện

### 3.4.1. Training History

Mô hình được huấn luyện trong khoảng **18 epochs** với các quan sát sau:

#### Loss Function
![Training & Validation Loss](../images/training_history.png)

- **Training Loss**: Giảm đều từ ~0.080 xuống ~0.055 qua các epochs
- **Validation Loss**: Ổn định quanh mức 0.070-0.071, cho thấy mô hình không bị overfitting nghiêm trọng

#### F1 Score
- **Training F1**: Tăng từ ~0.68 lên ~0.78
- **Validation F1**: Dao động trong khoảng 0.70-0.75
- **Best epoch**: Epoch 17 với Val OptF1 đạt ~0.75

#### AUC (Area Under ROC Curve)
- **Training AUC**: Tăng từ ~0.78 lên ~0.92
- **Validation AUC**: Ổn định và tăng dần từ ~0.82 lên ~0.87

#### Precision, Recall & F1 trên Validation
- **Precision**: Dao động 0.68-0.82, có xu hướng tăng
- **Recall**: Dao động 0.68-0.78
- **Val OptF1**: Ổn định quanh 0.74-0.76

### 3.4.2. Nhận xét về quá trình huấn luyện

1. **Convergence**: Mô hình hội tụ tốt, loss giảm đều đặn
2. **Generalization**: Khoảng cách Train-Val loss nhỏ, không overfitting
3. **Early stopping**: Best model được lưu tại epoch 17

## 3.5. Kết quả trên tập Test

### 3.5.1. Tổng quan metrics

Kết quả đánh giá trên tập test với **Ensemble + Calibration** (threshold t=0.36):

| Metric | Giá trị |
|--------|---------|
| **F1 Score (t=0.5)** | 0.7517 |
| **F1 Score (optimal, t=0.36)** | 0.7721 |
| **Precision** | 0.8327 |
| **Recall** | 0.7196 |
| **AUC-ROC** | 0.8858 |
| **AUC-PR** | 0.8956 |

### 3.5.2. Đường cong ROC và Precision-Recall

![ROC and PR Curves](../images/roc_pr_curves.png)

**ROC Curve:**
- AUC = 0.8858 cho thấy mô hình có khả năng phân biệt tốt giữa mã có lỗ hổng và không có lỗ hổng
- Đường cong nằm xa đường chéo (random classifier), thể hiện hiệu suất tốt

**Precision-Recall Curve:**
- AUC-PR = 0.8956
- Tại threshold t=0.36, Precision = 0.8327
- Mô hình duy trì precision cao ngay cả khi recall tăng

### 3.5.3. Confusion Matrix

![Confusion Matrix](../images/confusion_matrix.png)

#### Với threshold mặc định t=0.5:
| | Predicted Non-Vuln | Predicted Vuln |
|---|---|---|
| **Actual Non-Vuln** | 1386 (TN) | 88 (FP) |
| **Actual Vuln** | 445 (FN) | 807 (TP) |

- **F1 = 0.7517**
- False Positive Rate thấp: 88/(1386+88) = 5.97%
- False Negative Rate: 445/(445+807) = 35.54%

#### Với threshold tối ưu t=0.36:
| | Predicted Non-Vuln | Predicted Vuln |
|---|---|---|
| **Actual Non-Vuln** | 1293 (TN) | 181 (FP) |
| **Actual Vuln** | 351 (FN) | 901 (TP) |

- **F1 = 0.7510** (F1.5 metric)
- Recall cải thiện: 901/(351+901) = 71.96%
- Trade-off: FP tăng nhưng phát hiện được nhiều lỗ hổng hơn

### 3.5.4. Phân tích threshold

Việc điều chỉnh threshold từ 0.5 xuống 0.36 giúp:
- **Tăng Recall**: Từ 64.46% lên 71.96% (+7.5%)
- **Tăng True Positives**: Từ 807 lên 901 (+94 samples)
- **Giảm False Negatives**: Từ 445 xuống 351 (-94 samples)
- **Chi phí**: False Positives tăng từ 88 lên 181

Trong bối cảnh phát hiện lỗ hổng bảo mật, việc ưu tiên Recall (phát hiện nhiều lỗ hổng hơn) thường quan trọng hơn Precision vì bỏ sót lỗ hổng có thể gây hậu quả nghiêm trọng.

## 3.6. So sánh với các nghiên cứu liên quan

| Phương pháp | F1 Score | AUC-ROC |
|-------------|----------|---------|
| Devign (Original) | 0.62 | - |
| SySeVR | 0.68 | 0.85 |
| **BiGRU + Vuln Features (Ours)** | **0.7721** | **0.8858** |

Mô hình đề xuất đạt kết quả cạnh tranh so với các phương pháp state-of-the-art, với ưu điểm:
- Kết hợp thông tin ngữ nghĩa từ BiGRU và đặc trưng tĩnh về lỗ hổng
- Sử dụng program slicing để tập trung vào các phần code liên quan
- Ensemble nhiều seeds tăng độ ổn định

## 3.7. Kết luận chương

Qua quá trình thực nghiệm, mô hình BiGRU kết hợp với các đặc trưng lỗ hổng đã cho thấy:

1. **Hiệu quả cao**: Đạt F1 = 0.7721 và AUC = 0.8858 trên tập test
2. **Ổn định**: Ensemble 3 seeds giúp giảm variance
3. **Precision cao**: 83.27% giúp giảm false alarms trong thực tế
4. **Khả năng điều chỉnh**: Threshold calibration cho phép cân bằng precision-recall theo nhu cầu

Kết quả cho thấy phương pháp đề xuất có tiềm năng ứng dụng thực tế trong việc hỗ trợ phát hiện lỗ hổng bảo mật tự động cho mã nguồn C.
