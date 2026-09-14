# NPDP: kiến thức và cài đặt cơ bản

## 1. NPDP trong tài liệu này là gì?

NPDP ở đây là cách dự đoán bất định bằng **hồi quy phân vị không giao cắt**
(non-crossing quantile regression). Mô hình học trực tiếp nhiều phân vị của
phân phối target có điều kiện theo input, thay vì giả định một phân phối như Gaussian.

Đây là mô tả phương pháp NPDP đang được sử dụng, không khẳng định mọi phương pháp
mang tên NPDP đều có cùng cách triển khai.

Phương pháp gồm ba thành phần:

- Head trả nhiều phân vị có thứ tự.
- Pinball loss để học các phân vị.
- Median và hai phân vị biên để biểu diễn dự đoán điểm và khoảng dự báo.

Không dùng softmax, không cần dự đoán mean/std, không cần lấy mẫu trọng số Bayesian
hoặc bật dropout nhiều lần khi inference.

## 2. Phân vị và khoảng dự báo

Phân vị mức tau của Y khi biết X=x được định nghĩa:

~~~text
Q_tau(x) = inf { y : F(y | x) >= tau },  0 < tau < 1
~~~

Trong đó F là hàm phân phối tích lũy có điều kiện.

- Q0.5 là trung vị (median), không nhất thiết bằng kỳ vọng (mean).
- Khoảng [Q_(alpha/2), Q_(1-alpha/2)] có mức bao phủ danh nghĩa 1-alpha.
- Các phân vị dự đoán phải tăng theo mức tau.

Chọn danh sách K mức phân vị tăng nghiêm ngặt trong (0,1).
Nếu cần dự đoán điểm bằng median, danh sách phải có 0.5.
Nếu cần khoảng hai phía quanh median, cần các mức ở cả hai phía của 0.5.

Coverage thực tế phụ thuộc dữ liệu và chất lượng học, không tự động bằng mức
danh nghĩa. Với target rời rạc hoặc có khối xác suất tại một giá trị, tỷ lệ
ở mỗi đuôi không nhất thiết bằng alpha/2.

Không thể suy ra đầy đủ phân phối, mean hoặc std chính xác chỉ từ vài phân vị.
Phương pháp này cũng không tự tách bất định epistemic và aleatoric.

## 3. Head không giao cắt

Cho embedding h, một lớp Linear tạo K giá trị raw z.
Chuyển z thành các phân vị:

~~~text
q_0 = z_0
q_j = z_0 + sum(softplus(z_t), t=1..j), j >= 1
softplus(z) = log(1 + exp(z))
~~~

Các increment không âm nên q_(j+1) >= q_j.
Phân vị đầu không bị ép dương: target có thể âm hoặc đang ở không gian chuẩn hóa.

Không dùng softmax vì các phân vị không phải xác suất cần cộng lại bằng 1.
Không sort output để thay cơ chế này. Các increment không chia theo khoảng cách
giữa các mức tau.

## 4. Pinball loss

Với sai số e = y - q_tau:

~~~text
rho_tau(e) = max(tau * e, (tau - 1) * e)
~~~

Khi e >= 0, dự đoán thấp hơn target, hệ số phạt là tau.
Khi e < 0, dự đoán cao hơn target, hệ số phạt là 1-tau.
Sự bất đối xứng này khiến mô hình học phân vị tương ứng thay vì chỉ học một giá trị trung tâm.

Loss trên N nhãn hợp lệ và K phân vị:

~~~text
L = (1 / (N*K)) * sum_i sum_j rho_tau_j(y_i - q_ij)
~~~

Mọi mức phân vị đều tham gia backprop. Không thay pinball bằng MSE nếu muốn
giữ phương pháp học phân vị này.

## 5. Cài đặt PyTorch cốt lõi

Quy ước: embedding [N,D], prediction [N,K], target [N].
K tương ứng đúng thứ tự của quantile_levels.

~~~python
import torch
from torch import nn
import torch.nn.functional as F


class NPDPHead(nn.Module):
    def __init__(self, embedding_dim, quantile_levels):
        super().__init__()
        levels = torch.as_tensor(quantile_levels, dtype=torch.float32)
        if (
            levels.ndim != 1
            or levels.numel() < 2
            or not torch.all((levels > 0) & (levels < 1))
            or not torch.all(levels[1:] > levels[:-1])
        ):
            raise ValueError("Quantile levels must strictly increase within (0, 1)")
        self.register_buffer("quantile_levels", levels)
        self.output = nn.Linear(embedding_dim, levels.numel())

    def forward(self, embedding):
        raw = self.output(embedding)
        first = raw[:, :1]
        increments = F.softplus(raw[:, 1:])
        return torch.cat(
            (first, first + increments.cumsum(dim=-1)), dim=-1
        )


def pinball_loss(prediction, target, quantile_levels):
    tau = torch.as_tensor(
        quantile_levels, device=prediction.device, dtype=prediction.dtype
    )
    error = target.unsqueeze(-1) - prediction
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()
~~~

Đây là head tối thiểu; các lớp trích xuất embedding nằm ngoài phạm vi NPDP.
Danh sách mức phân vị được lưu trong state_dict dưới dạng buffer.
Khi reload, kiến trúc head và số mức phân vị phải tương ứng với checkpoint.

Chỉ đưa nhãn hợp lệ vào loss, chọn chúng trước phép trừ.
Không dùng phép nhân mask để che NaN vì NaN * 0 vẫn là NaN.
Batch không có nhãn hợp lệ phải được xử lý trước khi tính mean loss.
Giá trị target bằng 0 không đồng nghĩa với thiếu nhãn.
Phát hiện prediction không hữu hạn thay vì âm thầm loại chúng.

## 6. Target transform

NPDP có thể học trực tiếp target gốc. Scaler và log không phải điều kiện bắt buộc.

Nếu dùng affine scaler với s > 0:

~~~text
y_scaled = (y - c) / s
q_raw = q_scaled * s + c
~~~

Nếu dùng log1p cho target không âm trước khi scale:

~~~text
t = log1p(y)
y_scaled = (t - c) / s
q_raw = expm1(q_scaled * s + c)
~~~

Inverse theo thứ tự ngược với preprocessing, đúng một lần.
Biến đổi đơn điệu tăng giữ thứ tự phân vị.
Fit scaler trên tập train và lưu transform để inference dùng lại.

Không áp log hoặc clamp không âm một cách mặc định cho target có thể âm.
Nếu bài toán quy định target không âm, có thể chốt clamp về 0 sau inverse;
điều này có thể làm các phân vị bằng nhau.
Không ép riêng lower=0 chỉ để nâng coverage.

## 7. Huấn luyện và suy luận

Huấn luyện:

- Backbone tạo embedding, head trả K phân vị.
- Tính pinball loss với target trong cùng không gian biến đổi.
- Backprop và cập nhật tham số bằng optimizer.
- Theo dõi trên validation độc lập; không dùng test để chọn checkpoint.

Metric chọn best là một quyết định riêng.
Validation pinball đo chất lượng các phân vị; MAE/MSE của median đo chất lượng
dự đoán điểm. Nếu chọn theo MAE/MSE, phải ghi rõ đo ở raw, log hay scaled.
Metric chọn best không thay loss dùng để backprop.

Suy luận:

- Dùng eval() và no_grad().
- Một forward trả K phân vị; không cần nhiều lần lấy mẫu.
- Inverse target nếu có.
- Lấy Q0.5 làm median và các mức đã chọn làm lower/upper.
- Không gọi median là mean hoặc tự tạo std từ khoảng dự báo.

Head quantile phải được huấn luyện. Checkpoint chỉ dự đoán scalar không tự
có uncertainty bằng cách đổi tên output.

## 8. Đánh giá cơ bản

Trên cùng N nhãn hợp lệ, với y, lower và upper trong cùng đơn vị:

~~~text
coverage = count(lower <= y <= upper) / N
above = count(y > upper) / N
below = count(y < lower) / N
MPIW = mean(upper - lower)
~~~

Nhân 100 nếu báo cáo phần trăm. Ba tỷ lệ cộng lại bằng 1;
điểm nằm đúng biên được tính covered.

Đánh giá coverage cùng độ rộng MPIW: khoảng rất rộng có thể coverage cao nhưng
ít hữu ích. MAE/RMSE của median tốt không đảm bảo khoảng phân vị tốt.
Giữ zero hợp lệ, không thay tập đánh giá để làm tăng coverage.

## 9. Kiểm tra cài đặt

- Output đúng shape và đúng thứ tự quantile.
- Không có crossing hoặc giá trị không hữu hạn.
- Pinball bằng 0 khi mọi dự đoán bằng target.
- Gradient đi tới các tham số cần học và hữu hạn.
- Nhãn thiếu được loại đúng, zero hợp lệ vẫn được giữ.
- Inverse transform đúng, không áp hai lần.
- Median được lấy theo mức 0.5, không mặc định lấy phần tử giữa.
- Save/load giữ nguyên mức quantile, preprocessing và dự đoán trong tolerance.
