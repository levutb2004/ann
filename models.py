import torch.nn as nn
import torch
import torch.nn.functional as F

class Model_SI(nn.Module):  # scael ignored
    def __init__(self, in_dim):
        super(Model_SI, self).__init__()
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if train:
            return self.out(x)
        else:
            return x
        
class Model_MC_Dropout(nn.Module):  # scael ignored
    def __init__(self, in_dim):
        super(Model_MC_Dropout, self).__init__()
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid(),nn.Dropout(0.1))
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU(),nn.Dropout(0.1))
        self.out = nn.Sequential(nn.Linear(32, 1))

    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if train:
            return self.out(x)
        else:
            return x
        
class BayesianLinear(nn.Module):
    """Linear layer with a Gaussian mean-field variational posterior over
    weights/bias (Bayes-by-Backprop, reparameterization trick). Each forward
    pass draws a fresh weight sample, so calling the model repeatedly at
    inference already gives posterior predictive samples (no MC Dropout
    trick needed)."""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super(BayesianLinear, self).__init__()
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

    def kl_loss(self):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        return (self._kl_normal(self.weight_mu, weight_sigma)
                + self._kl_normal(self.bias_mu, bias_sigma))

    def _kl_normal(self, mu, sigma):
        # KL( N(mu, sigma^2) || N(0, prior_sigma^2) ), summed over all elements
        prior_sigma = self.prior_sigma
        return (torch.log(prior_sigma / sigma)
                + (sigma ** 2 + mu ** 2) / (2 * prior_sigma ** 2)
                - 0.5).sum()

class Model_EP(nn.Module):
    """Ensemble Prediction: một NN duy nhất, output layer có k 'thành viên'
    (k giá trị) thay vì 1. Huấn luyện bằng loss Winner-Takes-All (WTA) để
    các thành viên tự phân hoá và cùng nhau xấp xỉ phân phối của y_true,
    thay vì tất cả hội tụ về giá trị trung bình. Số lượng k phải chọn
    trước (a priori)."""
 
    def __init__(self, in_dim, k=5):
        super(Model_EP, self).__init__()
        self.k = k
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(32, k))
 
    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if train:
            return self.out(x)  # (batch, k)
        else:
            return x
 

class Model_BNN(nn.Module):
    """Bayesian neural network: same 32-32-1 architecture as Model_SI /
    Model_MC_Dropout, with BayesianLinear layers instead of nn.Linear.
    Train with an ELBO-style loss: data_loss + kl_weight * model.kl_loss()."""

    def __init__(self, in_dim, prior_sigma=1.0):
        super(Model_BNN, self).__init__()
        self.h1 = BayesianLinear(in_dim, 32, prior_sigma)
        self.act1 = nn.Sigmoid()
        self.h2 = BayesianLinear(32, 32, prior_sigma)
        self.act2 = nn.ReLU()
        self.out = BayesianLinear(32, 1, prior_sigma)

    def forward(self, x, mask=None, train=True):
        x = self.act1(self.h1(x))
        x = self.act2(self.h2(x))
        if train:
            return self.out(x)
        else:
            return x

    def kl_loss(self):
        return self.h1.kl_loss() + self.h2.kl_loss() + self.out.kl_loss()


class Model_PDP(nn.Module):
    """Parametric Distributional Prediction: NN dự đoán tham số của một phân
    phối xác suất cho y_true, thay vì dự đoán trực tiếp y_true. Loại phân
    phối chọn trước (a priori): mặc định Gaussian (mu, sigma); có thể mở
    rộng sang phân phối 4 tham số kiểu GAMLSS (mu, sigma, nu, tau — vd.
    Student-t lệch) qua distribution='student_t'.
    Huấn luyện bằng loss tối đa hoá log-likelihood (NLL)."""

    def __init__(self, in_dim, distribution='gaussian'):
        super(Model_PDP, self).__init__()
        assert distribution in ('gaussian', 'student_t')
        self.distribution = distribution
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        # custom output layer: mu tự do, các tham số còn lại > 0 qua softplus
        self.mu_head = nn.Linear(32, 1)
        self.sigma_head = nn.Linear(32, 1)
        if distribution == 'student_t':
            self.nu_head = nn.Linear(32, 1)   # bậc tự do (> 2 để có variance hữu hạn)
            self.tau_head = nn.Linear(32, 1)  # scale phụ

    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if not train:
            return x
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x)) + 1e-6
        if self.distribution == 'gaussian':
            return torch.cat([mu, sigma], dim=-1)          # (batch, 2)
        nu = F.softplus(self.nu_head(x)) + 2.0 + 1e-6       # nu > 2
        tau = F.softplus(self.tau_head(x)) + 1e-6
        return torch.cat([mu, sigma, nu, tau], dim=-1)      # (batch, 4)

    def nll_loss(self, params, target):
        """Negative log-likelihood. target: (batch,) hoặc (batch,1)."""
        target = target.reshape(-1, 1)
        if self.distribution == 'gaussian':
            mu, sigma = params[:, :1], params[:, 1:2]
            dist = torch.distributions.Normal(mu, sigma)
        else:
            mu, sigma, nu, tau = (params[:, :1], params[:, 1:2],
                                   params[:, 2:3], params[:, 3:4])
            dist = torch.distributions.StudentT(df=nu, loc=mu, scale=tau + sigma)
        return -dist.log_prob(target).squeeze(-1)


class Model_NPDP(nn.Module):
    """Non-Parametric Distributional Prediction: NN dự đoán trực tiếp một
    tập hợp thống kê tóm tắt của y_true (m quantiles) thay vì tham số phân
    phối. Kiến trúc tránh 'quantile crossing': quantile thấp nhất tự do,
    các quantile sau = tích luỹ (cumsum) của phần chênh lệch dương (softplus)
    so với quantile trước — đảm bảo không bao giờ đảo thứ tự.
    Huấn luyện bằng Pinball (Quantile) Loss."""

    def __init__(self, in_dim, quantile_levels=(0.025, 0.25, 0.5, 0.75, 0.975)):
        super(Model_NPDP, self).__init__()
        self.register_buffer(
            'quantile_levels',
            torch.tensor(quantile_levels, dtype=torch.float32))
        self.m = len(quantile_levels)
        self.h1 = nn.Sequential(nn.Linear(in_dim, 32), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        # custom output layer: chống crossover giữa các quantiles
        self.q0_head = nn.Linear(32, 1)
        self.delta_head = nn.Linear(32, self.m - 1)

    def forward(self, x, mask=None, train=True):
        x = self.h1(x)
        x = self.h2(x)
        if not train:
            return x
        q0 = self.q0_head(x)                       # (batch, 1)
        deltas = F.softplus(self.delta_head(x))     # (batch, m-1), luôn > 0
        return torch.cat([q0, deltas], dim=-1).cumsum(dim=-1)  # (batch, m), tăng dần

    def pinball_loss(self, preds, target):
        """preds: (batch, m); target: (batch,) hoặc (batch, 1)."""
        target = target.reshape(-1, 1)
        taus = self.quantile_levels.to(preds.device)
        errors = target - preds
        loss = torch.maximum(taus * errors, (taus - 1) * errors)
        return loss.mean()