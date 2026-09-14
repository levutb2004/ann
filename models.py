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
