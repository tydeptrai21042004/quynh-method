from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def require_gpytorch():
    try:
        import gpytorch  # noqa: F401
        return gpytorch
    except ImportError as e:
        raise RuntimeError(
            "chen2025_svdkl requires gpytorch. Install paper dependencies with: pip install -e '.[paper]'"
        ) from e


def fit_svdkl(Xtr, ytr, Xv, yv, *, hidden=64, dropout=.1, feature_dim=16,
              inducing=128, lr=1e-3, weight_decay=1e-4, batch_size=256,
              epochs=40, patience=8, device=None, optimizer="adamw"):
    gpytorch = require_gpytorch()
    from .models import SpatioTemporalCNN
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class GPRegressionLayer(gpytorch.models.ApproximateGP):
        def __init__(self):
            points = torch.randn(inducing, feature_dim)
            q = gpytorch.variational.CholeskyVariationalDistribution(inducing)
            strategy = gpytorch.variational.VariationalStrategy(self, points, q, learn_inducing_locations=True)
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    class SVDKL(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.feature = SpatioTemporalCNN(d, hidden=hidden, feature_dim=feature_dim, dropout=dropout)
            self.gp = GPRegressionLayer()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        def forward(self, x):
            return self.gp(self.feature(x))

    model = SVDKL(Xtr.shape[-1]).to(device)
    likelihood = model.likelihood.to(device)
    opt_cls = torch.optim.Adam if str(optimizer).lower() == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp, num_data=len(Xtr))
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
    xv = torch.from_numpy(Xv).to(device); yv_t = torch.from_numpy(yv).to(device)
    best = None; best_loss = float("inf"); bad = 0
    for _ in range(epochs):
        model.train(); likelihood.train()
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-4):
                loss = -mll(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(xv))
            val = torch.mean((pred.mean - yv_t) ** 2).item()
        if val < best_loss - 1e-7:
            best_loss = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return model


def predict_svdkl(model, X, *, batch_size=1024, device=None):
    gpytorch = require_gpytorch()
    device = device or next(model.parameters()).device
    model.eval(); model.likelihood.eval()
    means = []; stds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            dist = model.likelihood(model(xb))
            means.append(dist.mean.cpu().numpy())
            stds.append(dist.stddev.cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)
