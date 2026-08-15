"""FNO baseline — Fourier Neural Operator forward surrogate.

Maps a (decoded) parameter field to the observation vector via spectral
convolution layers. Supports 1D fields (Burgers, Helmholtz) and 2D fields
(Darcy); the spatial resolution and number of Fourier modes can be varied for
the operator-scaling study.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism.models.base import _Scaler
from ._common import resolve_device


class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, 2))

    def forward(self, x):                                   # (b, C, S)
        B, C, S = x.shape
        xft = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xft.shape[-1])
        wt = torch.view_as_complex(self.w)[:, :, :m]
        out = torch.zeros(B, self.w.shape[1], xft.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xft[..., :m], wt)
        return torch.fft.irfft(out, n=S, dim=-1)


class _SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.w = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))

    def forward(self, x):                                   # (b, C, H, W)
        B, C, H, W = x.shape
        xft = torch.fft.rfft2(x, dim=(-2, -1))
        m = min(self.modes, H, xft.shape[-1])
        wt = torch.view_as_complex(self.w)[:, :, :m, :m]
        out = torch.zeros(B, self.w.shape[1], H, xft.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bihw,iohw->bohw", xft[:, :, :m, :m], wt)
        return torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))


class _FNO1d(nn.Module):
    def __init__(self, modes, width, n_layers, out_dim):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.convs = nn.ModuleList([_SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):                                   # (b, S)
        x = self.fc0(x.unsqueeze(-1)).permute(0, 2, 1)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.mean(dim=-1)
        return self.fc2(F.gelu(self.fc1(x)))


class _FNO2d(nn.Module):
    def __init__(self, modes, width, n_layers, out_dim):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.convs = nn.ModuleList([_SpectralConv2d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):                                   # (b, H, W)
        x = self.fc0(x.unsqueeze(-1)).permute(0, 3, 1, 2)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.mean(dim=(-2, -1))
        return self.fc2(F.gelu(self.fc1(x)))


class FNO:
    """Forward operator surrogate. fit(fields, Y) / predict(fields)."""

    def __init__(self, modes=8, width=32, n_layers=3, lr=1e-3, epochs=200,
                 batch_size=64, device="auto", seed=0, verbose=False):
        self.modes = modes; self.width = width; self.n_layers = n_layers
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.device = device; self.seed = seed; self.verbose = verbose

    def fit(self, fields, Y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)
        fields = np.asarray(fields, float); Y = np.asarray(Y, float)
        self.fmu_, self.fsd_ = fields.mean(), fields.std() + 1e-8
        self.sy_ = _Scaler().fit(Y)
        Fn = torch.as_tensor((fields - self.fmu_) / self.fsd_, dtype=torch.float32, device=self.device_)
        Yt = torch.as_tensor(self.sy_.transform(Y), dtype=torch.float32, device=self.device_)
        ndim = fields.ndim - 1
        out_dim = Y.shape[1]
        modes = min(self.modes, fields.shape[-1] // 2)
        self.net_ = (_FNO1d if ndim == 1 else _FNO2d)(modes, self.width, self.n_layers, out_dim).to(self.device_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        n = Fn.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = ((self.net_(Fn[idx]) - Yt[idx]) ** 2).mean()
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, fields):
        fields = np.asarray(fields, float)
        Fn = torch.as_tensor((fields - self.fmu_) / self.fsd_, dtype=torch.float32, device=self.device_)
        return self.sy_.inverse_transform(self.net_(Fn).cpu().numpy())
