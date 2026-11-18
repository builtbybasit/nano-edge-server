"""nano-edge-tts: a 1.75M-parameter decoder for Kokoro-82M. Inference only.

Kokoro's text side is unchanged. A mapper (0.96M parameters, 80 frames/s) and a renderer network (0.79M, 160 frames/s) set
per-bin gains on a pulse train and on white noise; two inverse STFTs make the waveform. No network runs at audio rate.

    from nano_edge_tts import load
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', model=load())
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 24000
HERE = os.path.dirname(os.path.abspath(__file__))


class STFT(nn.Module):
    """STFT and inverse as matrix products (Hann window, centred, reflect padding)."""
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop, self.bins = n_fft, hop, n_fft // 2 + 1
        w = torch.hann_window(n_fft, periodic=True)
        k = torch.arange(self.bins)[:, None] * torch.arange(n_fft)[None] * 2 * math.pi / n_fft
        self.register_buffer('fwd', torch.cat([torch.cos(k), -torch.sin(k)]) * w, persistent=False)
        scale = torch.full((self.bins, 1), 2.0)
        scale[0] = scale[-1] = 1.0
        self.register_buffer('inv', (torch.cat([torch.cos(k) * scale, -torch.sin(k) * scale]) / n_fft * w).T.contiguous(), persistent=False)
        self.register_buffer('win_sq', (w ** 2)[None, :, None].clone(), persistent=False)

    def forward(self, x):
        x = F.pad(x[:, None], (self.n_fft // 2, self.n_fft // 2), mode='reflect')[:, 0]
        y = self.fwd @ x.unfold(-1, self.n_fft, self.hop).transpose(1, 2)
        return y[:, :self.bins], y[:, self.bins:]

    def inverse(self, re, im, length):
        frames = self.inv @ torch.cat([re, im], 1)
        T = re.shape[-1]
        size = (1, (T - 1) * self.hop + self.n_fft)
        fold = lambda f: F.fold(f, size, (1, self.n_fft), stride=(1, self.hop)).flatten(1)
        y = fold(frames) / fold(self.win_sq.expand(1, -1, T)).clamp_min(1e-8)
        return y[:, self.n_fft // 2:self.n_fft // 2 + length]


def _dirichlet(theta, K):
    """Sum of cos(k theta) for k = 1..K in closed form: a band-limited pulse per cycle at a cost independent of K."""
    half = torch.sin(theta / 2)
    near = half.abs() < 1e-3
    d = torch.sin((K + 0.5) * theta) / (2 * torch.where(near, torch.ones_like(half), half)) - 0.5
    return torch.where(near, K, d)


def pulse_train(f0, hop=300):
    """f0 [B, T] at 80 frames/s -> [B, T * hop]: every harmonic of f0 below Nyquist, zero where unvoiced."""
    f = F.interpolate(f0[:, None], scale_factor=hop, mode='linear', align_corners=False)[:, 0]
    voiced = f > 10.0
    f = torch.where(voiced, f, torch.full_like(f, 100.0))
    phase = torch.cumsum(f / SR, 1)
    theta = 2 * math.pi * (phase - torch.floor(phase))
    return _dirichlet(theta, torch.floor(SR / 2 / f).clamp(1, 240)) * voiced * 0.1


def creak_train(f0, rate, hop=300, jitter=300.0, bw=25.0):
    """Slow irregular pulses (about `rate` Hz, fixed seeded jitter) where the pitch track is unvoiced, for creaky voice."""
    f = F.interpolate(f0[:, None], scale_factor=hop, mode='linear', align_corners=False)[:, 0]
    un = f <= 10.0
    B, L = f.shape
    X = torch.fft.rfft(torch.randn(B, L, generator=torch.Generator().manual_seed(1)))
    X[:, int(bw * L / SR) + 1:] = 0
    x = torch.fft.irfft(X, L)
    cents = (x * jitter / (x.pow(2).mean(1, keepdim=True).sqrt() + 1e-9)).to(f.device)
    fc = rate * torch.exp2(cents / 1200)
    phase = torch.cumsum(fc * un / SR, 1)
    theta = 2 * math.pi * (phase - torch.floor(phase))
    return _dirichlet(theta, torch.floor(SR / 2 / fc).clamp(1, 240)) * un * 0.1


class AdaLN(nn.Module):
    """Layer norm whose gain and bias come from the voice vector."""
    def __init__(self, dim, style_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc = nn.Linear(style_dim, 2 * dim)

    def forward(self, x, s):
        g, b = self.fc(s)[:, None].chunk(2, -1)
        return self.norm(x) * (1 + g) + b


class AdaIN(AdaLN):
    """The same with statistics over the whole utterance, as in Kokoro's decoder."""
    def forward(self, x, s):
        g, b = self.fc(s)[:, None].chunk(2, -1)
        x = (x - x.mean(1, keepdim=True)) * torch.rsqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        return x * (1 + g) + b


class Block(nn.Module):
    """ConvNeXt block: depthwise convolution, voice-conditioned norm, inverted bottleneck."""
    def __init__(self, dim, style_dim, kernel=7, mult=3, instance=False):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = (AdaIN if instance else AdaLN)(dim, style_dim)
        self.up, self.down = nn.Linear(dim, mult * dim), nn.Linear(mult * dim, dim)
        self.scale = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x, s):
        h = self.down(F.gelu(self.up(self.norm(self.dw(x).transpose(1, 2), s))))
        return x + (self.scale * h).transpose(1, 2)


def _prosody(F0, N):
    return torch.stack([torch.log1p(F0.clamp_min(0)) / 6, (F0 > 10).float(), N / 4], 1)


class Mapper(nn.Module):
    """Kokoro's decoder inputs -> a 512-channel description of the sound at 80 frames/s."""
    def __init__(self, dim, depth, kernel, feat_dim, out_dim, style_dim):
        super().__init__()
        self.inp = nn.Conv1d(feat_dim + 3, dim, 1)
        self.blocks = nn.ModuleList([Block(dim, style_dim, kernel, instance=True) for _ in range(depth)])
        self.norm = AdaLN(dim, style_dim)
        self.head = nn.Linear(dim, out_dim)
        self.register_buffer('mean', torch.zeros(out_dim))
        self.register_buffer('std', torch.ones(out_dim))

    def forward(self, asr, F0, N, s):
        x = self.inp(torch.cat([F.interpolate(asr, size=F0.shape[-1], mode='nearest'), _prosody(F0, N)], 1))
        for b in self.blocks:
            x = b(x, s)
        return (self.head(self.norm(x.transpose(1, 2), s)) * self.std + self.mean).transpose(1, 2)


class Renderer(nn.Module):
    """Per frame (160/s): a gain and a phase rotation for each of 301 bins of the pulse train's STFT (25 ms window), and a gain
    for each of 61 bins of the noise's STFT (5 ms window) at five sub-steps. Two inverse STFTs, summed."""
    def __init__(self, n_fft, hop, dim, depth, feat_dim, style_dim, noise_nfft, creak):
        super().__init__()
        self.creak = creak
        self.stft, self.nstft = STFT(n_fft, hop), STFT(noise_nfft, noise_nfft // 4)
        self.sub = hop // (noise_nfft // 4)
        self.inp = nn.Conv1d(feat_dim + self.stft.bins + 3, dim, 1)
        self.blocks = nn.ModuleList([Block(dim, style_dim) for _ in range(depth)])
        self.norm = AdaLN(dim, style_dim)
        self.head = nn.Linear(dim, 2 * self.stft.bins + self.sub * self.nstft.bins)
        w = torch.hann_window(n_fft)
        self.h_scale, self.n_scale = float(0.05 * w.sum()), float(0.1 * w.pow(2).sum().sqrt())

    def forward(self, feat, F0, N, s):
        L, b = F0.shape[-1] * 300, self.stft.bins
        Hr, Hi = self.stft(pulse_train(F0))
        Nr, Ni = self.nstft(torch.randn(F0.shape[0], L, device=F0.device) * 0.1)
        T = Hr.shape[-1]
        logmag = 0.5 * torch.log(Hr ** 2 + Hi ** 2 + 1e-4) - math.log(self.h_scale)
        if self.creak:
            Cr, Ci = self.stft(creak_train(F0, self.creak))
            Hr, Hi = Hr + Cr, Hi + Ci
        x = torch.cat([F.interpolate(feat, size=T, mode='nearest'), F.interpolate(_prosody(F0, N), size=T, mode='linear'), logmag / 4], 1)
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x, s)
        h = self.head(self.norm(x.transpose(1, 2), s)).transpose(1, 2)
        gh = torch.exp(h[:, :b].clamp(max=8) - 3) / self.h_scale
        gn = torch.exp(h[:, 2 * b:].clamp(max=8) - 3) / self.n_scale
        c, sn = torch.cos(h[:, b:2 * b]), torch.sin(h[:, b:2 * b])
        Hr, Hi = Hr * c - Hi * sn, Hr * sn + Hi * c
        g = gn.view(-1, self.sub, self.nstft.bins, T).permute(0, 2, 3, 1).reshape(-1, self.nstft.bins, T * self.sub)
        g = g[..., self.sub // 2:self.sub // 2 + Nr.shape[-1]]
        return self.stft.inverse(gh * Hr, gh * Hi, L) + self.nstft.inverse(g * Nr, g * Ni, L)


class NanoDecoder(nn.Module):
    """Drop-in for Kokoro's decoder: forward(asr, F0, N, style) -> waveform [B, samples] at 24 kHz."""
    def __init__(self, cfg):
        super().__init__()
        self.mapper, self.renderer = Mapper(**cfg['mapper']), Renderer(**cfg['renderer'])

    def forward(self, asr, F0, N, s):
        return self.renderer(self.mapper(asr, F0, N, s), F0, N, s)


def load_decoder(folder=HERE):
    from safetensors.torch import load_file
    with open(os.path.join(folder, 'config.json')) as f:
        dec = NanoDecoder(json.load(f))
    dec.load_state_dict(load_file(os.path.join(folder, 'nano-edge-tts.safetensors')))
    return dec.eval()


def load(folder=HERE, repo_id='hexgrad/Kokoro-82M'):
    """Kokoro-82M with its decoder replaced. Use it wherever a KModel is expected, e.g. KPipeline(lang_code='a', model=load())."""
    from kokoro import KModel
    model = KModel(repo_id=repo_id)
    model.decoder = load_decoder(folder)
    return model.eval()
