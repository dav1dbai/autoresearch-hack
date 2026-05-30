"""Reference task: one batch-1 decode step of a small Llama-style transformer.

This is the regime where megakernels win: batch size 1, a single new token,
memory-bound (the forward is dominated by streaming weights from HBM, not by
flops). A stock PyTorch forward launches ~dozens of kernels per layer
(RMSNorm, QKV proj, RoPE, attention, O proj, SwiGLU MLP), each paying launch
and teardown overhead. The agent must collapse the whole forward into a single
fused / persistent (mega)kernel that is numerically equivalent and faster.

Drop-in contract (KernelBench style): produce `ModelNew(*get_init_inputs())`
with the same `forward(x, k_cache, v_cache)` signature. Weights are passed in
via `get_init_inputs()` so the reference and the submission share *identical*
parameters (otherwise random init would diverge and the allclose gate would
always fail).

Shapes are small enough for a ~20-minute rollout but large enough that
per-kernel launch bubbles dominate at batch 1. The KV cache is assumed
pre-rotated (standard during decode); we attend over the cached positions and
do not append the new token, keeping the op a clean pure function.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- model dims (small but launch-bubble dominated at batch 1) ---
H = 2048          # hidden size
N_LAYERS = 4
N_HEADS = 16
HEAD_DIM = H // N_HEADS  # 128
INTER = 4096      # SwiGLU intermediate
S = 512           # KV cache length (context already seen)
VOCAB = 8192
DTYPE = torch.float16
ROPE_THETA = 10000.0
WEIGHT_SEED = 1234


def _make_weights() -> dict:
    """Deterministic weights so Model and ModelNew get identical params.

    Built on a fixed generator seed and scaled down so a 4-layer fp16 forward
    stays numerically well-behaved for the allclose gate.
    """
    g = torch.Generator().manual_seed(WEIGHT_SEED)

    def randn(*shape, scale):
        return (torch.randn(*shape, generator=g) * scale).to(DTYPE)

    layers = []
    for _ in range(N_LAYERS):
        layers.append(
            {
                "ln1_w": randn(H, scale=0.0) + 1.0,
                "wq": randn(H, H, scale=H ** -0.5),
                "wk": randn(H, H, scale=H ** -0.5),
                "wv": randn(H, H, scale=H ** -0.5),
                "wo": randn(H, H, scale=H ** -0.5),
                "ln2_w": randn(H, scale=0.0) + 1.0,
                "w_gate": randn(INTER, H, scale=H ** -0.5),
                "w_up": randn(INTER, H, scale=H ** -0.5),
                "w_down": randn(H, INTER, scale=INTER ** -0.5),
            }
        )
    return {
        "layers": layers,
        "norm_w": randn(H, scale=0.0) + 1.0,
        "lm_head": randn(VOCAB, H, scale=H ** -0.5),
    }


def _rope_cos_sin(pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    ang = pos * inv_freq                       # [HEAD_DIM/2]
    ang = torch.cat([ang, ang], dim=-1)        # [HEAD_DIM]
    return ang.cos().to(DTYPE), ang.sin().to(DTYPE)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v.to(x.dtype)) * w


class Model(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        self.w = weights
        cos, sin = _rope_cos_sin(S)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _to(self):
        dev = self.cos.device
        for layer in self.w["layers"]:
            for k, t in layer.items():
                layer[k] = t.to(dev)
        self.w["norm_w"] = self.w["norm_w"].to(dev)
        self.w["lm_head"] = self.w["lm_head"].to(dev)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        # x: [1, 1, H]   k_cache/v_cache: [N_LAYERS, 1, N_HEADS, S, HEAD_DIM]
        self._to()
        cos, sin = self.cos, self.sin
        h = x
        for i, layer in enumerate(self.w["layers"]):
            n = _rmsnorm(h, layer["ln1_w"])
            q = (n @ layer["wq"].t()).view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)  # [1,H,1,hd]
            q = q * cos + _rotate_half(q) * sin
            k = k_cache[i]                                                            # [1,H,S,hd]
            v = v_cache[i]
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)  # [1,H,1,S]
            attn = F.softmax(scores, dim=-1) @ v.float()                             # [1,H,1,hd]
            attn = attn.to(DTYPE).transpose(1, 2).reshape(1, 1, H)
            h = h + attn @ layer["wo"].t()

            n2 = _rmsnorm(h, layer["ln2_w"])
            gate = n2 @ layer["w_gate"].t()
            up = n2 @ layer["w_up"].t()
            mlp = (F.silu(gate.float()).to(DTYPE) * up) @ layer["w_down"].t()
            h = h + mlp

        h = _rmsnorm(h, self.w["norm_w"])
        return h @ self.w["lm_head"].t()                                             # [1,1,VOCAB]


def get_inputs():
    x = torch.randn(1, 1, H, dtype=DTYPE)
    k_cache = torch.randn(N_LAYERS, 1, N_HEADS, S, HEAD_DIM, dtype=DTYPE) * (HEAD_DIM ** -0.5)
    v_cache = torch.randn(N_LAYERS, 1, N_HEADS, S, HEAD_DIM, dtype=DTYPE) * (HEAD_DIM ** -0.5)
    return [x, k_cache, v_cache]


def get_init_inputs():
    return [_make_weights()]
