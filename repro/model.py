"""Paper-faithful AP-MDM encoder Transformer.

This is a portable re-implementation of the upstream backbone in
``train/models/dit.py`` with **the same module structure, the same parameter
tensors, the same initialisation and the same forward semantics**, differing
only in the attention/rotary kernels:

* upstream calls ``flash_attn.flash_attn_varlen_qkvpacked_func`` and
  ``flash_attn.layers.rotary.apply_rotary_emb_qkv_``;
* this module calls ``torch.nn.functional.scaled_dot_product_attention`` and an
  explicit rotate-half implementation of the identical rotary convention.

FlashAttention is not installed on the target CUDA/Torch stack (torch 2.7.0 /
cu12.8) and its prebuilt wheels do not cover it, so the SDPA path is the one
that runs.  The kernels are *not* claimed to be bitwise identical; the test
suite pins the rotary convention and, when ``flash_attn`` is importable, checks
backend agreement within a declared tolerance.

Heads (paper, ``method.tex`` eq. 1-4):

* ``unmask``  - softmax over the vocabulary,
* ``remask``  - one binary logit per position,
* ``insert``  - one binary logit per position (upstream name: *expansion*),
* ``delete``  - one binary logit per position (upstream name: *contraction*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from repro.config import ModelSpec


class LayerNorm(nn.Module):
    """Upstream's LayerNorm: elementwise weight, no bias, computed in fp32."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = F.layer_norm(x.float(), [self.dim])
        return normed.to(x.dtype) * self.weight[None, None, :].to(x.dtype)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP (identical to upstream)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Rotary(nn.Module):
    """RoPE frequencies matching upstream's ``Rotary`` (base 10000, full head dim)."""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotate-half RoPE over the full head dimension.

    ``x`` is ``(batch, seq, heads, head_dim)``; ``cos``/``sin`` are
    ``(seq, head_dim // 2)``.  This is the convention that
    ``flash_attn.layers.rotary.apply_rotary_emb_qkv_`` implements when it is
    handed half-width cos/sin, which is exactly how upstream calls it.
    """
    cos_full = torch.cat((cos, cos), dim=-1)[None, :, None, :]
    sin_full = torch.cat((sin, sin), dim=-1)[None, :, None, :]
    return x * cos_full + rotate_half(x) * sin_full


class DDiTBlock(nn.Module):
    """Encoder block with adaLN modulation, matching upstream tensor-for-tensor."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"hidden size {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        rotary: tuple[torch.Tensor, torch.Tensor],
        c: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        manual_attention: bool = False,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        h = self._modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(h).view(batch, seq, 3, self.n_heads, self.head_dim)
        cos, sin = rotary
        q = apply_rotary(qkv[:, :, 0], cos.to(qkv.dtype), sin.to(qkv.dtype))
        k = apply_rotary(qkv[:, :, 1], cos.to(qkv.dtype), sin.to(qkv.dtype))
        v = qkv[:, :, 2]

        sdpa_mask = (
            attention_mask[:, None, None, :]
            if attention_mask is not None
            else None
        )
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        v_heads = v.transpose(1, 2)
        if manual_attention:
            # Energy-gradient objectives differentiate through the input
            # gradient.  PyTorch's fused SDPA kernels do not universally
            # implement that second derivative, whereas the mathematically
            # equivalent explicit attention does.
            scores = q_heads @ k_heads.transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)
            if sdpa_mask is not None:
                scores = scores.masked_fill(~sdpa_mask, -torch.inf)
            attn = scores.softmax(dim=-1) @ v_heads
        else:
            attn = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                attn_mask=sdpa_mask,
                is_causal=False,
            )
        attn = attn.transpose(1, 2).reshape(batch, seq, -1)
        x = x_skip + gate_msa * F.dropout(self.attn_out(attn), p=self.dropout, training=self.training)

        h = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * F.dropout(self.mlp(h), p=self.dropout, training=self.training)
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask[..., None], 0)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


class APMDMQuadrupleHead(nn.Module):
    """Four output functions: unmask + binary remask / insert / delete."""

    def __init__(self, hidden_size: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)

        self.unmasking_head = nn.Linear(hidden_size, vocab_size)
        self.unmasking_head.weight.data.zero_()
        self.unmasking_head.bias.data.zero_()

        self.remasking_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.remasking_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.remasking_head.bias, -2.0)

        self.expansion_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.expansion_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.expansion_head.bias, -2.0)

        self.contraction_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.contraction_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.contraction_head.bias, -2.0)

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        h = self.norm_final(x) * (1 + scale) + shift
        return {
            "unmasking_logits": self.unmasking_head(h),
            "remasking_logits": self.remasking_head(h),
            "expansion_logits": self.expansion_head(h),
            "contraction_logits": self.contraction_head(h),
        }


@dataclass(frozen=True)
class ParameterCount:
    """Exact parameter accounting for an architecture."""

    total: int
    trainable: int
    by_group: Dict[str, int]

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "by_group": dict(sorted(self.by_group.items())),
        }


class APMDMEncoder(nn.Module):
    """The paper's AP-MDM encoder Transformer with four output functions."""

    def __init__(self, spec: ModelSpec, time_conditioning: bool = False):
        super().__init__()
        self.spec = spec
        self.time_conditioning = bool(time_conditioning)
        self.vocab_size = spec.vocab_size
        self.max_length = spec.length

        self.vocab_embed = EmbeddingLayer(spec.hidden_size, spec.vocab_size)
        self.sigma_map = TimestepEmbedder(spec.cond_dim)
        self.rotary_emb = Rotary(spec.hidden_size // spec.n_heads)
        self.blocks = nn.ModuleList(
            DDiTBlock(
                spec.hidden_size,
                spec.n_heads,
                spec.cond_dim,
                mlp_ratio=spec.mlp_ratio,
                dropout=spec.dropout,
            )
            for _ in range(spec.n_blocks)
        )
        self.output_layer = APMDMQuadrupleHead(spec.hidden_size, spec.vocab_size, spec.cond_dim)

    def architecture_signature(self) -> str:
        return self.spec.signature()

    def parameter_count(self) -> ParameterCount:
        groups: Dict[str, int] = {}
        for name, param in self.named_parameters():
            head = name.split(".")[0]
            if head == "blocks":
                head = "blocks"
            groups[head] = groups.get(head, 0) + param.numel()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return ParameterCount(total=total, trainable=trainable, by_group=groups)

    def forward(
        self,
        indices: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder.

        Args:
            indices: ``(batch, seq)`` token ids.
            sigma: ``(batch,)`` or ``(batch, 1)`` time conditioning.  With
                ``time_conditioning=False`` (the paper's supervised Sudoku
                setting) any value supplied is forced to zero.
        """
        if indices.dim() != 2:
            raise ValueError(f"expected (batch, seq) indices, got {tuple(indices.shape)}")
        batch, seq = indices.shape
        if seq > self.max_length:
            raise ValueError(f"sequence length {seq} exceeds model maximum {self.max_length}")
        if int(indices.max()) >= self.vocab_size or int(indices.min()) < 0:
            raise ValueError(
                f"token ids outside [0, {self.vocab_size}) present in input"
            )
        if attention_mask is not None:
            if attention_mask.shape != indices.shape:
                raise ValueError("attention_mask must match indices")
            attention_mask = attention_mask.bool()

        if sigma is None:
            sigma = torch.zeros(batch, device=indices.device)
        sigma = sigma.reshape(batch, -1)[:, 0] if sigma.dim() > 1 else sigma
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)

        x = self.vocab_embed(indices)
        if attention_mask is not None:
            x = x.masked_fill(~attention_mask[..., None], 0)
        c = F.silu(self.sigma_map(sigma)).to(x.dtype)
        rotary = self.rotary_emb(seq, indices.device, x.dtype)
        for block in self.blocks:
            x = block(x, rotary, c, attention_mask=attention_mask)
        return self.output_layer(x, c)


def build_model(spec: ModelSpec, time_conditioning: bool = False, seed: int | None = None) -> APMDMEncoder:
    """Construct an :class:`APMDMEncoder`, optionally with a fixed init seed."""
    if seed is not None:
        torch.manual_seed(seed)
    return APMDMEncoder(spec, time_conditioning=time_conditioning)


def count_parameters(spec: ModelSpec, time_conditioning: bool = False) -> ParameterCount:
    """Exact parameter count for an architecture spec (no CUDA required)."""
    with torch.device("meta"):
        model = APMDMEncoder(spec, time_conditioning=time_conditioning)
    return model.parameter_count()


def analytic_parameter_count(spec: ModelSpec) -> int:
    """Closed-form parameter count, used to cross-check :func:`count_parameters`."""
    d, c, v, ff = spec.hidden_size, spec.cond_dim, spec.vocab_size, spec.mlp_ratio * spec.hidden_size
    embed = v * d
    sigma_map = 256 * c + c + c * c + c
    per_block = (
        d  # norm1
        + 3 * d * d  # attn_qkv (no bias)
        + d * d  # attn_out  (no bias)
        + d  # norm2
        + (d * ff + ff) + (ff * d + d)  # mlp with biases
        + (c * 6 * d + 6 * d)  # adaLN
    )
    head = (
        d  # norm_final
        + (d * v + v)  # unmask
        + 3 * (d + 1)  # remask / insert / delete
        + (c * 2 * d + 2 * d)  # adaLN
    )
    return embed + sigma_map + spec.n_blocks * per_block + head
