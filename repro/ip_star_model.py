"""Paper-shaped Transformer models and objectives for star-graph insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from repro.ip_star_data import EOS_ID, NODE_VOCAB_SIZE, PAD_ID, StarBatch


@dataclass(frozen=True)
class TransformerSpec:
    width: int
    layers: int
    heads: int
    dropout: float = 0.1
    rope_base: float = 10_000.0
    mlp_ratio: int = 4


PAPER_DECODER_SPEC = TransformerSpec(width=128, layers=12, heads=8)
PAPER_POSTERIOR_SPEC = TransformerSpec(width=64, layers=6, heads=4)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., ::2]
    second = value[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_rope(
    query: torch.Tensor, key: torch.Tensor, *, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: (B, H, T, D)
    dimension = query.shape[-1]
    if dimension % 2:
        raise ValueError("RoPE head dimension must be even")
    position = torch.arange(query.shape[-2], device=query.device, dtype=torch.float32)
    frequency = 1.0 / (
        base
        ** (
            torch.arange(0, dimension, 2, device=query.device, dtype=torch.float32)
            / dimension
        )
    )
    angle = torch.outer(position, frequency).repeat_interleave(2, dim=-1)
    cosine = angle.cos().to(query.dtype)[None, None]
    sine = angle.sin().to(query.dtype)[None, None]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


class SelfAttention(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        if spec.width % spec.heads:
            raise ValueError("width must be divisible by head count")
        self.spec = spec
        self.qkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.output = nn.Linear(spec.width, spec.width, bias=False)
        self.dropout = spec.dropout

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = value.shape
        head_width = width // self.spec.heads
        qkv = self.qkv(value).view(batch, sequence, 3, self.spec.heads, head_width)
        query, key, content = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        content = content.transpose(1, 2)
        query, key = _apply_rope(query, key, base=self.spec.rope_base)
        additive_mask = torch.zeros(
            (batch, 1, 1, sequence), device=value.device, dtype=value.dtype
        ).masked_fill(~mask[:, None, None], -torch.inf)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            attn_mask=additive_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        hidden = spec.width * spec.mlp_ratio
        self.input = nn.Linear(spec.width, 2 * hidden, bias=False)
        self.output = nn.Linear(hidden, spec.width, bias=False)
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(value).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * content))


class TransformerBlock(nn.Module):
    def __init__(self, spec: TransformerSpec):
        super().__init__()
        self.attention_norm = nn.LayerNorm(spec.width)
        self.attention = SelfAttention(spec)
        self.mlp_norm = nn.LayerNorm(spec.width)
        self.mlp = SwiGLU(spec)
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value + self.dropout(self.attention(self.attention_norm(value), mask))
        value = value + self.dropout(self.mlp(self.mlp_norm(value)))
        return value.masked_fill(~mask[..., None], 0)


class SequenceTransformer(nn.Module):
    def __init__(self, spec: TransformerSpec, vocab_size: int = EOS_ID + 1):
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(vocab_size, spec.width, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList(TransformerBlock(spec) for _ in range(spec.layers))
        self.norm = nn.LayerNorm(spec.width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = self.token_embedding(tokens)
        value = value.masked_fill(~mask[..., None], 0)
        for block in self.blocks:
            value = block(value, mask)
        return self.norm(value).masked_fill(~mask[..., None], 0)


@dataclass(frozen=True)
class DecoderOutput:
    location_logits: torch.Tensor
    content_logits: torch.Tensor
    slot_mask: torch.Tensor


def _compress_targets(
    targets: torch.Tensor, target_mask: torch.Tensor, selected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = selected.sum(dim=-1)
    width = max(1, int(lengths.max().item()))
    partial = torch.full(
        (targets.shape[0], width), PAD_ID, dtype=torch.long, device=targets.device
    )
    partial_mask = torch.zeros_like(partial, dtype=torch.bool)
    chosen = target_mask & selected
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None].expand_as(targets)
    compressed_positions = chosen.long().cumsum(dim=-1) - 1
    partial[rows[chosen], compressed_positions[chosen]] = targets[chosen]
    positions = torch.arange(width, device=targets.device)[None]
    partial_mask = positions < lengths[:, None]
    return partial, partial_mask


def _pack_decoder_inputs(
    batch: StarBatch, partial: torch.Tensor, partial_mask: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    condition_lengths = batch.condition_mask.sum(dim=-1)
    partial_lengths = partial_mask.sum(dim=-1)
    total_length = int((condition_lengths + partial_lengths + 1).max().item())
    maximum_slots = int(partial_lengths.max().item()) + 1
    tokens = torch.full(
        (partial.shape[0], total_length), PAD_ID, dtype=torch.long, device=partial.device
    )
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    slot_positions = torch.zeros(
        (partial.shape[0], maximum_slots), dtype=torch.long, device=partial.device
    )
    slot_mask = torch.zeros_like(slot_positions, dtype=torch.bool)
    term_positions = torch.zeros(partial.shape[0], dtype=torch.long, device=partial.device)
    positions = torch.arange(total_length, device=partial.device)[None]
    condition_region = positions < condition_lengths[:, None]
    partial_positions = positions - condition_lengths[:, None]
    partial_region = (partial_positions >= 0) & (
        partial_positions < partial_lengths[:, None]
    )
    condition_indices = positions.clamp_max(batch.conditions.shape[1] - 1).expand(
        partial.shape[0], -1
    )
    partial_indices = partial_positions.clamp(0, partial.shape[1] - 1)
    tokens = torch.where(
        condition_region,
        batch.conditions.gather(1, condition_indices),
        tokens,
    )
    tokens = torch.where(partial_region, partial.gather(1, partial_indices), tokens)
    term_positions = condition_lengths + partial_lengths
    rows = torch.arange(partial.shape[0], device=partial.device)
    tokens[rows, term_positions] = EOS_ID
    mask = positions <= term_positions[:, None]
    slot_indices = torch.arange(maximum_slots, device=partial.device)[None]
    slot_positions = condition_lengths[:, None] - 1 + slot_indices
    slot_mask = slot_indices <= partial_lengths[:, None]
    slot_positions = slot_positions.clamp_max(total_length - 1)
    return tokens, mask, slot_positions, slot_mask, term_positions


class InsertionDecoder(nn.Module):
    def __init__(self, spec: TransformerSpec = PAPER_DECODER_SPEC):
        super().__init__()
        self.spec = spec
        self.transformer = SequenceTransformer(spec)
        self.location_head = nn.Linear(spec.width, 1, bias=False)
        self.content_head = nn.Linear(spec.width, EOS_ID + 1, bias=False)

    def forward(
        self, batch: StarBatch, partial: torch.Tensor, partial_mask: torch.Tensor
    ) -> DecoderOutput:
        packed = _pack_decoder_inputs(batch, partial, partial_mask)
        tokens, mask, slot_positions, slot_mask, term_positions = packed
        hidden = self.transformer(tokens, mask)
        gather_slot = slot_positions[..., None].expand(-1, -1, hidden.shape[-1])
        slot_hidden = hidden.gather(1, gather_slot)
        slot_logits = self.location_head(slot_hidden).squeeze(-1)
        slot_logits = slot_logits.masked_fill(~slot_mask, -torch.inf)
        term_hidden = hidden[torch.arange(hidden.shape[0], device=hidden.device), term_positions]
        term_logits = self.location_head(term_hidden).squeeze(-1)
        return DecoderOutput(
            location_logits=torch.cat((slot_logits, term_logits[:, None]), dim=-1),
            content_logits=self.content_head(slot_hidden),
            slot_mask=slot_mask,
        )


class InsertionPosterior(nn.Module):
    def __init__(self, spec: TransformerSpec = PAPER_POSTERIOR_SPEC):
        super().__init__()
        self.spec = spec
        self.transformer = SequenceTransformer(spec)
        self.order_head = nn.Linear(spec.width, 1, bias=False)

    def forward(self, batch: StarBatch) -> torch.Tensor:
        condition_lengths = batch.condition_mask.sum(dim=-1)
        total_length = int((condition_lengths + batch.lengths).max().item())
        tokens = torch.full(
            (batch.targets.shape[0], total_length),
            PAD_ID,
            dtype=torch.long,
            device=batch.targets.device,
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        target_positions = torch.zeros_like(batch.targets)
        positions = torch.arange(total_length, device=tokens.device)[None]
        condition_region = positions < condition_lengths[:, None]
        target_indices = positions - condition_lengths[:, None]
        target_region = (target_indices >= 0) & (target_indices < batch.lengths[:, None])
        condition_indices = positions.clamp_max(batch.conditions.shape[1] - 1).expand(
            batch.targets.shape[0], -1
        )
        target_gather = target_indices.clamp(0, batch.targets.shape[1] - 1)
        tokens = torch.where(
            condition_region,
            batch.conditions.gather(1, condition_indices),
            tokens,
        )
        tokens = torch.where(target_region, batch.targets.gather(1, target_gather), tokens)
        mask = condition_region | target_region
        target_positions = condition_lengths[:, None] + torch.arange(
            batch.targets.shape[1], device=tokens.device
        )[None]
        target_positions = target_positions.clamp_max(total_length - 1)
        hidden = self.transformer(tokens, mask)
        gathered = hidden.gather(
            1, target_positions[..., None].expand(-1, -1, hidden.shape[-1])
        )
        return self.order_head(gathered).squeeze(-1).masked_fill(
            ~batch.target_mask, -torch.inf
        )


@dataclass(frozen=True)
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _sample_prefix_lengths(
    lengths: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    uniform = torch.rand(lengths.shape, device=lengths.device, generator=generator)
    return torch.floor(uniform * (lengths + 1)).long()


def _sample_orders(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    uniform = torch.rand(
        (samples,) + logits.shape,
        device=logits.device,
        dtype=logits.dtype,
        generator=generator,
    ).clamp_(1e-8, 1 - 1e-8)
    gumbel = -torch.log(-torch.log(uniform))
    score = logits[None] + gumbel
    return score.masked_fill(~target_mask[None], -torch.inf).argsort(
        dim=-1, descending=True
    )


def _prefix_state(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    order: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = torch.arange(order.shape[-1], device=order.device)[None]
    take = rank < prefix_lengths[:, None]
    selected = torch.zeros_like(target_mask)
    selected.scatter_(1, order, take)

    ordered_logits = logits.gather(1, order)
    ordered_valid = target_mask.gather(1, order)
    ordered_logits = ordered_logits.masked_fill(~ordered_valid, -torch.inf)
    denominator = torch.logcumsumexp(ordered_logits.flip(-1), dim=-1).flip(-1)
    contribution = torch.where(
        ordered_valid, ordered_logits - denominator, torch.zeros_like(ordered_logits)
    )
    log_probability = torch.where(take, contribution, torch.zeros_like(contribution)).sum(
        dim=-1
    )
    return log_probability, selected


def _exact_next_value(
    decoder: InsertionDecoder,
    batch: StarBatch,
    q_logits: torch.Tensor,
    selected: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    partial, partial_mask = _compress_targets(batch.targets, batch.target_mask, selected)
    output = decoder(batch, partial, partial_mask)
    # Keep Transformer matmuls in BF16, but perform probability arithmetic in
    # FP32. Besides being numerically safer, this keeps diagnostics and indexed
    # assignments dtype-consistent under autocast.
    location_log_probability = F.log_softmax(output.location_logits.float(), dim=-1)
    term = prefix_lengths == batch.lengths
    value = location_log_probability[:, -1].clone()
    q_entropy = q_logits.new_zeros(q_logits.shape[0])
    location_probability = location_log_probability.exp()
    location_entropy = -torch.where(
        torch.isfinite(location_log_probability),
        location_probability * location_log_probability,
        torch.zeros_like(location_log_probability),
    ).sum(dim=-1)
    expected_content_nll = q_logits.new_zeros(q_logits.shape[0])

    active = ~term
    if bool(active.any()):
        remaining = batch.target_mask[active] & ~selected[active]
        active_q_logits = q_logits[active].masked_fill(~remaining, -torch.inf)
        q_log_probability = F.log_softmax(active_q_logits, dim=-1)
        q_probability = q_log_probability.exp()
        slot_index = selected[active].long().cumsum(dim=-1) - selected[active].long()
        location = location_log_probability[active, :-1].gather(1, slot_index)
        content_log_probability = F.log_softmax(
            output.content_logits[active, :, :NODE_VOCAB_SIZE].float(), dim=-1
        )
        row_index = torch.arange(slot_index.shape[0], device=slot_index.device)[:, None]
        content = content_log_probability[
            row_index, slot_index, batch.targets[active].clamp_max(NODE_VOCAB_SIZE - 1)
        ]
        bracket = location + content - q_log_probability
        value[active] = (
            q_probability
            * torch.where(remaining, bracket, torch.zeros_like(bracket))
        ).sum(dim=-1)
        q_entropy[active] = -(
            q_probability
            * torch.where(
                remaining, q_log_probability, torch.zeros_like(q_log_probability)
            )
        ).sum(dim=-1)
        expected_content_nll[active] = -(
            q_probability
            * torch.where(remaining, content, torch.zeros_like(content))
        ).sum(dim=-1)
    return value, {
        "q_next_entropy": q_entropy,
        "location_entropy": location_entropy,
        "expected_content_nll": expected_content_nll,
        "termination_fraction": term.float(),
    }


def learned_ip_objective(
    decoder: InsertionDecoder,
    posterior: InsertionPosterior,
    batch: StarBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    q_logits = posterior(batch).float()
    orders = _sample_orders(q_logits, batch.target_mask, 2, generator)
    prefix_lengths = _sample_prefix_lengths(batch.lengths, generator)
    prefix_logs = []
    values = []
    diagnostics = []
    for sample in range(2):
        prefix_log, selected = _prefix_state(
            q_logits, batch.target_mask, orders[sample], prefix_lengths
        )
        value, diagnostic = _exact_next_value(
            decoder, batch, q_logits, selected, prefix_lengths
        )
        prefix_logs.append(prefix_log)
        values.append(value)
        diagnostics.append(diagnostic)
    f_1, f_2 = values
    score = (prefix_logs[0] - prefix_logs[1]) * (f_1 - f_2).detach()
    scale = (batch.lengths + 1).float()
    surrogate = 0.5 * scale * (score + f_1 + f_2)
    monitor_elbo = 0.5 * scale * (f_1 + f_2)
    metrics = {
        "elbo": monitor_elbo.mean().detach(),
        "surrogate": surrogate.mean().detach(),
        "rloo_value_variance": (0.5 * (f_1 - f_2).square()).mean().detach(),
        "rloo_absolute_advantage": (f_1 - f_2).abs().mean().detach(),
        "q_prefix_log_probability": torch.stack(prefix_logs).mean().detach(),
        "mean_prefix_length": prefix_lengths.float().mean().detach(),
    }
    for name in diagnostics[0]:
        metrics[name] = torch.stack([row[name] for row in diagnostics]).mean().detach()
    return ObjectiveOutput(loss=-surrogate.mean(), metrics=metrics)


def random_ip_objective(
    decoder: InsertionDecoder,
    batch: StarBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    q_logits = torch.zeros_like(batch.targets, dtype=torch.float32)
    q_logits = q_logits.masked_fill(~batch.target_mask, -torch.inf)
    orders = _sample_orders(q_logits, batch.target_mask, 2, generator)
    prefix_lengths = _sample_prefix_lengths(batch.lengths, generator)
    values = []
    diagnostics = []
    for sample in range(2):
        _, selected = _prefix_state(
            q_logits, batch.target_mask, orders[sample], prefix_lengths
        )
        value, diagnostic = _exact_next_value(
            decoder, batch, q_logits, selected, prefix_lengths
        )
        values.append(value)
        diagnostics.append(diagnostic)
    scale = (batch.lengths + 1).float()
    objective = 0.5 * scale * (values[0] + values[1])
    metrics = {
        "elbo": objective.mean().detach(),
        "mean_prefix_length": prefix_lengths.float().mean().detach(),
    }
    for name in diagnostics[0]:
        metrics[name] = torch.stack([row[name] for row in diagnostics]).mean().detach()
    return ObjectiveOutput(loss=-objective.mean(), metrics=metrics)


def fixed_order_objective(
    decoder: InsertionDecoder,
    batch: StarBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    prefix_lengths = _sample_prefix_lengths(batch.lengths, generator)
    position = torch.arange(batch.targets.shape[-1], device=batch.targets.device)[None]
    selected = batch.target_mask & (position < prefix_lengths[:, None])
    partial, partial_mask = _compress_targets(batch.targets, batch.target_mask, selected)
    output = decoder(batch, partial, partial_mask)
    row = torch.arange(batch.targets.shape[0], device=batch.targets.device)
    next_slot = prefix_lengths
    content = output.content_logits[row, next_slot].float()
    allowed = torch.cat((content[:, :NODE_VOCAB_SIZE], content[:, EOS_ID : EOS_ID + 1]), dim=-1)
    target = torch.where(
        prefix_lengths == batch.lengths,
        torch.full_like(prefix_lengths, NODE_VOCAB_SIZE),
        batch.targets.gather(
            1, prefix_lengths.clamp_max(batch.targets.shape[-1] - 1)[:, None]
        ).squeeze(1),
    )
    log_probability = F.log_softmax(allowed, dim=-1).gather(1, target[:, None]).squeeze(1)
    objective = (batch.lengths + 1).float() * log_probability
    return ObjectiveOutput(
        loss=-objective.mean(),
        metrics={
            "elbo": objective.mean().detach(),
            "mean_prefix_length": prefix_lengths.float().mean().detach(),
            "next_token_accuracy": (allowed.argmax(dim=-1) == target).float().mean().detach(),
            "termination_fraction": (prefix_lengths == batch.lengths).float().mean().detach(),
        },
    )
