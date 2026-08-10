"""Classifier-terminated permutation-variational insertion model for SMILES."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from repro.ip_chem_data import ChemBatch, SmilesTokenizer
from repro.ip_star_model import ObjectiveOutput, SequenceTransformer, TransformerSpec


# The paper states 18 and 3 layers but omits every remaining dimension. These
# values are a declared capacity choice shared by all reconstruction arms.
CHEM_DECODER_SPEC = TransformerSpec(width=256, layers=18, heads=8, dropout=0.1)
CHEM_POSTERIOR_SPEC = TransformerSpec(width=128, layers=3, heads=4, dropout=0.1)


@dataclass(frozen=True)
class ChemDecoderOutput:
    location_logits: torch.Tensor
    content_logits: torch.Tensor
    slot_mask: torch.Tensor


class ChemInsertionDecoder(nn.Module):
    """Scores insertion slots and content; EOS is content at the final slot."""

    def __init__(self, tokenizer: SmilesTokenizer, spec: TransformerSpec = CHEM_DECODER_SPEC):
        super().__init__()
        self.spec = spec
        self.tokenizer = tokenizer
        self.transformer = SequenceTransformer(
            spec, vocab_size=len(tokenizer.tokens), padding_idx=tokenizer.pad_id
        )
        self.location_head = nn.Linear(spec.width, 1, bias=False)
        self.content_head = nn.Linear(spec.width, len(tokenizer.tokens), bias=False)

    def forward(self, partial: torch.Tensor, partial_mask: torch.Tensor) -> ChemDecoderOutput:
        lengths = partial_mask.sum(dim=-1)
        width = int(lengths.max().item()) + 1
        tokens = torch.full(
            (partial.shape[0], width),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=partial.device,
        )
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, 0] = self.tokenizer.bos_id
        mask[:, 0] = True
        if partial.shape[1]:
            positions = torch.arange(width - 1, device=partial.device)[None]
            active = positions < lengths[:, None]
            gather = positions.clamp_max(partial.shape[1] - 1).expand(partial.shape[0], -1)
            tokens[:, 1:] = torch.where(
                active, partial.gather(1, gather), tokens[:, 1:]
            )
            mask[:, 1:] = active
        hidden = self.transformer(tokens, mask)
        slot_positions = torch.arange(width, device=partial.device)[None]
        slot_mask = slot_positions <= lengths[:, None]
        return ChemDecoderOutput(
            location_logits=self.location_head(hidden).squeeze(-1).masked_fill(
                ~slot_mask, -torch.inf
            ),
            content_logits=self.content_head(hidden),
            slot_mask=slot_mask,
        )


class ChemInsertionPosterior(nn.Module):
    def __init__(self, tokenizer: SmilesTokenizer, spec: TransformerSpec = CHEM_POSTERIOR_SPEC):
        super().__init__()
        self.spec = spec
        self.tokenizer = tokenizer
        self.transformer = SequenceTransformer(
            spec, vocab_size=len(tokenizer.tokens), padding_idx=tokenizer.pad_id
        )
        self.order_head = nn.Linear(spec.width, 1, bias=False)

    def forward(self, targets: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(targets, target_mask)
        logits = self.order_head(hidden).squeeze(-1).masked_fill(~target_mask, -torch.inf)
        # Appendix: classifier termination augments the observation with EOS and
        # fixes its posterior logit to -1e7 so it is always inserted last.
        eos = targets == self.tokenizer.eos_id
        return logits.masked_fill(eos & target_mask, -1e7)


def _augment_with_eos(batch: ChemBatch, tokenizer: SmilesTokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = batch.targets.shape[1] + 1
    targets = torch.full(
        (batch.targets.shape[0], width),
        tokenizer.pad_id,
        dtype=torch.long,
        device=batch.targets.device,
    )
    mask = torch.zeros_like(targets, dtype=torch.bool)
    targets[:, : batch.targets.shape[1]] = batch.targets
    mask[:, : batch.target_mask.shape[1]] = batch.target_mask
    rows = torch.arange(targets.shape[0], device=targets.device)
    targets[rows, batch.lengths] = tokenizer.eos_id
    mask[rows, batch.lengths] = True
    return targets, mask, batch.lengths + 1


def _sample_prefix_lengths(total_lengths: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # One ELBO term among i=1,...,L+1; prefix size is therefore 0,...,L.
    uniform = torch.rand(total_lengths.shape, device=total_lengths.device, generator=generator)
    return torch.floor(uniform * total_lengths).long()


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
    return (logits[None] + gumbel).masked_fill(~target_mask[None], -torch.inf).argsort(
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
    log_probability = torch.where(take, contribution, torch.zeros_like(contribution)).sum(-1)
    return log_probability, selected


def _compress_targets(
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    selected: torch.Tensor,
    tokenizer: SmilesTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = target_mask & selected & (targets != tokenizer.eos_id)
    lengths = chosen.sum(dim=-1)
    width = max(1, int(lengths.max().item()))
    partial = torch.full(
        (targets.shape[0], width), tokenizer.pad_id, dtype=torch.long, device=targets.device
    )
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None].expand_as(targets)
    positions = chosen.long().cumsum(dim=-1) - 1
    partial[rows[chosen], positions[chosen]] = targets[chosen]
    partial_mask = torch.arange(width, device=targets.device)[None] < lengths[:, None]
    return partial, partial_mask


def _exact_next_value(
    decoder: ChemInsertionDecoder,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    q_logits: torch.Tensor,
    selected: torch.Tensor,
    tokenizer: SmilesTokenizer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    partial, partial_mask = _compress_targets(targets, target_mask, selected, tokenizer)
    output = decoder(partial, partial_mask)
    location_logp = F.log_softmax(output.location_logits.float(), dim=-1)
    remaining = target_mask & ~selected
    remaining_q = q_logits.masked_fill(~remaining, -torch.inf)
    q_logp = F.log_softmax(remaining_q, dim=-1)
    q_probability = q_logp.exp()
    slot_index = selected.long().cumsum(dim=-1) - selected.long()
    location = location_logp.gather(1, slot_index.clamp_max(location_logp.shape[1] - 1))
    content_logp = F.log_softmax(output.content_logits.float(), dim=-1)
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None]
    content = content_logp[rows, slot_index, targets]
    bracket = location + content - q_logp
    value = (
        q_probability * torch.where(remaining, bracket, torch.zeros_like(bracket))
    ).sum(dim=-1)
    loc_probability = location_logp.exp()
    return value, {
        "q_next_entropy": -(
            q_probability
            * torch.where(remaining, q_logp, torch.zeros_like(q_logp))
        ).sum(dim=-1),
        "location_entropy": -torch.where(
            torch.isfinite(location_logp),
            loc_probability * location_logp,
            torch.zeros_like(location_logp),
        ).sum(dim=-1),
        "expected_content_nll": -(
            q_probability
            * torch.where(remaining, content, torch.zeros_like(content))
        ).sum(dim=-1),
        "eos_next_probability": (
            q_probability * (targets == tokenizer.eos_id).float()
        ).sum(dim=-1),
    }


def learned_ip_objective(
    decoder: ChemInsertionDecoder,
    posterior: ChemInsertionPosterior,
    batch: ChemBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    targets, mask, total_lengths = _augment_with_eos(batch, decoder.tokenizer)
    q_logits = posterior(targets, mask).float()
    orders = _sample_orders(q_logits, mask, 2, generator)
    prefixes = _sample_prefix_lengths(total_lengths, generator)
    prefix_logs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    diagnostics: list[dict[str, torch.Tensor]] = []
    for sample in range(2):
        prefix_log, selected = _prefix_state(q_logits, mask, orders[sample], prefixes)
        value, diagnostic = _exact_next_value(
            decoder, targets, mask, q_logits, selected, decoder.tokenizer
        )
        prefix_logs.append(prefix_log)
        values.append(value)
        diagnostics.append(diagnostic)
    f_1, f_2 = values
    score = (prefix_logs[0] - prefix_logs[1]) * (f_1 - f_2).detach()
    scale = total_lengths.float()
    surrogate = 0.5 * scale * (score + f_1 + f_2)
    monitor = 0.5 * scale * (f_1 + f_2)
    metrics = {
        "elbo": monitor.mean().detach(),
        "surrogate": surrogate.mean().detach(),
        "rloo_value_variance": (0.5 * (f_1 - f_2).square()).mean().detach(),
        "q_prefix_log_probability": torch.stack(prefix_logs).mean().detach(),
        "mean_prefix_length": prefixes.float().mean().detach(),
    }
    for name in diagnostics[0]:
        metrics[name] = torch.stack([row[name] for row in diagnostics]).mean().detach()
    return ObjectiveOutput(loss=-surrogate.mean(), metrics=metrics)


def random_ip_objective(
    decoder: ChemInsertionDecoder,
    batch: ChemBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    targets, mask, total_lengths = _augment_with_eos(batch, decoder.tokenizer)
    q_logits = torch.zeros_like(targets, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    q_logits = q_logits.masked_fill((targets == decoder.tokenizer.eos_id) & mask, -1e7)
    orders = _sample_orders(q_logits, mask, 2, generator)
    prefixes = _sample_prefix_lengths(total_lengths, generator)
    values: list[torch.Tensor] = []
    diagnostics: list[dict[str, torch.Tensor]] = []
    for sample in range(2):
        _, selected = _prefix_state(q_logits, mask, orders[sample], prefixes)
        value, diagnostic = _exact_next_value(
            decoder, targets, mask, q_logits, selected, decoder.tokenizer
        )
        values.append(value)
        diagnostics.append(diagnostic)
    objective = 0.5 * total_lengths.float() * (values[0] + values[1])
    metrics = {
        "elbo": objective.mean().detach(),
        "mean_prefix_length": prefixes.float().mean().detach(),
    }
    for name in diagnostics[0]:
        metrics[name] = torch.stack([row[name] for row in diagnostics]).mean().detach()
    return ObjectiveOutput(loss=-objective.mean(), metrics=metrics)


def fixed_order_objective(
    decoder: ChemInsertionDecoder,
    batch: ChemBatch,
    generator: torch.Generator,
) -> ObjectiveOutput:
    prefixes = _sample_prefix_lengths(batch.lengths + 1, generator)
    positions = torch.arange(batch.targets.shape[1], device=batch.targets.device)[None]
    selected = batch.target_mask & (positions < prefixes[:, None])
    partial, partial_mask = _compress_targets(
        batch.targets, batch.target_mask, selected, decoder.tokenizer
    )
    output = decoder(partial, partial_mask)
    rows = torch.arange(batch.targets.shape[0], device=batch.targets.device)
    target = torch.where(
        prefixes == batch.lengths,
        torch.full_like(prefixes, decoder.tokenizer.eos_id),
        batch.targets.gather(
            1, prefixes.clamp_max(batch.targets.shape[1] - 1)[:, None]
        ).squeeze(1),
    )
    content = output.content_logits[rows, prefixes]
    logp = F.log_softmax(content.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
    objective = (batch.lengths + 1).float() * logp
    return ObjectiveOutput(
        loss=-objective.mean(),
        metrics={
            "elbo": objective.mean().detach(),
            "mean_prefix_length": prefixes.float().mean().detach(),
            "next_token_accuracy": (content.argmax(-1) == target).float().mean().detach(),
            "eos_next_probability": (prefixes == batch.lengths).float().mean().detach(),
        },
    )
