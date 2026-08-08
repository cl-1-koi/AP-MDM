"""The paper's supervised AP-MDM objective (``method.tex`` Appendix "Supervised Training").

.. math::

    \\mathcal{L}^{sup}_{unmask} &= -\\frac{1}{|M|}\\sum_{i \\in M} \\log p_\\theta(y^*_i \\mid x_k) \\\\
    \\mathcal{L}^{sup}_{op}     &= \\frac{1}{\\sum_i m_i} \\sum_i m_i \\, \\mathrm{BCE}(\\mathrm{logit}_{op,i}, c^*_i)

with :math:`M = \\{i : x_{k,i} = \\mathrm{MASK},\\, y^*_i \\neq \\mathrm{MASK}\\}` and
:math:`m` the validity mask.  The combined objective is
:math:`\\mathcal{L}_{unmask} + \\lambda_r \\mathcal{L}_{remask} + \\lambda_i \\mathcal{L}_{insert} + \\lambda_d \\mathcal{L}_{delete}`
with all :math:`\\lambda = 1`.

This mirrors ``train/diffusion.py::_forward_pass_apmdm_precomputed`` including
its SUBS parameterisation of the unmasking logits, but computes the loss in
float32 and without mutating the model output in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from repro.vocab import MASK

NEG_INFINITY = -1_000_000.0


def subs_parameterization(
    logits: torch.Tensor, xt: torch.Tensor, mask_index: int = MASK
) -> torch.Tensor:
    """SUBS log-probabilities: never predict ``[MASK]``; pin unmasked positions.

    Returns a new tensor; the input is not modified.
    """
    out = logits.float().clone()
    out[:, :, mask_index] += NEG_INFINITY
    out = out - torch.logsumexp(out, dim=-1, keepdim=True)
    unmasked = xt != mask_index
    if unmasked.any():
        out[unmasked] = NEG_INFINITY
        out[unmasked, xt[unmasked]] = 0.0
    return out


def subs_argmax(
    logits: torch.Tensor, xt: torch.Tensor, mask_index: int = MASK
) -> torch.Tensor:
    """``argmax`` of the SUBS-parameterised distribution, without materialising it.

    Equivalent to ``subs_parameterization(logits, xt).argmax(-1)``: at masked
    positions the arg-max is taken with ``[MASK]`` excluded, and at unmasked
    positions SUBS pins all probability on the current token.  Used on the
    inference hot path, where building the full ``(B, L, V)`` log-probability
    tensor is the dominant cost.  A test pins the two against each other.
    """
    masked = logits.float().clone()
    masked[:, :, mask_index] = float("-inf")
    predicted = masked.argmax(dim=-1)
    return torch.where(xt == mask_index, predicted, xt)


@dataclass
class LossBreakdown:
    """Scalar loss components plus the counts they were normalised by."""

    total: torch.Tensor
    unmask: torch.Tensor
    remask: torch.Tensor
    insert: torch.Tensor
    delete: torch.Tensor
    n_unmask_positions: int
    n_valid_positions: int

    def scalars(self) -> Dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "loss_unmask": float(self.unmask.detach()),
            "loss_remask": float(self.remask.detach()),
            "loss_insert": float(self.insert.detach()),
            "loss_delete": float(self.delete.detach()),
            "n_unmask_positions": self.n_unmask_positions,
            "n_valid_positions": self.n_valid_positions,
        }


def _binary_loss(
    logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    valid: torch.Tensor,
    valid_count: torch.Tensor,
) -> torch.Tensor:
    if logits is None:
        raise ValueError("model did not emit the requested control-head logits")
    pointwise = F.binary_cross_entropy_with_logits(
        logits.squeeze(-1).float(), labels.float(), reduction="none"
    )
    return (pointwise * valid).sum() / valid_count


def supervised_loss(
    outputs: Dict[str, torch.Tensor],
    x_k: torch.Tensor,
    y_star: torch.Tensor,
    r_star: torch.Tensor,
    e_star: torch.Tensor,
    c_star: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    lambda_remask: float = 1.0,
    lambda_insert: float = 1.0,
    lambda_delete: float = 1.0,
    mask_index: int = MASK,
) -> LossBreakdown:
    """Compute the four supervised loss components and their weighted sum."""
    if attention_mask is None:
        attention_mask = torch.ones_like(x_k, dtype=torch.float32)
    valid = attention_mask.float()
    valid_count = valid.sum().clamp(min=1.0)

    log_probs = subs_parameterization(outputs["unmasking_logits"], x_k, mask_index)
    gathered = torch.gather(log_probs, dim=-1, index=y_star[:, :, None]).squeeze(-1)
    unmask_positions = (x_k == mask_index) & (y_star != mask_index) & valid.bool()
    n_unmask = int(unmask_positions.sum())
    if n_unmask > 0:
        unmask_loss = -(torch.where(unmask_positions, gathered, torch.zeros_like(gathered)).sum()) / n_unmask
    else:
        # Keep the graph connected without contributing signal, exactly as the
        # upstream implementation does for pure-remask transitions.
        unmask_loss = outputs["unmasking_logits"].float().mean() * 0.0

    remask_loss = _binary_loss(outputs.get("remasking_logits"), r_star, valid, valid_count)
    insert_loss = _binary_loss(outputs.get("expansion_logits"), e_star, valid, valid_count)
    delete_loss = _binary_loss(outputs.get("contraction_logits"), c_star, valid, valid_count)

    total = (
        unmask_loss
        + lambda_remask * remask_loss
        + lambda_insert * insert_loss
        + lambda_delete * delete_loss
    )
    return LossBreakdown(
        total=total,
        unmask=unmask_loss,
        remask=remask_loss,
        insert=insert_loss,
        delete=delete_loss,
        n_unmask_positions=n_unmask,
        n_valid_positions=int(valid.sum()),
    )


@torch.no_grad()
def head_diagnostics(
    outputs: Dict[str, torch.Tensor],
    x_k: torch.Tensor,
    y_star: torch.Tensor,
    r_star: torch.Tensor,
    e_star: torch.Tensor,
    c_star: torch.Tensor,
    thresholds: Dict[str, float] | None = None,
    mask_index: int = MASK,
) -> Dict[str, float]:
    """Per-head statistics for diagnosing collapse, non-emission and looping."""
    thresholds = thresholds or {"remask": 0.5, "insert": 0.5, "delete": 0.5}
    stats: Dict[str, float] = {}

    log_probs = subs_parameterization(outputs["unmasking_logits"], x_k, mask_index)
    pred = log_probs.argmax(dim=-1)
    positions = (x_k == mask_index) & (y_star != mask_index)
    n = int(positions.sum())
    stats["unmask_positions"] = float(n)
    stats["unmask_token_accuracy"] = (
        float((pred[positions] == y_star[positions]).float().mean()) if n else float("nan")
    )

    for key, logits_key, labels in (
        ("remask", "remasking_logits", r_star),
        ("insert", "expansion_logits", e_star),
        ("delete", "contraction_logits", c_star),
    ):
        logits = outputs.get(logits_key)
        if logits is None:
            continue
        probs = torch.sigmoid(logits.squeeze(-1).float())
        fired = probs > thresholds[key]
        target = labels.bool()
        stats[f"{key}_prob_mean"] = float(probs.mean())
        stats[f"{key}_prob_max"] = float(probs.max())
        stats[f"{key}_pred_positive_rate"] = float(fired.float().mean())
        stats[f"{key}_target_positive_rate"] = float(target.float().mean())
        tp = float((fired & target).sum())
        fp = float((fired & ~target).sum())
        fn = float((~fired & target).sum())
        stats[f"{key}_true_positives"] = tp
        stats[f"{key}_false_positives"] = fp
        stats[f"{key}_false_negatives"] = fn
        stats[f"{key}_recall"] = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        stats[f"{key}_precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return stats
