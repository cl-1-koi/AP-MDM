"""Architecture, parameter-count, head-shape and attention-backend tests."""

from __future__ import annotations

import math

import pytest
import torch

from repro import vocab
from repro.config import ModelSpec
from repro.model import (
    APMDMEncoder,
    analytic_parameter_count,
    apply_rotary,
    build_model,
    count_parameters,
)

#: Exact parameter counts measured for the two checked-in Sudoku architectures.
PAPER_FAITHFUL_PARAMS = 6_050_594
HISTORICAL_PARAMS = 1_035_426

TINY = ModelSpec(
    vocab_size=vocab.VOCAB_SIZE, length=64, hidden_size=32, n_heads=2, n_blocks=2,
    cond_dim=16, dropout=0.0,
)


def test_paper_faithful_architecture_fields(paper_config):
    spec = paper_config.model
    assert spec.n_blocks == 6
    assert spec.n_heads == 4
    assert spec.hidden_size == 256
    assert spec.mlp_ratio * spec.hidden_size == 1024
    assert spec.length == 400
    assert spec.vocab_size == vocab.VOCAB_SIZE == 31
    assert paper_config.time_conditioning is False
    assert paper_config.seed == 42


def test_exact_parameter_counts(paper_config, historical_config):
    paper = count_parameters(paper_config.model)
    historical = count_parameters(historical_config.model)
    assert paper.total == PAPER_FAITHFUL_PARAMS
    assert historical.total == HISTORICAL_PARAMS
    assert paper.total == analytic_parameter_count(paper_config.model)
    assert historical.total == analytic_parameter_count(historical_config.model)
    assert paper.total == paper.trainable


def test_paper_reported_parameter_count_does_not_match_its_own_architecture(paper_config):
    """The paper reports ~1.2M parameters for a 6x4x256 encoder; it is ~6.05M.

    This test pins the discrepancy so it cannot be silently "fixed" later.
    """
    reported = 1_200_000
    actual = count_parameters(paper_config.model).total
    assert actual > 4 * reported


def test_four_output_heads_have_the_declared_shapes(paper_config):
    model = build_model(paper_config.model, time_conditioning=False, seed=0)
    x = torch.randint(0, paper_config.model.vocab_size, (2, vocab.SEQUENCE_LENGTH))
    out = model(x)
    assert set(out) == {
        "unmasking_logits",
        "remasking_logits",
        "expansion_logits",
        "contraction_logits",
    }
    assert out["unmasking_logits"].shape == (2, vocab.SEQUENCE_LENGTH, paper_config.model.vocab_size)
    for key in ("remasking_logits", "expansion_logits", "contraction_logits"):
        assert out[key].shape == (2, vocab.SEQUENCE_LENGTH, 1)


def test_time_conditioning_disabled_makes_sigma_irrelevant(paper_config):
    model = build_model(paper_config.model, time_conditioning=False, seed=0).eval()
    x = torch.randint(0, paper_config.model.vocab_size, (2, 64))
    with torch.no_grad():
        zero = model(x, torch.zeros(2))
        large = model(x, torch.full((2,), 7.5))
    for key in zero:
        assert torch.equal(zero[key], large[key])


def test_time_conditioning_enabled_does_change_the_output():
    spec = ModelSpec(
        vocab_size=vocab.VOCAB_SIZE, length=64, hidden_size=32, n_heads=2, n_blocks=1,
        cond_dim=16, dropout=0.0,
    )
    model = build_model(spec, time_conditioning=True, seed=0).eval()
    # adaLN modulation is zero-initialised, so give it non-trivial weights.
    for block in model.blocks:
        torch.nn.init.normal_(block.adaLN_modulation.weight, std=0.05)
    torch.nn.init.normal_(model.output_layer.adaLN_modulation.weight, std=0.05)
    # The unmasking head is zero-initialised upstream, so read a head that is not.
    x = torch.randint(0, spec.vocab_size, (2, 32))
    with torch.no_grad():
        a = model(x, torch.zeros(2))["remasking_logits"]
        b = model(x, torch.full((2,), 3.0))["remasking_logits"]
    assert not torch.allclose(a, b)


def test_model_rejects_out_of_range_tokens_and_overlong_sequences(paper_config):
    model = build_model(paper_config.model, seed=0)
    with pytest.raises(ValueError):
        model(torch.tensor([[paper_config.model.vocab_size]]))
    with pytest.raises(ValueError):
        model(torch.zeros((1, paper_config.model.length + 1), dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 1, 4), dtype=torch.long))


def test_variable_sequence_length_is_supported(paper_config):
    model = build_model(paper_config.model, seed=0).eval()
    for length in (8, 324, 400):
        out = model(torch.zeros((1, length), dtype=torch.long))
        assert out["unmasking_logits"].shape[1] == length


def test_architecture_signature_is_stable_and_distinguishing(paper_config, historical_config):
    assert paper_config.model.signature() != historical_config.model.signature()
    assert "L=6" in paper_config.model.signature()
    assert "d=256" in paper_config.model.signature()
    assert "V=31" in paper_config.model.signature()


def test_rotary_convention_is_rotate_half_over_the_full_head_dim():
    head_dim = 8
    x = torch.zeros(1, 2, 1, head_dim)
    x[0, 1, 0, 0] = 1.0
    freqs = torch.arange(2, dtype=torch.float32)[:, None] * torch.ones(1, head_dim // 2)
    cos, sin = freqs.cos(), freqs.sin()
    out = apply_rotary(x, cos, sin)
    # A unit vector at position 1 dimension 0 rotates into dims (0, head_dim/2).
    assert math.isclose(float(out[0, 1, 0, 0]), float(cos[1, 0]), rel_tol=1e-6)
    assert math.isclose(float(out[0, 1, 0, head_dim // 2]), float(sin[1, 0]), rel_tol=1e-6)
    # Position 0 has zero frequency, so it is untouched.
    assert torch.allclose(out[0, 0], x[0, 0])


def test_dropout_is_inactive_in_eval_mode():
    spec = ModelSpec(
        vocab_size=vocab.VOCAB_SIZE, length=64, hidden_size=32, n_heads=2, n_blocks=2,
        cond_dim=16, dropout=0.5,
    )
    model = build_model(spec, seed=0).eval()
    x = torch.randint(0, spec.vocab_size, (2, 16))
    with torch.no_grad():
        a = model(x)["unmasking_logits"]
        b = model(x)["unmasking_logits"]
    assert torch.equal(a, b)


def test_parameter_groups_cover_every_parameter(paper_config):
    model = build_model(paper_config.model, seed=0)
    counted = model.parameter_count()
    assert sum(counted.by_group.values()) == counted.total
    assert set(counted.by_group) == {"vocab_embed", "sigma_map", "blocks", "output_layer"}


@pytest.mark.skipif(
    torch.__version__ < "2.0", reason="scaled_dot_product_attention requires torch>=2.0"
)
def test_sdpa_matches_an_explicit_softmax_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, dtype=torch.float64) for _ in range(3))
    reference = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1) @ v
    fused = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert torch.allclose(reference, fused, atol=1e-10)


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        import flash_attn.layers.rotary  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _flash_attn_available(),
    reason="flash_attn is not installed on this CUDA/Torch stack; SDPA path is used",
)
def test_flash_attention_and_sdpa_agree_within_declared_tolerance():
    """Declared tolerance: max |delta| <= 2e-2 in bf16, <= 2e-4 in fp32."""
    import flash_attn
    import flash_attn.layers.rotary

    torch.manual_seed(0)
    batch, seq, heads, head_dim = 2, 32, 4, 64
    qkv = torch.randn(batch, seq, 3, heads, head_dim, device="cuda", dtype=torch.float16)
    inv_freq = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim)).cuda()
    t = torch.arange(seq, device="cuda").float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos, sin = freqs.cos().half(), freqs.sin().half()

    flash_qkv = qkv.clone()
    flash_qkv = flash_attn.layers.rotary.apply_rotary_emb_qkv_(flash_qkv, cos, sin)
    cu = torch.arange(0, (batch + 1) * seq, step=seq, dtype=torch.int32, device="cuda")
    flash_out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        flash_qkv.reshape(batch * seq, 3, heads, head_dim), cu, seq, 0.0, causal=False
    ).reshape(batch, seq, heads, head_dim)

    q = apply_rotary(qkv[:, :, 0], cos, sin)
    k = apply_rotary(qkv[:, :, 1], cos, sin)
    v = qkv[:, :, 2]
    sdpa_out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
    ).transpose(1, 2)

    assert float((flash_out.float() - sdpa_out.float()).abs().max()) <= 2e-2
