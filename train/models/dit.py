import typing
import math

try:
    import flash_attn
    import flash_attn.layers.rotary
    FLASH_ATTN_AVAILABLE = True
except (ImportError, OSError):
    # The released environment assumes a FlashAttention wheel.  Torch 2.7 /
    # CUDA 12.8 on the target A10 has no compatible wheel, so retain the
    # released tensor/module structure and use PyTorch SDPA as a kernel-only
    # compatibility fallback.
    flash_attn = None
    FLASH_ATTN_AVAILABLE = False
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:

    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    if FLASH_ATTN_AVAILABLE:
        return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

    # flash_attn's packed-QKV helper rotates Q and K, but leaves V unchanged.
    # Its rotate-half convention consumes half-width cos/sin tables and repeats
    # them over the full head dimension.
    cos = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(qkv.dtype)
    sin = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(qkv.dtype)
    q, k, v = qkv.unbind(dim=2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return torch.stack((q, k, v), dim=2)


# function overload
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat(
                [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference


    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                                        'b s (three h d) -> b s three h d',
                                        three=3,
                                        h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        if FLASH_ATTN_AVAILABLE:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=qkv.device)
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            q, k, v = qkv.unbind(dim=2)
            x = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False)
            x = rearrange(x, 'b h s d -> b s (h d)')

        x = bias_dropout_scale_fn(self.attn_out(x),
                                                            None,
                                                            gate_msa,
                                                            x_skip,
                                                            self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim,
                                                                            2 * hidden_size,
                                                                            bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class APMDMDualHead(nn.Module):
    """    
    Implements the dual-head architecture described in the paper:
    1. Unmasking Head: Predicts original tokens for standard denoising
    2. Remasking Head: Predicts remasking probabilities for error correction
    
    This design allows the model to:
    - Generate new tokens through the unmasking head
    - Identify and correct erroneous tokens through the remasking head
    """
    def __init__(self, hidden_size, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        
        # Unmasking head: outputs logits of vocabulary size
        self.unmasking_head = nn.Linear(hidden_size, vocab_size)
        self.unmasking_head.weight.data.zero_()
        self.unmasking_head.bias.data.zero_()
        
        # Remasking head: outputs binary classification logits (whether to remask)
        self.remasking_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.remasking_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.remasking_head.bias, -2.0)  # Initially biased towards not remasking

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        """
        Forward pass, returns dual-head outputs
        
        Returns:
            dict: {
                'unmasking_logits': (batch_size, seq_len, vocab_size),
                'remasking_logits': (batch_size, seq_len, 1)
            }
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        
        unmasking_logits = self.unmasking_head(x)
        remasking_logits = self.remasking_head(x)
        
        return {
            'unmasking_logits': unmasking_logits,
            'remasking_logits': remasking_logits
        }


class APMDMTripleHead(nn.Module):
    """
    APMDM triple-head output layer - supports expansion functionality
    
    Implements the triple-head architecture described in the paper:
    1. Unmasking Head: Predicts original tokens for standard denoising
    2. Remasking Head: Predicts remasking probabilities for error correction
    3. Expansion Head: Predicts whether to insert new tokens after this position
    
    This design allows the model to:
    - Generate new tokens through the unmasking head
    - Identify and correct erroneous tokens through the remasking head
    - Dynamically adjust sequence length through the expansion head
    """
    def __init__(self, hidden_size, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        
        # Unmasking head: outputs logits of vocabulary size
        self.unmasking_head = nn.Linear(hidden_size, vocab_size)
        self.unmasking_head.weight.data.zero_()
        self.unmasking_head.bias.data.zero_()
        
        # Remasking head: outputs binary classification logits (whether to remask)
        self.remasking_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.remasking_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.remasking_head.bias, -2.0)  # Initially biased towards not remasking
        
        # Expansion head: outputs binary classification logits (whether expansion is needed)
        self.expansion_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.expansion_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.expansion_head.bias, -2.0)  # Initially biased towards no expansion

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        """
        Forward pass, returns triple-head outputs
        
        Returns:
            dict: {
                'unmasking_logits': (batch_size, seq_len, vocab_size),
                'remasking_logits': (batch_size, seq_len, 1),
                'expansion_logits': (batch_size, seq_len, 1)
            }
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        
        unmasking_logits = self.unmasking_head(x)
        remasking_logits = self.remasking_head(x)
        expansion_logits = self.expansion_head(x)
        
        return {
            'unmasking_logits': unmasking_logits,
            'remasking_logits': remasking_logits,
            'expansion_logits': expansion_logits
        }


class APMDMQuadrupleHead(nn.Module):
    """
    APMDM quadruple-head output layer - supports full APMDM functionality
    
    Implements the quadruple-head architecture described in the paper:
    1. Unmasking Head: Predicts original tokens for standard denoising
    2. Remasking Head: Predicts remasking probabilities for error correction
    3. Expansion Head: Predicts whether to insert new tokens after this position
    4. Contraction Head: Predicts whether this mask token is redundant (should be removed)
    
    This design allows the model to:
    - Generate new tokens through the unmasking head
    - Identify and correct erroneous tokens through the remasking head
    - Dynamically grow sequence length through the expansion head
    - Remove redundant mask tokens through the contraction head
    """
    def __init__(self, hidden_size, vocab_size, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        
        # Unmasking head: outputs logits of vocabulary size
        self.unmasking_head = nn.Linear(hidden_size, vocab_size)
        self.unmasking_head.weight.data.zero_()
        self.unmasking_head.bias.data.zero_()
        
        # Remasking head: outputs binary classification logits (whether to remask)
        self.remasking_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.remasking_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.remasking_head.bias, -2.0)  # Initially biased towards not remasking
        
        # Expansion head: outputs binary classification logits (whether expansion is needed)
        self.expansion_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.expansion_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.expansion_head.bias, -2.0)  # Initially biased towards no expansion
        
        # Contraction head: outputs binary classification logits (whether this mask is redundant)
        self.contraction_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.contraction_head.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.contraction_head.bias, -2.0)  # Initially biased towards not removing masks

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        """
        Forward pass, returns quadruple-head outputs
        
        Returns:
            dict: {
                'unmasking_logits': (batch_size, seq_len, vocab_size),
                'remasking_logits': (batch_size, seq_len, 1),
                'expansion_logits': (batch_size, seq_len, 1),
                'contraction_logits': (batch_size, seq_len, 1)
            }
        """
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        
        unmasking_logits = self.unmasking_head(x)
        remasking_logits = self.remasking_head(x)
        expansion_logits = self.expansion_head(x)
        contraction_logits = self.contraction_head(x)
        
        return {
            'unmasking_logits': unmasking_logits,
            'remasking_logits': remasking_logits,
            'expansion_logits': expansion_logits,
            'contraction_logits': contraction_logits
        }


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                                                            vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        
        self.rotary_emb = Rotary(
            config.model.hidden_size // config.model.n_heads)

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(config.model.hidden_size,
                                                            config.model.n_heads,
                                                            config.model.cond_dim,
                                                            dropout=config.model.dropout))
        self.blocks = nn.ModuleList(blocks)

        if hasattr(config, 'parameterization') and config.parameterization == 'apmdm':
            enable_expansion = (hasattr(config, 'apmdm') and 
                               hasattr(config.apmdm, 'enable_expansion') and 
                               config.apmdm.enable_expansion)
            
            enable_contraction = (hasattr(config, 'apmdm') and 
                                 hasattr(config.apmdm, 'enable_contraction') and 
                                 config.apmdm.enable_contraction)
            
            if enable_expansion and enable_contraction:
                self.output_layer = APMDMQuadrupleHead(
                    config.model.hidden_size,
                    vocab_size,
                    config.model.cond_dim)
                self.is_quadruple_head = True
                self.is_triple_head = False
                self.is_dual_head = False
            elif enable_expansion:
                self.output_layer = APMDMTripleHead(
                    config.model.hidden_size,
                    vocab_size,
                    config.model.cond_dim)
                self.is_triple_head = True
                self.is_quadruple_head = False
                self.is_dual_head = False
            else:
                self.output_layer = APMDMDualHead(
                    config.model.hidden_size,
                    vocab_size,
                    config.model.cond_dim)
                self.is_dual_head = True
                self.is_triple_head = False
                self.is_quadruple_head = False
        else:
            self.output_layer = DDitFinalLayer(
                config.model.hidden_size,
                vocab_size,
                config.model.cond_dim)
            self.is_dual_head = False
            self.is_triple_head = False
            self.is_quadruple_head = False
            
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return  bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            
            x = self.output_layer(x, c)

        return x
