import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils

LOG2 = math.log(2)

def sample_categorical(categorical_probs):
    gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def unsqueeze_like(x, reference):
    dims_to_add = len(reference.shape) - len(x.shape)
    new_shape = x.shape + (1,) * dims_to_add
    return x.view(*new_shape)

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor

class NLL(torchmetrics.MeanMetric):
    pass

class BPD(NLL):
    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)

class Diffusion(L.LightningModule):    
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # ===== Basic component initialization =====
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor
        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        
        # ===== Training strategy configuration =====
        self.antithetic_sampling = self.config.training.antithetic_sampling
        
        # ===== Mask token handling =====
        # If tokenizer doesn't have mask_token, add one at the end of vocabulary
        if (not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
            
        # ===== Parameterization method selection =====
        self.parameterization = self.config.parameterization
        
        # ===== Backbone model initialization =====
        if self.config.backbone == 'dit':
            self.backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.backbone == 'ar':
            self.backbone = models.autoregressive.AR(
                self.config, vocab_size=self.vocab_size, mask_index=self.mask_index)
        elif self.config.backbone == 'hf_dit':
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True)
        else:
            raise ValueError(f'Unknown backbone: {self.config.backbone}')

        self.softplus = torch.nn.Softplus()
        
        # ===== Evaluation metrics initialization =====
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        # ===== Generative perplexity evaluation =====
        self.gen_ppl_metric = Perplexity()
        self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.gen_ppl_eval_model_name_or_path)
        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = self.eval_model_tokenizer.eos_token_id

        # ===== Noise scheduler initialization =====
        self.noise = noise_schedule.get_noise(self.config, dtype=torch.float32)
        
        # ===== EMA model weight management =====
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema)
        else:
            self.ema = None

        # ===== Training parameters =====
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameter compatibility - check if parameterization method is supported"""
        assert self.parameterization in {'subs', 'apmdm', 'ar'}

    # ===============================
    # Lightning lifecycle callbacks
    # ===============================
    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        
        # Fix Lightning training progress tracking
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] = \
            checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] = \
            checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] = \
            checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed']
        
        # Save data sampler state
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if self.trainer.train_dataloader and hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        """Training start callback"""
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
        # Setup fault-tolerant sampler
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        
        # Update data loaders
        updated_dls = []
        if self.trainer.fit_loop._combined_loader:
            for dl in self.trainer.fit_loop._combined_loader.flattened:
                # The chunk loader already yields fully collated batches in
                # sequential chunk order and is re-iterable across epochs.
                # Replacing it with a random-index DataLoader makes every
                # worker reopen unrelated 10k-sample pickle chunks, leaving
                # the GPU idle at the first epoch boundary.  It also changes
                # the declared streaming data path.  Keep this loader intact;
                # the replacement below is only for ordinary datasets.
                if dataloader.is_sequential_chunk_loader(dl):
                    updated_dls.append(dl)
                    continue
                if hasattr(dl.sampler, 'shuffle'):
                    dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
                else:
                    dl_sampler = sampler_cls(dl.dataset)
                
                if (distributed and self.fast_forward_epochs is not None 
                    and self.fast_forward_batches is not None):
                    dl_sampler.load_state_dict({
                        'epoch': self.fast_forward_epochs,
                        'start_index': (self.fast_forward_batches * self.config.loader.batch_size)})
                
                # Keep original DataLoader's collate_fn to support custom batching logic like dynamic padding
                updated_dls.append(torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=self.config.loader.num_workers > 0,
                    collate_fn=getattr(dl, 'collate_fn', None)))
            self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def on_train_epoch_start(self):
        """Training epoch start"""
        self.backbone.train()
        self.noise.train()

    def on_validation_epoch_start(self):
        """Validation epoch start"""
        if self.ema:
            self.ema.store(itertools.chain(self.backbone.parameters(), self.noise.parameters()))
            self.ema.copy_to(itertools.chain(self.backbone.parameters(), self.noise.parameters()))
        self.backbone.eval()
        self.noise.eval()

    def on_validation_epoch_end(self):
        """Validation epoch end processing"""
        if ((self.config.eval.compute_perplexity_on_sanity or not self.trainer.sanity_checking)
            and self.config.eval.generate_samples and not self.parameterization == 'ar'):
            samples, text_samples = None, None
            for _ in range(self.config.sampling.num_sample_batches):
                samples = self.sample()
                text_samples = self.tokenizer.batch_decode(samples)
                if self.config.eval.compute_generative_perplexity:
                    self.compute_generative_perplexity(text_samples)
            
            if (self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table') 
                and text_samples is not None):
                text_samples = text_samples[: self.config.sampling.num_sample_log]
                self.trainer.logger.log_table(
                    key=f'samples@global_step{self.global_step}',
                    columns=['Generated Samples'],
                    data=[[s] for s in text_samples])
            
            if self.config.eval.compute_generative_perplexity:
                self.log('val/gen_ppl', self.gen_ppl_metric, on_epoch=True, on_step=False, sync_dist=True)
        
        if self.ema:
            self.ema.restore(itertools.chain(self.backbone.parameters(), self.noise.parameters()))

    def optimizer_step(self, *args, **kwargs):
        """Optimizer step callback"""
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters(), self.noise.parameters()))

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler, 'interval': 'step',
            'monitor': 'val/loss', 'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    # ===============================
    # Training and validation steps
    # ===============================
    def training_step(self, batch, batch_idx):
        """Single training step - PyTorch Lightning training interface: called once per batch
        Args: batch (batch data), batch_idx (batch index)
        Returns: loss value (for backpropagation)"""
        loss = self.compute_loss(batch, prefix='train')
        self.log(name='trainer/loss', value=loss.item(), on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step
        Args: batch (validation data), batch_idx (index)
        Returns: validation loss"""
        return self.compute_loss(batch, prefix='val')

    # ===============================
    # Evaluation related methods
    # ===============================
    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenize generated samples for evaluation model"""
        if 'llama2' in self.gen_ppl_eval_model_name_or_path:
            tokenizer_kwargs = {
                'text_samples': text_samples, 'return_tensors': 'pt',
                'return_token_type_ids': False, 'return_attention_mask': True,
                'truncation': True, 'padding': True, 'max_length': max_length,
            }
            eval_context_size = 4096
        else:
            tokenizer_kwargs = {
                'return_tensors': 'pt', 'return_token_type_ids': False,
                'return_attention_mask': True, 'truncation': True,
                'padding': True, 'max_length': max_length,
            }
            eval_context_size = 1024
        samples = self.eval_model_tokenizer(text_samples, **tokenizer_kwargs)
        attn_mask = samples['attention_mask']
        samples = samples['input_ids']
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            attn_mask = attn_mask.to(self.device)
            samples = samples.to(self.device)      
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def compute_generative_perplexity(self, text_samples: typing.List[str], 
                                    retokenize: bool = True,
                                    max_length: typing.Optional[int] = None) -> None:
        """Compute generative text perplexity - use pretrained AR model (GPT2) to evaluate generation quality, lower perplexity means closer to real language distribution
        Args: text_samples (list of texts), retokenize (whether to retokenize), max_length (maximum length)"""
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.gen_ppl_eval_model_name_or_path).eval()
        if max_length is None:
            max_length = self.config.model.length
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            eval_model = eval_model.to(self.device)
        
        if retokenize:
            (samples, attn_mask, eval_context_size) = self.eval_retokenize(text_samples, max_length=max_length)
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(self.device)
            eval_context_size = samples.shape[-1]
        
        batch_size = min(self.config.eval.perplexity_batch_size, samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size: (i + 1) * batch_size],
                eval_context_size, dim=-1)
            _attn_mask = torch.split(
                attn_mask[i * batch_size: (i + 1) * batch_size],
                eval_context_size, dim=-1)
            for (sample_chunk, attn_mask_chunk) in zip(_samples, _attn_mask):
                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
                logits = logits.transpose(-1, -2)
                nlls = F.cross_entropy(logits[..., :-1], sample_chunk[..., 1:], reduction='none')
                first_eos = (sample_chunk == self.eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
                token_mask = (sample_chunk != self.eval_model_tokenizer.eos_token_id)
                self.gen_ppl_metric.update(nlls, first_eos[..., 1:] + token_mask[..., 1:])

    # ===============================
    # Utility methods
    # ===============================
    def _process_sigma(self, sigma):
        """Process time/noise conditioning signal - standardize sigma parameter: AR mode can be None->squeeze to 1D->optionally disable time info"""
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, device):
        """Sample timesteps - support antithetic sampling to reduce variance
        Args: n (number of samples), device (device)
        Returns: timestep tensor"""
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        return t

    def _sample_prior(self, *batch_dims):
        """Sample initial state from prior distribution - in MDLM this is a fully masked sequence, diffusion endpoint is fully masked, reverse sampling starts from here
        Args: *batch_dims (tensor dimensions like batch_size, seq_len)
        Returns: fully masked sequence, all positions are mask_index"""
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _maybe_sub_sample(self, x0, attention_mask):
        """Subsampling processing - handle different sequence lengths and parameterization methods for data preprocessing
        Args: x0 (original sequence), attention_mask (attention mask)
        Returns: processed input/output sequences and mask"""
        seqlen = x0.shape[1]
        if seqlen > self.config.model.length:
            # Handle extra-long sequences (e.g. text8-crop dataset) - random crop to target length
            assert seqlen == 2 * self.config.model.length
            # cropping is needed for text8-crop dataset - try the same starting point for now
            start = np.random.choice(self.config.model.length)  # Random start position
            end = start + self.config.model.length
            input_tokens = x0[:, start: end]  # Input sequence
            output_tokens = x0[:, start + 1: end + 1]  # Target sequence (shifted right by 1)
            new_attention_mask = attention_mask[:, start: end]
            # Helps with validation PPL, since the val examples will all start and end with BOS/EOS
            input_tokens[:, 0] = self.tokenizer.bos_token_id  # Ensure start token
            output_tokens[:, -1] = self.tokenizer.eos_token_id  # Ensure end token
        elif self.parameterization == 'ar':
            # AR mode: standard input-output sequence pair (teacher forcing)
            input_tokens = x0[:, :-1]  # Input without last token
            output_tokens = x0[:, 1:]   # Output without first token
            new_attention_mask = attention_mask[:, 1:]  # Mask corresponds to output
        else:
            # Non-AR mode (SUBS/APMDM): input equals output, no need for sequence pair
            input_tokens = x0
            output_tokens = None  # No need for output sequence
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _ddpm_update(self, x, t, dt):
        """Standard DDPM update - implement denoising update formula p(x_{t-dt}|x_t), compute current and next timestep noise levels->model prediction->sample from denoising distribution
        Args: x (current state), t (current time), dt (timestep size)
        Returns: updated token sequence"""
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division term that is missing. This division term doesn't affect the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        """DDPM update with caching optimization - reuse previously computed p_x0 to accelerate sampling, avoid repeated forward passes when sequence state is unchanged
        Args: x (current state), t (current time), dt (timestep size), p_x0 (cached distribution)
        Returns: new p_x0 distribution and updated state"""
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    # ===============================
    # Loss computation methods
    # ===============================
    def compute_loss(self, batch, prefix):
        """Compute loss and update evaluation metrics - training/validation core: extract mask->compute diffusion loss->update metrics->log
        Args: batch (dict containing input_ids/attention_mask), prefix (train/val/test)
        Returns: scalar loss"""
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
        else:
            attention_mask = None
        
        # Detect whether to use precomputed labels (by checking if batch contains APMDM required label fields)
        use_precomputed_labels = self._should_use_precomputed_labels(batch)
        
        if use_precomputed_labels:
            # Loss computation using precomputed labels
            losses = self._compute_loss_internal_precomputed(batch, prefix)
        else:
            # Loss computation using traditional dynamic label generation
            losses = self._compute_loss_internal(batch['input_ids'], attention_mask, prefix)
            
        loss = losses.loss
        
        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')
        
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def _should_use_precomputed_labels(self, batch):
        """Detect whether to use precomputed labels
        Determined by checking if batch contains required fields for APMDM precomputed labels"""
        required_fields = ['target_ids', 'remask_labels', 'expansion_labels', 'contraction_labels']
        return all(field in batch for field in required_fields)
    
    def _compute_loss_internal_precomputed(self, batch, prefix):
        """Compute loss using precomputed labels - specifically for pre-generated APMDM data like demo.jsonl
        Args: batch (dict) containing all APMDM label fields, prefix (str) logging prefix
        Returns: Loss object (loss: scalar, nlls: dummy NLL, token_mask: mask)"""
        
        if self.parameterization != 'apmdm':
            raise ValueError(f"Precomputed labels only support APMDM parameterization, current: {self.parameterization}")
        
        # Use precomputed label version of APMDM forward pass
        loss_dict = self._forward_pass_apmdm_precomputed(batch)
        loss = loss_dict['total_loss']
        
        # Log losses for each head, consistent with dynamic version
        self.log(f'{prefix}/unmasking_loss', loss_dict['unmasking_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/remasking_loss', loss_dict['remasking_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/expansion_loss', loss_dict['expansion_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}/contraction_loss', loss_dict['contraction_loss'], 
                on_step=True, on_epoch=True, sync_dist=True)
        
        # Construct compatible Loss object
        attention_mask = batch['attention_mask']
        input_ids = batch['input_ids']
        # Create dummy nlls for metric computation compatibility
        nlls = torch.zeros_like(input_ids, dtype=torch.float) + loss.item()
        token_mask = attention_mask
        
        return Loss(loss=loss, nlls=nlls, token_mask=token_mask)

    def _compute_loss_internal(self, x0, attention_mask, prefix):
        """Compute final training loss - integrate all loss components: subsampling->compute loss based on parameterization->apply mask->normalize
        Args: x0 (batch,seq) original sequence, attention_mask (batch,seq) mask, prefix (str) logging prefix
        Returns: Loss object (loss: scalar, nlls: per-position negative log likelihood, token_mask: valid mask)"""
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(x0, attention_mask)  # Data preprocessing
        
        if self.parameterization == 'ar':
            # AR mode: standard language modeling loss - teacher forcing training
            logprobs = self.backbone(input_tokens, None)  # Get log probabilities
            loss = - logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]  # Negative log likelihood
        elif self.parameterization == 'apmdm':
            loss_dict = self._forward_pass_apmdm(input_tokens, attention_mask)
            loss = loss_dict['total_loss']  # Combined loss
            # Log APMDM individual losses for monitoring
            self.log(f'{prefix}/unmasking_loss', loss_dict['unmasking_loss'], 
                    on_step=True, on_epoch=True, sync_dist=True)
            self.log(f'{prefix}/remasking_loss', loss_dict['remasking_loss'], 
                    on_step=True, on_epoch=True, sync_dist=True)
            # Log expansion loss
            if 'expansion_loss' in loss_dict:
                self.log(f'{prefix}/expansion_loss', loss_dict['expansion_loss'], 
                        on_step=True, on_epoch=True, sync_dist=True)
            # Log contraction loss
            if 'contraction_loss' in loss_dict:
                self.log(f'{prefix}/contraction_loss', loss_dict['contraction_loss'], 
                        on_step=True, on_epoch=True, sync_dist=True)
        else:
            # SUBS diffusion training: use continuous time diffusion loss
            loss = self._forward_pass_mdlm(input_tokens)

        if self.parameterization == 'apmdm': 
            # For APMDM, loss is already scalar and attention_mask has been considered
            # Create dummy nlls for compatibility
            nlls = torch.zeros_like(input_tokens, dtype=torch.float) + loss.item()
            token_mask = attention_mask
            return Loss(loss=loss, nlls=nlls, token_mask=token_mask)
        else:
            # Apply attention mask and normalize
            nlls = loss * attention_mask  # Only compute loss for valid positions
            count = attention_mask.sum()  # Total number of valid tokens
            batch_nll = nlls.sum()  # Total batch loss
            token_nll = batch_nll / count  # Average loss per token
            return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)

    # ===============================
    # Forward pass methods
    # ===============================
    def forward(self, x, sigma):
        """Model forward pass - main entry point: process time conditioning->backbone gets logits->post-process based on parameterization method
        Args: x (batch,seq), sigma (batch,)
        Returns: single-head log probabilities or APMDM multi-head dict {'unmasking_logits', 'remasking_logits'}"""
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(x, sigma)
        
        if self.parameterization == 'apmdm':
            return model_output
        elif self.parameterization == 'subs':
            return self._subs_parameterization(logits=model_output, xt=x)
        return model_output

    def _forward_pass_mdlm(self, x0):
        """Diffusion forward pass - MDLM training core: sample timestep->add noise->model prediction->compute loss
        Args: x0 (batch,seq) clean sequence
        Returns: training loss (batch,seq)
        Formula: L_CE = E[-1/|{i:x_t^i=[M]}| * Σ_{i:x_t^i=[M]} log p_θ(x_0^i|x_t,t)]"""            
        t = self._sample_t(x0.shape[0], x0.device)  # Randomly sample timestep
        sigma, dsigma = self.noise(t)  # Compute noise level and its derivative
        unet_conditioning = sigma[:, None]  # Time conditioning signal (batch, 1)
        move_chance = 1 - torch.exp(-sigma[:, None])  # Compute masking probability/noise probability
        xt = self._create_masked_sequence(x0, move_chance)  # Forward diffusion process: generate noisy data
        model_output = self.forward(xt, unet_conditioning)  # Model predicts log p_θ(x_0|x_t,t)
        utils.print_nans(model_output, 'model_output')  # NaN check
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)  # Extract log probability of target token
        return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]  # Apply continuous time weighting

    # APMDM training
    def _forward_pass_apmdm(self, x0, attention_mask):
        """APMDM multi-head training - unmasking, remasking, expansion and contraction loss computation
        Args: x0 (batch,seq) original sequence, attention_mask (batch,seq) mask
        Returns: loss dict {unmasking_loss, remasking_loss, expansion_loss, contraction_loss, total_loss}"""
        
        # 1. Unmasking training data preparation - standard diffusion process
        t_unmask = self._sample_t(x0.shape[0], x0.device)  # Sample timestep
        sigma_unmask, _ = self.noise(t_unmask)  # Get noise parameters (dsigma unused)
        unet_conditioning_unmask = sigma_unmask[:, None]  # Time conditioning
        move_chance_unmask = 1 - torch.exp(-sigma_unmask[:, None])  # Compute masking probability (unmask)
        xt_unmask = self._create_masked_sequence(x0, move_chance_unmask)  # Generate masked sequence
        
        # 2. Remasking training data preparation - based on randomly corrupted sequence
        t_remask = self._sample_t(x0.shape[0], x0.device)  # Independently sample timestep
        sigma_remask, _ = self.noise(t_remask)  # Get noise parameters
        unet_conditioning_remask = sigma_remask[:, None]  # Time conditioning
        corruption_ratio = 1 - torch.exp(-sigma_remask[:, None])  # Compute corruption ratio
        xt_remask = self._create_corrupted_sequence(x0, corruption_ratio, "shuffle")  # Generate corrupted sequence (Shuffle implements vocab_freq effect)
        
        # 3. Expansion training data preparation - based on deflated sequence
        deflated_sequences, expansion_labels, _ = self._create_deflated_sequence_and_labels(x0, attention_mask)
        
        # Check if there are valid deflated sequences for expansion training
        has_valid_expansion = False
        if len(deflated_sequences) > 0:
            # Filter empty sequences
            valid_sequences = [seq for seq in deflated_sequences if len(seq) > 0]
            if len(valid_sequences) > 0:
                max_deflated_len = max(len(seq) for seq in valid_sequences)
                # Ensure deflated sequence length doesn't exceed original sequence length
                max_allowed_len = min(max_deflated_len, x0.shape[1])
                
                # Convert to batch tensor format
                xt_expand, expansion_labels_tensor, expansion_mask = self._pad_sequences_for_batch(
                    deflated_sequences, expansion_labels, max_len=max_allowed_len)
                xt_expand = xt_expand.to(x0.device)
                expansion_labels_tensor = expansion_labels_tensor.to(x0.device)
                expansion_mask = expansion_mask.to(x0.device)
                
                # Check if there is valid expansion data
                if expansion_mask.any():
                    has_valid_expansion = True
                    # Time conditioning for expansion
                    t_expand = self._sample_t(x0.shape[0], x0.device)
                    sigma_expand, _ = self.noise(t_expand)
                    unet_conditioning_expand = sigma_expand[:, None]

        # 4. Contraction training data preparation - based on two-step masking process
        contracted_sequences, contraction_labels, _ = self._create_contracted_sequence_and_labels(x0, attention_mask)
        
        # Check if there are valid contracted sequences for contraction training
        has_valid_contraction = False
        if len(contracted_sequences) > 0:
            # Filter empty sequences
            valid_contracted_sequences = [seq for seq in contracted_sequences if len(seq) > 0]
            if len(valid_contracted_sequences) > 0:
                max_contracted_len = max(len(seq) for seq in valid_contracted_sequences)
                # Ensure contracted sequence length doesn't exceed original sequence length
                max_allowed_contracted_len = min(max_contracted_len, x0.shape[1])
                
                # Convert to batch tensor format
                xt_contract, contraction_labels_tensor, contraction_mask = self._pad_sequences_for_batch(
                    contracted_sequences, contraction_labels, max_len=max_allowed_contracted_len)
                xt_contract = xt_contract.to(x0.device)
                contraction_labels_tensor = contraction_labels_tensor.to(x0.device)
                contraction_mask = contraction_mask.to(x0.device)
                
                # Check if there is valid contraction data
                if contraction_mask.any():
                    has_valid_contraction = True
                    # Time conditioning for contraction
                    t_contract = self._sample_t(x0.shape[0], x0.device)
                    sigma_contract, _ = self.noise(t_contract)
                    unet_conditioning_contract = sigma_contract[:, None]

        # 5. Model forward pass - get predictions
        model_output_unmask = self.forward(xt_unmask, unet_conditioning_unmask)
        model_output_remask = self.forward(xt_remask, unet_conditioning_remask)
        
        # Perform expansion forward pass only when valid expansion data exists
        if has_valid_expansion:
            model_output_expand = self.forward(xt_expand, unet_conditioning_expand)
            
        # Perform contraction forward pass only when valid contraction data exists
        if has_valid_contraction:
            model_output_contract = self.forward(xt_contract, unet_conditioning_contract)
        
        # 6. Compute unmasking loss - consistent with _forward_pass_mdlm
        unmasking_logits = model_output_unmask['unmasking_logits']
        # Apply SUBS constraint processing, consistent with original method
        unmasking_logits_processed = self._subs_parameterization(unmasking_logits, xt_unmask)
        # Use the same loss computation as _forward_pass_mdlm
        log_p_theta = torch.gather(input=unmasking_logits_processed, dim=-1, index=x0[:, :, None]).squeeze(-1)
        # [Optimization] Assuming loglinear schedule, weight dsigma / expm1(sigma) can be simplified to 1/t
        # t_unmask sampling is guaranteed non-zero by self.sampling_eps, can safely use as denominator
        weighted_loss = - log_p_theta / t_unmask[:, None]
        
        # Apply attention_mask, compute loss only for valid positions
        masked_weighted_loss = weighted_loss * attention_mask
        valid_token_count = attention_mask.sum()
        unmasking_loss = masked_weighted_loss.sum() / valid_token_count if valid_token_count > 0 else 0.0
        
        # 7. Compute remasking loss - binary classification task to predict if remask is needed
        remasking_logits = model_output_remask['remasking_logits'].squeeze(-1)
        remasking_labels = (x0 != xt_remask).float()  # 1 if corrupted, 0 if correct
        
        # Apply attention_mask, compute loss only for valid positions
        # Compute BCE loss for each position
        pointwise_bce_loss = F.binary_cross_entropy_with_logits(
            remasking_logits, remasking_labels, reduction='none')
        # Apply attention_mask and normalize
        masked_bce_loss = pointwise_bce_loss * attention_mask
        remasking_loss = masked_bce_loss.sum() / valid_token_count if valid_token_count > 0 else 0.0
        
        # 8. Compute expansion loss - binary classification task to predict if expansion is needed
        if has_valid_expansion:
            expansion_logits = model_output_expand['expansion_logits'].squeeze(-1)
            
            batch_size_expand, seq_len_expand = expansion_labels_tensor.shape
            batch_size_logits, seq_len_logits = expansion_logits.shape
            
            if batch_size_expand != batch_size_logits:
                # Batch size mismatch, skip expansion loss computation
                expansion_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
            else:
                # Adjust sequence length to the smaller value
                min_seq_len = min(seq_len_expand, seq_len_logits)
                expansion_logits = expansion_logits[:, :min_seq_len]
                expansion_labels_tensor = expansion_labels_tensor[:, :min_seq_len]
                expansion_mask = expansion_mask[:, :min_seq_len]
                
                # Compute BCE loss
                pointwise_expansion_loss = F.binary_cross_entropy_with_logits(
                    expansion_logits, expansion_labels_tensor, reduction='none')
                
                # Apply expansion_mask and normalize
                masked_expansion_loss = pointwise_expansion_loss * expansion_mask
                valid_expansion_count = expansion_mask.sum()
                
                if valid_expansion_count > 0:
                    expansion_loss = masked_expansion_loss.sum() / valid_expansion_count
                else:
                    expansion_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
        else:
            expansion_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
        
        # 9. Compute contraction loss - binary classification task to predict if mask is redundant
        if has_valid_contraction:
            contraction_logits = model_output_contract['contraction_logits'].squeeze(-1)
            
            # Shape matching check
            batch_size_contract, seq_len_contract = contraction_labels_tensor.shape
            batch_size_logits_contract, seq_len_logits_contract = contraction_logits.shape
            
            if batch_size_contract != batch_size_logits_contract:
                # Batch size mismatch, skip contraction loss computation
                contraction_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
            else:
                # Adjust sequence length to the smaller value
                min_seq_len_contract = min(seq_len_contract, seq_len_logits_contract)
                contraction_logits = contraction_logits[:, :min_seq_len_contract]
                contraction_labels_tensor = contraction_labels_tensor[:, :min_seq_len_contract]
                contraction_mask = contraction_mask[:, :min_seq_len_contract]
                
                # Compute BCE loss
                pointwise_contraction_loss = F.binary_cross_entropy_with_logits(
                    contraction_logits, contraction_labels_tensor, reduction='none')
                
                # Apply contraction_mask and normalize
                masked_contraction_loss = pointwise_contraction_loss * contraction_mask
                valid_contraction_count = contraction_mask.sum()
                
                if valid_contraction_count > 0:
                    contraction_loss = masked_contraction_loss.sum() / valid_contraction_count
                else:
                    contraction_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
        else:
            contraction_loss = torch.tensor(0.0, device=x0.device, requires_grad=True)
        
        # 10. Combine losses - weighted average of four losses
        lambda_remask = getattr(self.config.apmdm, 'lambda_remask', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        lambda_expand = getattr(self.config.apmdm, 'lambda_expand', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        lambda_contract = getattr(self.config.apmdm, 'lambda_contract', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        
        total_loss = unmasking_loss + lambda_remask * remasking_loss + lambda_expand * expansion_loss + lambda_contract * contraction_loss
        
        return {
            'unmasking_loss': unmasking_loss,
            'remasking_loss': remasking_loss,
            'expansion_loss': expansion_loss,
            'contraction_loss': contraction_loss,
            'total_loss': total_loss
        }

    def _forward_pass_apmdm_precomputed(self, batch):
        """APMDM precomputed label training
        Args: batch (dict) contains input_ids, target_ids, remask_labels, expansion_labels, contraction_labels, attention_mask
        Returns: loss dict {unmasking_loss, remasking_loss, expansion_loss, contraction_loss, total_loss}"""
        
        # 1. Extract precomputed data
        input_ids = batch['input_ids']  # x_k - current state sequence
        target_ids = batch['target_ids']  # y_star - target sequence  
        remask_labels = batch['remask_labels']  # r_star - remasking labels 
        expansion_labels = batch['expansion_labels']  # e_star - expansion labels
        contraction_labels = batch['contraction_labels']  # c_star - contraction labels
        attention_mask = batch['attention_mask']  # Valid position mask
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 2. Precomputed label mode uses zero time conditioning
        zero_sigma = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
        model_output = self.forward(input_ids, zero_sigma)
        
        # 3. Extract head logits (some heads may not be implemented, need null checks)
        unmasking_logits = model_output['unmasking_logits']                 # [batch, seq, vocab]
        remasking_logits = model_output.get('remasking_logits', None)       # [batch, seq, 1] or None
        expansion_logits = model_output.get('expansion_logits', None)       # [batch, seq, 1] or None
        contraction_logits = model_output.get('contraction_logits', None)   # [batch, seq, 1] or None
        
        # 4. Compute Unmasking loss (standard APMDM flow)
        unmasking_logits_processed = self._subs_parameterization(unmasking_logits, input_ids)
        log_p_theta = torch.gather(input=unmasking_logits_processed, dim=-1, index=target_ids[:, :, None]).squeeze(-1)
        
        # Create more precise loss mask: only compute loss for masked positions where target is not mask
        should_compute_loss = (input_ids == self.mask_index) & (target_ids != self.mask_index) & attention_mask.bool()
        
        log_p_theta_masked = torch.where(should_compute_loss, log_p_theta, torch.tensor(0.0, device=device))
        weighted_loss = - log_p_theta_masked
        
        # Only compute average loss for positions that need learning
        valid_token_count = should_compute_loss.sum()
        if valid_token_count > 0:
            unmasking_loss = weighted_loss.sum() / valid_token_count
        else:
            # When there are no MASK tokens, create a dummy gradient connection to avoid DDP errors
            # Use mean of unmasking_logits multiplied by 0, maintaining gradient connection without affecting training
            unmasking_loss = (unmasking_logits.mean() * 0.0)
        
        # Helper function: compute binary classification loss (BCE + attention mask normalization)
        def compute_binary_loss(logits, labels, name="binary"):
            if logits is None or labels is None:
                return torch.tensor(0.0, device=device, requires_grad=True)
            logits_squeezed = logits.squeeze(-1)
            # Fix: for binary classification tasks, should use all valid positions, not just MASK positions
            valid_positions = attention_mask.sum()
            if valid_positions > 0:
                pointwise_loss = F.binary_cross_entropy_with_logits(
                    logits_squeezed, labels.float(), reduction='none')
                masked_loss = pointwise_loss * attention_mask
                return masked_loss.sum() / valid_positions
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 5. Compute three binary classification losses
        remasking_loss = compute_binary_loss(remasking_logits, remask_labels, "remasking")
        expansion_loss = compute_binary_loss(expansion_logits, expansion_labels, "expansion") 
        contraction_loss = compute_binary_loss(contraction_logits, contraction_labels, "contraction")
        
        # 6. Combine losses - use the same weight configuration
        lambda_remask = getattr(self.config.apmdm, 'lambda_remask', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        lambda_expand = getattr(self.config.apmdm, 'lambda_expand', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        lambda_contract = getattr(self.config.apmdm, 'lambda_contract', 1.0) if hasattr(self.config, 'apmdm') else 1.0
        
        total_loss = (unmasking_loss + 
                     lambda_remask * remasking_loss + 
                     lambda_expand * expansion_loss + 
                     lambda_contract * contraction_loss)
        
        return {
            'unmasking_loss': unmasking_loss,
            'remasking_loss': remasking_loss,
            'expansion_loss': expansion_loss,
            'contraction_loss': contraction_loss,
            'total_loss': total_loss
        }

    # ===============================
    # Parameterization and noise-related methods
    # ===============================
    def _subs_parameterization(self, logits, xt):
        """SUBS parameterization - paper core innovation: forbid predicting [MASK], non-mask positions stay unchanged, mask positions can predict any token
        Args: logits (batch,seq,vocab), xt (batch,seq)
        Returns: log probability distribution
        Formula: p_θ(x_0^i|x_t,t) = Cat(x_0^i; softmax(logits)) only valid when x_t^i=[MASK]"""
        # Step 1: Set mask token probability to -∞ (not selectable)
        logits[:, :, self.mask_index] += self.neg_infinity
        # Step 2: Normalize logits to make it a valid probability distribution
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Step 3: For non-mask positions, only keep original token's probability, set all other tokens' probabilities to -∞, original token probability to 0 (i.e. probability is 1)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _create_masked_sequence(self, x, move_chance):
        """Forward diffusion process - generate noisy data from clean data, implement x_0 to x_t transformation based on absorbing mask kernel stochastic process
        Args: x (batch,seq) clean sequence, move_chance (batch,1) masking probability
        Returns: noisy sequence
        Formula: q(x_t^i|x_0^i) = (1-α_t)*δ_[MASK] + α_t*δ_{x_0^i}, α_t=1-move_chance"""
        # Generate random masking decisions - each position independently decides whether to mask
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        # Apply masking: selected positions replaced with [MASK], unselected positions keep original token
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _create_corrupted_sequence(self, x0, corruption_ratio, corruption_strategy='shuffle'):
        """Create randomly corrupted sequence for remasking training - supports uniform and shuffle strategies
        Args: 
            x0 (batch,seq) original sequence, 
            corruption_ratio (batch,1) corruption ratio,
            corruption_strategy (str): 'uniform' uses random tokens, 'shuffle' uses intra-batch shuffle (implements paper's vocab_freq effect)
        Returns: corrupted sequence
        """
        # Create random corruption mask - each position independently decides whether to corrupt
        corrupt_mask = torch.rand_like(x0, dtype=torch.float) < corruption_ratio
        
        if corruption_strategy == 'shuffle':
            # Use intra-batch shuffle strategy - implements paper's vocab_freq effect
            xt_corrupt = self._create_corrupted_sequence_shuffle(x0, corrupt_mask)
        else:
            # Use uniform random sampling strategy
            xt_corrupt = self._create_corrupted_sequence_uniform(x0, corrupt_mask)
        
        return xt_corrupt
    
    def _create_corrupted_sequence_uniform(self, x0, corrupt_mask):
        """Create corrupted sequence using uniform random sampling"""
        # Generate random tokens to replace corrupted positions (excluding mask token)
        if self.mask_index < self.vocab_size:
            # mask token is within vocabulary, need to exclude - elegant solution: sample [0, vocab_size-1), then skip mask_index
            random_tokens = torch.randint(0, self.vocab_size - 1, x0.shape, device=x0.device)
            # Increment values >= mask_index by 1, skip mask_index position
            random_tokens = torch.where(random_tokens >= self.mask_index, random_tokens + 1, random_tokens)
        else:
            # mask token is outside vocabulary, directly sample
            random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=x0.device)
        # Apply corruption: selected positions replaced with random token, other positions keep original token
        xt_corrupt = torch.where(corrupt_mask, random_tokens, x0)
        return xt_corrupt
        
    def _create_corrupted_sequence_shuffle(self, x0, corrupt_mask):
        """Create corrupted sequence using intra-batch shuffle strategy - maintain token distribution while performing corruption"""
        # Find all positions that need corruption
        corrupt_positions = torch.where(corrupt_mask.view(-1))[0]
        if len(corrupt_positions) <= 1:
            return x0.clone()  # No need to corrupt or only 1 position cannot shuffle
        
        # Extract tokens that need corruption and shuffle (avoid returning to original position)
        x0_flat = x0.view(-1)
        corrupt_tokens = x0_flat[corrupt_positions]
        n = len(corrupt_tokens)
        
        # Smart shuffle: prioritize random permutation, fix conflicts if failed, use cyclic shift in worst case
        indices = torch.arange(n, device=x0.device)
        for attempt in range(100):  # Maximum 100 attempts
            perm = torch.randperm(n, device=x0.device)
            if not torch.any(perm == indices):  # No token returns to original position
                break
            if attempt == 99:  # Last attempt: fix conflicts
                conflicts = torch.where(perm == indices)[0]
                if len(conflicts) >= 2:  # Swap conflicting positions
                    perm[conflicts[0]], perm[conflicts[1]] = perm[conflicts[1]], perm[conflicts[0]]
                else:  # Single conflict: cyclic shift
                    perm = (indices + 1) % n
        
        # Apply shuffle result
        xt_corrupt = x0.clone()
        xt_corrupt.view(-1)[corrupt_positions] = corrupt_tokens[perm]
        return xt_corrupt

    def _create_deflated_sequence_and_labels(self, x0, attention_mask):
        """Create deflated sequence and expansion labels - implement Expansion Loss data preparation in the paper
        
        Args:
            x0 (batch, seq): original sequence
            attention_mask (batch, seq): attention mask
        Returns:
            deflated_sequences (list): deflated sequence for each sample
            expansion_labels (list): expansion labels for each sample
            original_positions (list): original position mapping
        """
        batch_size, seq_len = x0.shape
        deflated_sequences = []
        expansion_labels = []
        original_positions = []
        
        for batch_idx in range(batch_size):
            # 1. Sample deletion probability δ_t ~ Uniform(0, 1)
            delta_t = torch.rand(1).item()
            
            # 2. Generate deletion indicator for each position D^i ~ Bernoulli(δ_t)
            # Only delete valid positions (attention_mask=1)
            valid_positions = attention_mask[batch_idx].bool()
            seq = x0[batch_idx][valid_positions]  # Get valid sequence
            n = len(seq)
            
            if n <= 1:  # Sequence too short, skip
                deflated_sequences.append(seq)
                expansion_labels.append(torch.zeros(len(seq), dtype=torch.float32))
                original_positions.append(torch.arange(len(seq)))
                continue
            
            # Generate deletion indicators (keep at least one token)
            deletion_indicators = torch.bernoulli(torch.full((n,), delta_t)).bool()
            # Ensure at least one token is kept
            if deletion_indicators.all():
                deletion_indicators[0] = False  # Keep the first token
            
            # 3. Create deflated sequence x̃ = {x_0^i : D^i = 0}
            remaining_mask = ~deletion_indicators  # R = {i : D^i = 0}
            deflated_seq = seq[remaining_mask]
            remaining_indices = torch.where(remaining_mask)[0]  # Original indices of remaining positions
            
            # 4. Compute expansion labels
            expansion_lbls = torch.zeros(len(deflated_seq), dtype=torch.float32)
            for idx, orig_pos in enumerate(remaining_indices):
                # e^j = 1[D^{j+1} = 1] if j+1 ≤ n, else 0
                if orig_pos + 1 < n:  # j+1 ≤ n
                    expansion_lbls[idx] = float(deletion_indicators[orig_pos + 1])  # 1[D^{j+1} = 1]
                else:
                    expansion_lbls[idx] = 0.0  # End of sequence
            
            deflated_sequences.append(deflated_seq)
            expansion_labels.append(expansion_lbls)
            original_positions.append(remaining_indices)
        
        return deflated_sequences, expansion_labels, original_positions

    def _create_contracted_sequence_and_labels(self, x0, attention_mask):
        """Create contracted sequence and contraction labels - used during training"""
        # Keep this function as it's needed during training
        batch_size, seq_len = x0.shape
        contracted_sequences = []
        contraction_labels = []
        
        for batch_idx in range(batch_size):
            valid_positions = attention_mask[batch_idx].bool()
            seq = x0[batch_idx][valid_positions]
            n = len(seq)
            
            if n <= 1:
                contracted_sequences.append(seq)
                contraction_labels.append(torch.zeros(len(seq), dtype=torch.float32))
                continue
            
            # Simplified label generation
            t = torch.rand(1).item()
            mask_prob = 1 - t
            
            mask_indicators = torch.bernoulli(torch.full((n,), mask_prob)).bool()
            x_contracted = seq.clone()
            x_contracted[mask_indicators] = self.mask_index
            
            contraction_lbls = torch.zeros(n, dtype=torch.float32)
            # Randomly mark some masks as redundant
            if mask_indicators.any():
                redundant_prob = 0.3
                redundant_masks = torch.bernoulli(torch.full((mask_indicators.sum(),), redundant_prob)).bool()
                contraction_lbls[mask_indicators] = redundant_masks.float()
            
            contracted_sequences.append(x_contracted)
            contraction_labels.append(contraction_lbls)
        
        return contracted_sequences, contraction_labels, []

    def _pad_sequences_for_batch(self, sequences, labels, max_len=None):
        """Pad variable-length sequences to uniform length for batch processing
        
        Args:
            sequences (list): list of variable-length sequences
            labels (list): corresponding label list
            max_len (int): maximum length, None to use maximum length within batch
        Returns:
            padded_sequences (tensor): padded sequences (batch, max_len)
            padded_labels (tensor): padded labels (batch, max_len) 
            sequence_masks (tensor): valid position mask (batch, max_len)
        """
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        batch_size = len(sequences)
        padded_sequences = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
        sequence_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, (seq, lbls) in enumerate(zip(sequences, labels)):
            seq_len = len(seq)
            if seq_len > 0:
                padded_sequences[i, :seq_len] = seq
                padded_labels[i, :seq_len] = lbls
                sequence_masks[i, :seq_len] = True
        
        return padded_sequences, padded_labels, sequence_masks
    
    # ===============================
    # Sampling methods
    # ===============================
    @torch.no_grad()
    def sample(self, num_steps=None, eps=1e-5):
        """Generate samples from model - unified sampling interface: supports AR/APMDM/DDPM/DDPM_cache samplers
        Args: num_steps (sampling steps), eps (minimum time value)
        Returns: generated token sequence (batch,seq)"""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        
        if self.parameterization == 'ar':
            return self._sample_ar(batch_size_per_gpu)
        elif self.parameterization == 'apmdm' or self.sampler == 'apmdm':
            return self._sample_apmdm(max_steps=num_steps)
        else:
            return self._sample_ddpm(num_steps, eps)

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Restore model and generate samples - temporarily switch to EMA weights for high-quality sampling, then restore training weights"""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(itertools.chain(self.backbone.parameters(), self.noise.parameters()))
            self.ema.copy_to(itertools.chain(self.backbone.parameters(), self.noise.parameters()))
        self.backbone.eval()
        self.noise.eval()
        samples = self.sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(itertools.chain(self.backbone.parameters(), self.noise.parameters()))
        self.backbone.train()
        self.noise.train()
        return samples

    def _sample_ar(self, bsz):
        """Autoregressive sampler - traditional left-to-right token generation, using Gumbel noise to improve diversity
        Args: bsz (batch size)
        Returns: generated sequence (batch_size,seq_len)"""
        num_pred_tokens = self.config.model.length - 1
        x = torch.zeros((bsz, num_pred_tokens + 1), dtype=torch.long, device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        noise = torch.distributions.Gumbel(0, 1).sample((bsz, num_pred_tokens, self.vocab_size)).to(self.device)
        
        for i in range(num_pred_tokens):
            next_logits = self.forward(x[:, :i + 1], None)[:, -1]
            y = (next_logits + noise[:, i]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _sample_ddpm(self, num_steps=None, eps=1e-5):
        """DDPM sampling - standard diffusion denoising sampling process
        Args: num_steps (sampling steps), eps (minimum time value)
        Returns: generated token sequence (batch,seq)"""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        # Step 1: Sample initial state from prior distribution (all masks)
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
        # Step 2: Set timestep sequence (decreasing from 1 to eps)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None  # For caching optimization
        
        # Step 3: Iterative denoising process
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            # Choose different sampling strategies based on configuration
            if self.sampler == 'ddpm_cache':
                # DDPM sampling with caching, can reuse computation results in some cases
                p_x0_cache, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    # Disable caching when state changes or using time conditioning
                    p_x0_cache = None
                x = x_next
            else:
                # Default to standard DDPM sampling
                x = self._ddpm_update(x, t, dt)
        
        # Step 4: Optional final denoising step
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            # Use model to directly predict final result
            unet_conditioning = self.noise(t)[0]
            model_output = self.forward(x, unet_conditioning)
            if isinstance(model_output, dict):
                x = model_output['unmasking_logits'].argmax(dim=-1)
            else:
                # Standard single-head output
                x = model_output.argmax(dim=-1)
        return x

    def _sample_apmdm(self, max_steps=None):
        """APMDM iterative sampler - refine sequences iteratively through Contraction→Unmask/Remask→Expansion operations, predict all decisions in single forward then apply transformations each step, until EOS or max steps
        Args: max_steps (maximum iteration steps, default 100)
        Returns: generated token sequence (batch,seq_len)"""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        max_steps = getattr(self.config.sampling, 'max_steps', 100) if max_steps is None else max_steps
        eos_token = getattr(self.tokenizer, 'eos_token_id', getattr(self.tokenizer, 'sep_token_id', self.vocab_size - 1))
        
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
        
        for step in range(max_steps):
            # Single forward to get all decisions from four heads
            model_output = self.forward(x, torch.zeros(x.shape[0], 1, device=self.device))
            # Apply SUBS parameterization: forbid predicting MASK token, keep non-MASK positions unchanged
            raw_unmasking_logits = model_output['unmasking_logits']
            subs_unmasking_logits = self._subs_parameterization(raw_unmasking_logits, x)
            y_star = subs_unmasking_logits.argmax(dim=-1)
            r_star = torch.sigmoid(model_output['remasking_logits'].squeeze(-1)) > 0.5
            e_star = torch.sigmoid(model_output['expansion_logits'].squeeze(-1)) > 0.5
            c_star = torch.sigmoid(model_output['contraction_logits'].squeeze(-1)) > 0.5
            
            # Apply sequence transformation x_k -> x_{k+1} in batch following demo.py logic
            x_k_plus_1_list = []
            for b in range(x.shape[0]):
                x_k_plus_1_sequence = []
                seq_len = x.shape[1]
                for i in range(seq_len):
                    # Contraction: skip mask positions to be contracted (with boundary check)
                    if i < c_star.shape[1] and c_star[b, i] and x[b, i] == self.mask_index:
                        continue
                    # Decide token at current position: Remask > Unmask > Keep
                    if i < r_star.shape[1] and r_star[b, i]:
                        token_to_add = self.mask_index
                    elif x[b, i] == self.mask_index:
                        token_to_add = y_star[b, i] if i < y_star.shape[1] else self.mask_index
                    else:
                        token_to_add = x[b, i]
                    x_k_plus_1_sequence.append(token_to_add.item() if torch.is_tensor(token_to_add) else token_to_add)
                    # Expansion: insert mask after current position (with boundary check)
                    if i < e_star.shape[1] and e_star[b, i]:
                        x_k_plus_1_sequence.append(self.mask_index)
                
                # Prevent empty sequence
                if not x_k_plus_1_sequence:
                    x_k_plus_1_sequence = [self.mask_index]
                x_k_plus_1_list.append(torch.tensor(x_k_plus_1_sequence, device=self.device))
            
            # Reassemble batch (handle variable-length sequence padding)
            if x.shape[0] == 1:
                x = x_k_plus_1_list[0].unsqueeze(0)
            else:
                max_len = max(len(seq) for seq in x_k_plus_1_list)
                x = torch.stack([torch.cat([seq, torch.full((max_len - len(seq),), self.mask_index, device=self.device)]) 
                                if len(seq) < max_len else seq for seq in x_k_plus_1_list])
            
            # EOS stop check
            if eos_token is not None and (x == eos_token).any():
                break

        # Final cleanup: fill remaining masks
        masked_positions = (x == self.mask_index)
        if masked_positions.any():
            final_output = self.forward(x, torch.zeros(x.shape[0], 1, device=self.device))
            # Apply SUBS parameterization to ensure not predicting MASK token
            final_subs_logits = self._subs_parameterization(final_output['unmasking_logits'], x)
            x[masked_positions] = final_subs_logits[masked_positions].argmax(dim=-1)
        
        return x
    
    def predict_operations(self, x, r_threshold=0.5, e_threshold=0.5, c_threshold=0.5):
        """
        Predict APMDM's four operations: unmasking, remasking, expansion, contraction
        
        Args:
            x: input sequence [batch_size, seq_len]
            r_threshold: threshold for remasking operation (default 0.5)
            e_threshold: threshold for expansion operation (default 0.5)
            c_threshold: threshold for contraction operation (default 0.5)
            
        Returns:
            dict: dictionary containing various prediction results
                - y_star_pred: unmasking prediction [batch_size, seq_len]
                - r_star_pred: remasking prediction [batch_size, seq_len] (0/1)
                - e_star_pred: expansion prediction [batch_size, seq_len] (0/1)
                - c_star_pred: contraction prediction [batch_size, seq_len] (0/1)
        """
        with torch.no_grad():
            # Use time t=0 for prediction
            sigma = torch.zeros(x.shape[0], 1, device=x.device)
            
            # Get model output
            model_output = self.forward(x, sigma)
            
            # Extract various predictions
            # Apply SUBS parameterization to unmasking logits to avoid predicting MASK token
            unmasking_logits_processed = self._subs_parameterization(model_output['unmasking_logits'], x)
            y_star_pred = unmasking_logits_processed.argmax(dim=-1)  # [batch, seq]
            
            r_star_pred = (torch.sigmoid(model_output['remasking_logits'].squeeze(-1)) > r_threshold).long()  # [batch, seq]
            e_star_pred = (torch.sigmoid(model_output['expansion_logits'].squeeze(-1)) > e_threshold).long()  # [batch, seq]
            c_star_pred = (torch.sigmoid(model_output['contraction_logits'].squeeze(-1)) > c_threshold).long()  # [batch, seq]
            
            return {
                'y_star_pred': y_star_pred,
                'r_star_pred': r_star_pred,
                'e_star_pred': e_star_pred,
                'c_star_pred': c_star_pred
            }
