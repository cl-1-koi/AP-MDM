import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import utils

# Register OmegaConf resolvers
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
    """Load trained diffusion model from checkpoint"""
    if 'hf' in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
    """Print and save config tree using Rich library"""
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)

        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    rich.print(tree)

    if save_cfg:
        config_save_path = '{}/config_tree.txt'.format(config.checkpointing.save_dir)
        with fsspec.open(config_save_path, 'w') as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    """Print batch information for debugging"""
    for dl_type, dl in [('train', train_ds), ('valid', valid_ds)]:
        print(f'\n=== {dl_type} dataloader batch ===')

        batch = next(iter(dl))
        print(f'input_ids shape: {batch["input_ids"].shape}')

        first_tokens = batch['input_ids'][0, :k]
        last_tokens = batch['input_ids'][0, -k:]

        print(f'First {k} tokens:')
        print(f'  Text: {tokenizer.decode(first_tokens)}')
        print(f'  IDs: {first_tokens.tolist()}')

        print(f'Last {k} tokens:')
        print(f'  Text: {tokenizer.decode(last_tokens)}')
        print(f'  IDs: {last_tokens.tolist()}')


def generate_samples(config, logger, tokenizer):
    """Generate text samples and compute generative perplexity"""
    logger.info('Generating text samples...')

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()

    if config.eval.disable_ema:
        logger.info('Disabling EMA weights.')
        model.ema = None

    for batch_idx in range(config.sampling.num_sample_batches):
        logger.info(f'Generating batch {batch_idx + 1}/{config.sampling.num_sample_batches}')

        logger.info('Using standard diffusion sampling')
        samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
        text_samples = model.tokenizer.batch_decode(samples)
        model.compute_generative_perplexity(text_samples)

    print('\n=== Generated text samples ===')
    for i, sample in enumerate(text_samples[:3]):
        print(f'Sample {i+1}:\n{sample}\n{"-"*50}')

    gen_ppl = model.gen_ppl_metric.compute()
    print(f'\nGenerative Perplexity: {gen_ppl:.4f}')
    logger.info(f'Generative Perplexity: {gen_ppl:.4f}')

    return text_samples


def _ppl_eval(config, logger, tokenizer):
    """Evaluate zero-shot perplexity on validation set"""
    logger.info('Starting perplexity evaluation...')

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)

    if config.eval.disable_ema:
        logger.info('Disabling EMA weights.')
        model.ema = None

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb)
        logger.info('Initialized WandB logger')

    callbacks = []
    if 'callbacks' in config:
        for callback_name, callback_config in config.callbacks.items():
            callback_instance = hydra.utils.instantiate(callback_config)
            callbacks.append(callback_instance)
            logger.info(f'Added callback: {callback_name}')

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)

    logger.info('Loading validation dataset...')
    _, valid_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=config.seed)

    logger.info('Computing perplexity...')
    trainer.validate(model, valid_ds)
    logger.info('Perplexity evaluation completed!')


def _train(config, logger, tokenizer):
    """Execute model training"""
    logger.info('Starting model training...')

    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb)
        logger.info('Initialized WandB logger')

    ckpt_path = None
    if (config.checkpointing.resume_from_ckpt and 
        config.checkpointing.resume_ckpt_path is not None and
        utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):

        ckpt_path = config.checkpointing.resume_ckpt_path
        logger.info(f'Resuming from checkpoint: {ckpt_path}')
    else:
        logger.info('Training from scratch')

    callbacks = []
    if 'callbacks' in config:
        for callback_name, callback_config in config.callbacks.items():
            callback_instance = hydra.utils.instantiate(callback_config)
            callbacks.append(callback_instance)
            logger.info(f'Added callback: {callback_name}')

    logger.info('Loading datasets...')
    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)

    _print_batch(train_ds, valid_ds, tokenizer)

    logger.info('Initializing diffusion model...')
    model = diffusion.Diffusion(config, tokenizer=valid_ds.tokenizer)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)

    logger.info('Starting training loop...')
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

    logger.info('Training completed!')


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point - execute task based on config mode"""
    L.seed_everything(config.seed)
    print(f'Random seed: {config.seed}')

    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    logger.info('=== MDLM (Masked Diffusion Language Model) ===')
    logger.info(f'Mode: {config.mode}')

    logger.info('Initializing tokenizer...')
    tokenizer = dataloader.get_tokenizer(config)
    logger.info(f'Vocab size: {tokenizer.vocab_size}')

    if config.mode == 'sample_eval':
        logger.info('Entering sample generation mode')
        text_samples = generate_samples(config, logger, tokenizer)
        logger.info(f'Generated {len(text_samples)} text samples')

    elif config.mode == 'ppl_eval':
        logger.info('Entering perplexity evaluation mode')
        _ppl_eval(config, logger, tokenizer)

    else:
        logger.info('Entering training mode')
        _train(config, logger, tokenizer)


if __name__ == '__main__':
    main()