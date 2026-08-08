import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


# Detokenization functions
def wt_detokenizer(string):
    """WikiText detokenizer"""
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    """Penn Treebank detokenizer"""
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
            x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    """One Billion Word detokenizer"""
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    """LAMBADA detokenizer"""
    text = text.replace(""", '"')
    text = text.replace(""", '"')
    return '\n'+text.strip()


def scientific_papers_detokenizer(x):
    """Scientific papers detokenizer"""
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


# Text8 character-level tokenizer
class Text8Tokenizer(transformers.PreTrainedTokenizer):
    """Text8 character-level tokenizer (26 lowercase letters + space, vocab size 35)"""
    
    def __init__(
        self,
        bos_token='[BOS]',
        eos_token='[EOS]',
        sep_token='[SEP]',
        cls_token='[CLS]',
        pad_token='[PAD]',
        mask_token='[MASK]',
        unk_token='[UNK]',
        **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz ')
        
        self._vocab_str_to_int = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[PAD]': 5,
            '[RESERVED]': 6,
            '[UNK]': 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)}
        }
        
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()
        }
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


# Dataset loading functions

def get_lambada_test_dataset():
    """Get LAMBADA test dataset for long-range dependency modeling"""
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset

def get_text8_dataset(cache_dir, max_seq_length=256,
                    drop_last=True, crop_train=False):
    """Get Text8 dataset (character-level: a-z + space, 100MB text)"""
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not crop_train:
        cache_dir = f'{cache_dir}/text8'
    else:
        cache_dir = f'{cache_dir}/text8-crop-train'
    split_names = ['train', 'validation', 'test']
    if not all([
        utils.fsspec_exists(os.path.join(cache_dir, split))
        for split in split_names
    ]):
        raw_cache_dir = os.path.join(cache_dir, 'raw_data')
        if not all([
            utils.fsspec_exists(
                os.path.join(raw_cache_dir, f'text8.{split}.txt'))
            for split in split_names
        ]):
            if not utils.fsspec_exists(
                os.path.join(raw_cache_dir, 'text8.zip')):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info('Downloading text8 dataset from {}'.format(url))
                with (urllib.request.urlopen(url) as in_stream,
                    open(os.path.join(raw_cache_dir, 'text8.zip'),
                    'wb') as out_file):
                    shutil.copyfileobj(in_stream, out_file)

            with fsspec.open(
                os.path.join(raw_cache_dir, 'text8.zip'),
                'rb') as f:
                rawdata = zipfile.ZipFile(f).read(
                    'text8').decode('utf-8')

            splits = {
                'train': rawdata[:90000000],
                'validation': rawdata[90000000: 95000000],
                'test': rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir,
                        f'text8.{split}.txt')
                with fsspec.open(_path, 'w') as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir,
                        f'text8.{split}.txt')
                with fsspec.open(_path, 'r') as f:
                    splits[split] = f.read()

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == 'train' and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def _group_texts(examples, block_size, bos, eos):
    """Concatenate and group text samples into fixed-length blocks"""
    concatenated_examples = list(itertools.chain(* examples['input_ids']))
    total_length = len(concatenated_examples)

    new_block_size = block_size - 2
    total_length = (total_length // new_block_size) * new_block_size

    result = {}
    _values = []
    _attn_masks = []

    for i in range(0, total_length, new_block_size):
        _values.append(
            [bos]
            + concatenated_examples[i : i + new_block_size]
            + [eos])

        _attn_masks.append(torch.ones(block_size))

    result['input_ids'] = _values
    result['attention_mask'] = _attn_masks
    return result


def get_dataset(
        dataset_name, tokenizer, wrap, mode, cache_dir,
        block_size=1024, num_proc=len(os.sched_getaffinity(0)), streaming=False, 
        config=None):
    """Get and preprocess specified dataset"""
    if wrap:
        filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped.dat'
    else:
        filename = f'{dataset_name}_{mode}_bs{block_size}_unwrapped.dat'
    _path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(_path):
        LOGGER.info(f'Loading from cache: {_path}')
        return datasets.load_from_disk(_path).with_format('torch')
    LOGGER.info(f'Generating new data at: {_path}')

    crop_train = dataset_name == 'text8-crop'
    if mode == 'train' and crop_train:
        block_size *= 2
    if dataset_name == 'wikitext103':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-103-raw-v1',
            cache_dir=cache_dir)
    elif dataset_name == 'wikitext2':
        dataset = datasets.load_dataset(
            'wikitext',
            name='wikitext-2-raw-v1',
            cache_dir=cache_dir)
    elif dataset_name == 'ptb':
        dataset = datasets.load_dataset(
            'ptb_text_only', cache_dir=cache_dir)
    elif dataset_name == 'lambada':
        dataset = get_lambada_test_dataset()
    elif dataset_name == 'text8':
        assert wrap  
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size)
    elif dataset_name == 'text8-crop':
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True)
    elif dataset_name == 'openwebtext-train':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[:-100000]', 
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True)
    elif dataset_name == 'openwebtext-valid':
        dataset = datasets.load_dataset(
            'openwebtext',
            split='train[-100000:]',     
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True)
    elif dataset_name == 'scientific_papers_arxiv':
        dataset = datasets.load_dataset(
            'scientific_papers', 'arxiv',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'scientific_papers_pubmed':
        dataset = datasets.load_dataset(
            'scientific_papers', 'pubmed',
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming)
    elif dataset_name == 'ag_news':
        dataset = datasets.load_dataset(
            'ag_news',
            cache_dir=cache_dir,
            streaming=streaming)

    elif dataset_name in ['sudoku_apmdm', 'apmdm']:
        # APMDM Dataset - Custom dataloader for APMDM training
        LOGGER.info(f'Loading APMDM dataset - {mode} mode')
        import apmdm_dataloader

        # Read dataset path from config
        if config and hasattr(config, 'data') and hasattr(config.data, 'dataset_path'):
            data_path = config.data.dataset_path
            LOGGER.info(f'Using configured dataset path: {data_path}')
        else:
            data_path = '/workspace/projects/APMDM/data/apmdm_training_dataset_10k.pkl.gz'
            LOGGER.warning(f'Dataset path not found in config, using default: {data_path}')
        
        # Train/validation split ratio
        if config and hasattr(config, 'data') and hasattr(config.data, 'train_ratio'):
            train_ratio = config.data.train_ratio
        else:
            train_ratio = 0.99
        
        # Vocab file path - read from config or use default
        if config and hasattr(config, 'data') and hasattr(config.data, 'vocab_cache_path'):
            vocab_file = config.data.vocab_cache_path
        else:
            vocab_file = '/workspace/projects/APMDM/data/vocab_cache.pkl'
        
        # Create APMDM tokenizer and dataset
        tokenizer_obj = apmdm_dataloader.APMDMTokenizer(vocab_file)
        
        # Read streaming-related settings from config
        streaming = getattr(config.data, 'streaming', False) if config and hasattr(config, 'data') else False
        cache_dir = getattr(config.data, 'cache_dir', None) if config and hasattr(config, 'data') else None
        chunk_size = getattr(config.data, 'chunk_size', 100000) if config and hasattr(config, 'data') else 100000
        
        dataset = apmdm_dataloader.APMDMDataset(
            data_path=data_path,
            tokenizer=tokenizer_obj,
            vocab_cache_path=vocab_file,
            mode=mode,
            train_ratio=train_ratio,
            streaming=streaming,
            cache_dir=cache_dir,
            chunk_size=chunk_size
        )
        
        LOGGER.info(f"🚀 Streaming mode: {streaming}, Cache: {cache_dir or 'default'}, Chunk size: {chunk_size}")
        LOGGER.info(f"💾 Using internal streaming cache, no duplicate storage")
        
        return dataset
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True)

    if dataset_name in ['lambada', 'openwebtext-train',
                    'openwebtext-valid']:
        data = dataset
    else:
        data = dataset[mode]

    if dataset_name.startswith('wikitext'):
        detokenizer = wt_detokenizer
    elif dataset_name == 'ptb':
        detokenizer = ptb_detokenizer
    elif dataset_name == 'lm1b':
        detokenizer = lm1b_detokenizer
    elif dataset_name == 'lambada':
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith('scientific_papers'):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        """Preprocess and tokenize text samples"""
        if dataset_name == 'ptb':
            text = example['sentence']
        elif 'scientific_papers' in dataset_name:
            text = example['article']
        else:
            text = example['text']

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        if wrap:
            tokens = tokenizer(text,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False)
            
            tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
        else:
            tokens = tokenizer(text,
                        max_length=block_size,
                        padding='max_length',
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_token_type_ids=True)
        
        return tokens

    if streaming:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            desc='Tokenizing')
    else:
        tokenized_dataset = data.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc='Tokenizing')
    if dataset_name == 'ptb':
        tokenized_dataset = tokenized_dataset.remove_columns(
            'sentence')
    elif 'scientific_papers' in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns([
            'article', 'abstract', 'section_names'])
    elif dataset_name == 'ag_news':
        tokenized_dataset = tokenized_dataset.remove_columns(
            ['text', 'label'])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns(
            'text')

    if not wrap:
        tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format('torch')

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS)
    if streaming:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            desc='Grouping')
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc='Grouping')
        chunked_dataset.save_to_disk(_path)
    chunked_dataset = chunked_dataset.with_format('torch')
    return chunked_dataset


def get_tokenizer(config):
    """Get tokenizer based on config"""
    if config.data.tokenizer_name_or_path == 'text8':
        tokenizer = Text8Tokenizer()

    elif config.data.tokenizer_name_or_path in ['sudoku_apmdm', 'apmdm']:
        # APMDM Tokenizer - Custom vocabulary for APMDM training
        import apmdm_dataloader
        
        # Read vocab path from config to avoid hardcoding
        if hasattr(config.data, 'vocab_cache_path') and config.data.vocab_cache_path:
            vocab_cache_path = config.data.vocab_cache_path
        else:
            vocab_cache_path = '/workspace/projects/APMDM/data/vocab_cache.pkl'
        
        tokenizer = apmdm_dataloader.APMDMTokenizer(
            vocab_file=vocab_cache_path if os.path.exists(vocab_cache_path) else None
        )
    elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.\
            from_pretrained('bert-base-uncased')
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path)

    if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
            or isinstance(tokenizer, transformers.GPT2Tokenizer)):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id))

    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                'Tokenizer must have a bos_token or '
                f'cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                'Tokenizer must have a eos_token '
                f'or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
    """Build train and validation dataloaders"""
    num_gpus = torch.cuda.device_count()

    assert (config.loader.global_batch_size
                    == (config.loader.batch_size
                        * config.trainer.num_nodes
                        * num_gpus
                        * config.trainer.accumulate_grad_batches))

    if config.loader.global_batch_size % (
        num_gpus * config.trainer.accumulate_grad_batches) != 0:
        raise ValueError(
            f'Train batch size {config.training.batch_size} '
            f'not divisible by {num_gpus} GPUs and '
            f'{config.trainer.accumulate_grad_batches} gradient accumulation steps.')

    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f'Eval batch size {config.eval.batch_size} '
            f'not divisible by {num_gpus} GPUs.')

    if skip_train:
        train_set = None
    else:
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode='train',
            wrap=config.data.wrap,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            config=config)

    if config.data.valid in ['text8', 'lm1b', 'ag_news']:
        validation_split = 'test'
    else:
        validation_split = 'validation'

    if skip_valid:
        valid_set = None
    else:
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=False,
            config=config)

    if skip_train:
        train_loader = None
    else:
        # APMDM Streaming Support - Sequential chunk access for efficient loading
        import apmdm_dataloader as _apmdm
        if (config.data.train in ['sudoku_apmdm', 'apmdm'] and 
            config.data.streaming and 
            hasattr(train_set, 'data') and 
            hasattr(train_set.data, 'iter_chunks_sequentially')):
            
            # ChunkedStreamingDataset uses sequential chunk access to avoid multiprocessing issues
            LOGGER.info("🌊 Detected ChunkedStreamingDataset, using sequential chunk access mode")
            
            # Custom DataLoader for sequential chunk iteration
            class SequentialChunkDataLoader:
                def __init__(self, dataset, batch_size, tokenizer):
                    self.dataset = dataset
                    self.batch_size = batch_size
                    self.tokenizer = tokenizer
                    self.sampler = None  # Lightning compatibility
                
                def __len__(self):
                    return (len(self.dataset) + self.batch_size - 1) // self.batch_size
                
                def __iter__(self):
                    return self.dataset.iter_chunks_sequentially(self.batch_size, self.tokenizer)
            
            train_loader = SequentialChunkDataLoader(train_set.data, config.loader.batch_size, tokenizer)
            
        else:
            # Traditional DataLoader construction
            # Dynamic padding support: detect APMDM dataset and use appropriate collate_fn
            train_collate_fn = None
            num_workers = config.loader.num_workers
            
            try:
                # Check for APMDM dataset
                if config.data.train in ['sudoku_apmdm', 'apmdm']:
                    import apmdm_dataloader as _apmdm
                    train_collate_fn = lambda batch: _apmdm.collate_fn(batch, tokenizer)
                    
                    # Streaming mode supports multi-worker parallel loading for better performance
                    if config.data.streaming:
                        LOGGER.info(f"🚀 Streaming mode with multi-worker: num_workers={num_workers}")
                else:
                    import apmdm_dataloader as _apmdm
                    if isinstance(train_set, _apmdm.APMDMDataset):
                        train_collate_fn = lambda batch: _apmdm.collate_fn(batch, train_set.tokenizer)
            except Exception:
                try:
                    import apmdm_dataloader as _apmdm
                    if isinstance(train_set, _apmdm.APMDMDataset) and getattr(train_set, 'dynamic_padding', False):
                        train_collate_fn = train_set.dynamic_collate_fn
                except Exception:
                    train_collate_fn = None
            
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.loader.batch_size,
                num_workers=num_workers,
                pin_memory=config.loader.pin_memory,
                shuffle=not config.data.streaming,
                persistent_workers=num_workers > 0,
                collate_fn=train_collate_fn)

        train_loader.tokenizer = tokenizer

    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)

        # Dynamic padding support for validation set
        valid_collate_fn = None
        try:
            if config.data.valid in ['sudoku_apmdm', 'apmdm']:
                import apmdm_dataloader as _apmdm
                valid_collate_fn = lambda batch: _apmdm.collate_fn(batch, tokenizer)
            else:
                import apmdm_dataloader as _apmdm
                if isinstance(valid_set, _apmdm.APMDMDataset):
                    valid_collate_fn = lambda batch: _apmdm.collate_fn(batch, valid_set.tokenizer)
        except Exception:
            try:
                import apmdm_dataloader as _apmdm
                if isinstance(valid_set, _apmdm.APMDMDataset) and getattr(valid_set, 'dynamic_padding', False):
                    valid_collate_fn = valid_set.dynamic_collate_fn
            except Exception:
                valid_collate_fn = None
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
            collate_fn=valid_collate_fn)

        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# Fault-tolerant samplers for training interruption recovery
class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):
    """Random sampler with fault tolerance for training interruption recovery"""

    def __init__(self, *args, generator=None, **kwargs):
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        super().__init__(*args, generator=generator, **kwargs)
        self.epoch = 0
        self.start_index = 0

    def state_dict(self):
        return {'epoch': self.epoch, 'start_index': self.start_index}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.start_index = state_dict['start_index']

    def __iter__(self) -> typing.Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = self.generator

        generator.manual_seed(generator.initial_seed() + self.epoch)

        for i in torch.randperm(self.num_samples, generator=generator).tolist()[self.start_index:]:
            yield i

        self.start_index = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    """Distributed sampler with fault tolerance for multi-GPU training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def state_dict(self):
        return {'epoch': self.epoch, 'start_index': self.start_index}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.start_index = state_dict['start_index']

    def __iter__(self):
        indices = list(super().__iter__())

        for index in indices[self.start_index:]:
            yield index

        self.start_index = 0
