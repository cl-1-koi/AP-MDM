import os
import logging
import random
import hashlib
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

class APMDMTokenizer:
    """Universal APMDM Tokenizer - backward compatible with legacy versions"""
    
    def __init__(self, vocab_cache_path=None, vocab_file=None):
        # Backward compatibility: support vocab_file parameter
        if vocab_file:
            vocab_cache_path = vocab_file
        self.vocab_cache_path = vocab_cache_path or '/workspace/projects/APMDM/data/vocab_cache.pkl'
        
        # Initialize mapping dictionaries
        self.stoi = {}  # string to int
        self.itos = {}  # int to string
        
        # Compatibility attributes
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        
        if self.vocab_cache_path and os.path.exists(self.vocab_cache_path):
            self._load_vocab()
        else:
            LOGGER.error(f"Vocabulary file does not exist or path is empty: {self.vocab_cache_path}")
            self._create_default_vocab()
    
    def _load_vocab(self):
        """Load vocabulary from pickle file"""
        try:
            with open(self.vocab_cache_path, 'rb') as f:
                vocab_data = pickle.load(f)
            
            if isinstance(vocab_data, dict):
                # Check if stoi and itos fields exist
                if 'stoi' in vocab_data and 'itos' in vocab_data:
                    self.stoi = vocab_data['stoi']
                    self.itos = vocab_data['itos']
                else:
                    # Assume id->token mapping, keep original mapping
                    self.itos = vocab_data
                    self.stoi = {token: token_id for token_id, token in vocab_data.items()}
            
            LOGGER.info(f"Loaded vocabulary from {self.vocab_cache_path}, size: {self.vocab_size}")
            
            # Ensure required special tokens exist without changing existing mappings
            self._ensure_special_tokens()
            
        except Exception as e:
            LOGGER.error(f"Failed to load vocabulary: {e}")
            self._create_default_vocab()
    
    def _create_default_vocab(self):
        """Create default vocabulary - only as fallback, should not typically be used"""
        # Digits 0-9
        for i in range(10):
            self.stoi[str(i)] = i
            self.itos[i] = str(i)
        
        # Dynamically add special tokens to available IDs
        next_id = 10  # Start searching for available ID from 10
        special_tokens = ['[MASK]', '[EOS]', '[PAD]', '[BOS]']
        
        for token in special_tokens:
            # Find next available ID
            while next_id in self.itos:
                next_id += 1
            self.stoi[token] = next_id
            self.itos[next_id] = token
            next_id += 1
        
        LOGGER.warning(f"Created default vocabulary, size: {self.vocab_size} (this should not typically happen)")
    
    def _ensure_special_tokens(self):
        """Ensure special tokens exist without changing existing mappings, and ensure vocab size is 34 to match checkpoint"""
        required_tokens = ['[MASK]', '[EOS]', '[PAD]', '[BOS]']
        missing_tokens = []
        
        # Check which tokens are missing
        for token in required_tokens:
            if token not in self.stoi:
                missing_tokens.append(token)
        
        # Only assign new IDs for missing tokens
        if missing_tokens:
            used_ids = set(self.itos.keys())
            next_id = 0
            
            for token in missing_tokens:
                # Find next available ID
                while next_id in used_ids:
                    next_id += 1
                
                self.stoi[token] = next_id
                self.itos[next_id] = token
                used_ids.add(next_id)
                LOGGER.info(f"Added missing special token: {token} -> {next_id}")
                next_id += 1
        
        # Ensure vocabulary size is 34 to match checkpoint
        target_vocab_size = 34
        current_size = len(self.itos)
        if current_size < target_vocab_size:
            used_ids = set(self.itos.keys())
            next_id = 0
            needed = target_vocab_size - current_size
            
            for i in range(needed):
                # Find next available ID
                while next_id in used_ids:
                    next_id += 1
                
                token = f'[RESERVED{i}]'
                self.stoi[token] = next_id
                self.itos[next_id] = token
                used_ids.add(next_id)
                LOGGER.info(f"Added reserved token to match checkpoint: {token} -> {next_id}")
                next_id += 1
        
        LOGGER.info(f"Special token mapping: MASK={self.mask_token_id}, EOS={self.eos_token_id}, PAD={self.pad_token_id}, BOS={self.bos_token_id}")
        LOGGER.info(f"Final vocabulary size: {self.vocab_size} (target: 34)")
    
    @property
    def vocab_size(self):
        # Return max token ID + 1, ensuring embedding layer can cover all possible tokens
        if self.itos:
            return max(self.itos.keys()) + 1
        return 0
    
    @property
    def mask_token_id(self):
        return self.stoi.get('[MASK]', -1)
    
    @property 
    def eos_token_id(self):
        return self.stoi.get('[EOS]', -1)
    
    @property
    def pad_token_id(self):
        return self.stoi.get('[PAD]', -1)
    
    @property
    def bos_token_id(self):
        return self.stoi.get('[BOS]', -1)
    
    def decode(self, token_ids):
        """Decode token id list to text"""
        tokens = [self.itos.get(id, str(id)) for id in token_ids]
        return ' '.join(tokens)
    
    def batch_decode(self, batch_token_ids, skip_special_tokens=True):
        """Batch decode token id lists to text list"""
        if hasattr(batch_token_ids, 'cpu'):
            batch_token_ids = batch_token_ids.cpu().numpy()
        
        results = []
        for token_ids in batch_token_ids:
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            
            # Filter special tokens (if needed)
            if skip_special_tokens:
                filtered_ids = []
                for token_id in token_ids:
                    if token_id not in [self.pad_token_id, self.bos_token_id]:
                        filtered_ids.append(token_id)
                    elif token_id == self.eos_token_id:
                        break  # Stop when encountering EOS
                token_ids = filtered_ids
            
            tokens = [self.itos.get(int(id), str(id)) for id in token_ids]
            results.append(' '.join(tokens))
        
        return results


class ChunkedStreamingDataset(Dataset):
    """
    Chunked Streaming Dataset - truly memory-friendly implementation
    - First run: Process in chunks and cache to separate directory
    - Subsequent runs: Load chunks on-demand, keeping only one chunk in memory
    """
    
    def __init__(self, data_path, mode='train', train_ratio=0.9, max_length=512, 
                 tokenizer=None, cache_dir=None, chunk_size=10000):
        self.data_path = data_path
        self.mode = mode
        self.train_ratio = train_ratio
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir or '/workspace/data'
        self.chunk_size = chunk_size
        
        # Create dataset-specific directory (including chunk_size info)
        dataset_name = os.path.splitext(os.path.splitext(os.path.basename(self.data_path))[0])[0]
        self.chunks_dir = os.path.join(self.cache_dir, f"{dataset_name}_chunks_{chunk_size}")
        self.meta_path = os.path.join(self.chunks_dir, f"meta_{mode}.pkl")
        
        # Currently loaded chunk
        self.current_chunk_idx = -1
        self.current_chunk_data = []
        
        # Initialize
        if os.path.exists(self.meta_path):
            self._load_metadata()
        else:
            self._create_chunks()
        
        LOGGER.info(f"✅ ChunkedStreamingDataset initialized: {self.total_samples:,} samples, {len(self.chunk_files)} chunks")
    
    def _load_metadata(self):
        """Load metadata"""
        LOGGER.info(f"📦 Loading metadata from cache: {self.meta_path}")
        with open(self.meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        self.chunk_files = meta['chunk_files']
        self.total_samples = meta['total_samples']
        self.chunk_samples = meta['chunk_samples']
    
    def _create_chunks(self):
        """Create chunk cache"""
        LOGGER.info(f"🔨 Creating chunk cache: chunk_size={self.chunk_size}")
        
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        # Read raw data at once
        LOGGER.info(f"📂 Reading raw file: {self.data_path}")
        if self.data_path.endswith('.gz'):
            import gzip
            with gzip.open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            all_samples = data['samples']
        elif isinstance(data, list):
            all_samples = data
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        del data  # Release original wrapper
        
        # Split by train_ratio
        total = len(all_samples)
        split_point = int(total * self.train_ratio)
        
        if self.mode == 'train':
            my_samples = all_samples[:split_point]
        else:
            my_samples = all_samples[split_point:]
        
        del all_samples  # Release all data
        
        LOGGER.info(f"Starting chunked processing for {len(my_samples):,} samples...")
        
        self.chunk_files = []
        self.chunk_samples = []
        
        # Process in chunks
        for chunk_idx in range(0, len(my_samples), self.chunk_size):
            chunk_end = min(chunk_idx + self.chunk_size, len(my_samples))
            chunk_raw = my_samples[chunk_idx:chunk_end]
            
            # Process current chunk
            processed_chunk = []
            for sample in chunk_raw:
                processed = self._process_item(sample)
                if processed:
                    processed_chunk.append(processed)
            
            # Save chunk
            # Train and validation share ``chunks_dir`` and are materialised
            # sequentially.  The released name ``chunk_N.pkl`` let validation
            # overwrite the first training chunks while both metadata files
            # continued to reference them.
            chunk_file = f"{self.mode}_chunk_{len(self.chunk_files)}.pkl"
            chunk_path = os.path.join(self.chunks_dir, chunk_file)
            with open(chunk_path, 'wb') as f:
                pickle.dump(processed_chunk, f)
            
            self.chunk_files.append(chunk_file)
            self.chunk_samples.append(len(processed_chunk))
            
            LOGGER.info(f"Processed chunk {len(self.chunk_files)}: {len(processed_chunk)} samples")
            
            # Immediately release
            del chunk_raw, processed_chunk
        
        del my_samples  # Release remaining data
        
        self.total_samples = sum(self.chunk_samples)
        
        # Save metadata
        meta = {
            'chunk_files': self.chunk_files,
            'total_samples': self.total_samples,
            'chunk_samples': self.chunk_samples
        }
        with open(self.meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        LOGGER.info(f"✅ Chunk caching complete: {len(self.chunk_files)} chunks, {self.total_samples:,} samples")
    
    def _process_item(self, item):
        """Process a single sample"""
        try:
            # Process APMDM format
            x_k = item['x_k'].tolist() if hasattr(item['x_k'], 'tolist') else list(item['x_k'])
            y_star = item['y_star'].tolist() if hasattr(item['y_star'], 'tolist') else list(item['y_star'])
            r_star = item['r_star'].tolist() if hasattr(item['r_star'], 'tolist') else list(item['r_star'])
            e_star = item['e_star'].tolist() if hasattr(item['e_star'], 'tolist') else list(item['e_star'])
            c_star = item['c_star'].tolist() if hasattr(item['c_star'], 'tolist') else list(item['c_star'])
            
            # Fix length inconsistency
            min_len = min(len(x_k), len(r_star), len(e_star), len(c_star))
            
            x_k = x_k[:min_len]
            r_star = r_star[:min_len]
            e_star = e_star[:min_len]
            c_star = c_star[:min_len]
            
            # y_star can be different length
            if abs(len(y_star) - min_len) > 10:
                y_star = y_star[:min_len + 3]
            
            # Check length limit
            if len(x_k) > self.max_length or len(y_star) > self.max_length:
                return None
            
            # Generate attention mask
            attention_mask = [1] * len(x_k)
            target_attention_mask = [1] * len(y_star)
            
            return {
                'input_ids': x_k,
                'target_ids': y_star,
                'attention_mask': attention_mask,
                'target_attention_mask': target_attention_mask,
                'remask_labels': r_star,
                'expansion_labels': e_star,
                'contraction_labels': c_star
            }
            
        except Exception as e:
            LOGGER.debug(f"Failed to process sample: {e}")
            return None
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Load samples on-demand - keep only current chunk in memory"""
        # Find corresponding chunk
        chunk_idx = 0
        cumulative_samples = 0
        
        for i, chunk_sample_count in enumerate(self.chunk_samples):
            if idx < cumulative_samples + chunk_sample_count:
                chunk_idx = i
                sample_idx_in_chunk = idx - cumulative_samples
                break
            cumulative_samples += chunk_sample_count
        else:
            raise IndexError(f"Sample index out of range: {idx}")
        
        # Load chunk on-demand
        if self.current_chunk_idx != chunk_idx:
            self._load_chunk(chunk_idx)
        
        # Ensure chunk data is loaded (protection for multiprocessing environment)
        if not self.current_chunk_data:
            self._load_chunk(chunk_idx)
        
        # Validate index range
        if sample_idx_in_chunk >= len(self.current_chunk_data):
            raise IndexError(f"Index {sample_idx_in_chunk} in Chunk {chunk_idx} out of range {len(self.current_chunk_data)}")
        
        return self.current_chunk_data[sample_idx_in_chunk]
    
    def _load_chunk(self, chunk_idx):
        """Load specific chunk to memory"""
        chunk_file = self.chunk_files[chunk_idx]
        chunk_path = os.path.join(self.chunks_dir, chunk_file)
        
        # Release previous chunk
        if hasattr(self, 'current_chunk_data') and self.current_chunk_data:
            del self.current_chunk_data
            import gc
            gc.collect()  # Force garbage collection
        
        # Load new chunk
        with open(chunk_path, 'rb') as f:
            self.current_chunk_data = pickle.load(f)
        
        self.current_chunk_idx = chunk_idx
    
    def iter_chunks_sequentially(self, batch_size, tokenizer):
        """Iterate in chunk order - training-specific, avoid random access across chunks"""
        import random
        
        for chunk_idx in range(len(self.chunk_files)):
            # Load current chunk
            self._load_chunk(chunk_idx)
            chunk_data = self.current_chunk_data.copy()
            
            # Shuffle within chunk
            random.shuffle(chunk_data)
            
            # Process current chunk in batches
            for batch_start in range(0, len(chunk_data), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_data))
                batch_samples = chunk_data[batch_start:batch_end]
                
                if batch_samples:
                    yield collate_fn(batch_samples, tokenizer)
            
            # Release after processing current chunk
            del chunk_data


class APMDMDataset(Dataset):
    """Universal APMDM Dataset Main Interface - Simplified Version"""
    
    def __init__(self, data_path, mode='train', streaming=False, max_length=512, 
                 train_ratio=0.9, vocab_cache_path=None, cache_dir=None, chunk_size=100000, **kwargs):
        self.data_path = data_path
        self.mode = mode
        self.streaming = streaming
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.cache_dir = cache_dir
        
        # Initialize tokenizer
        self.tokenizer = APMDMTokenizer(vocab_cache_path)
        
        # Use user-configured chunk_size
        self.chunk_size = chunk_size
        
        # Choose loading method based on streaming
        if streaming:
            LOGGER.info(f"🌊 Using streaming mode (chunked loading, chunk_size={chunk_size:,})")
            self.data = ChunkedStreamingDataset(
                data_path=self.data_path,
                mode=self.mode,
                train_ratio=self.train_ratio,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                chunk_size=self.chunk_size
            )
        else:
            LOGGER.info(f"📚 Using traditional mode (load all to memory)")
            # Traditional mode: directly load all data to memory
            self._load_all_data_traditional()
    
    def _load_all_data_traditional(self):
        """Traditional mode: directly load all data to memory"""
        # Directly load PKL file
        if self.data_path.endswith('.gz'):
            import gzip
            with gzip.open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
        
        if isinstance(data, dict) and 'samples' in data:
            all_samples = data['samples']
        elif isinstance(data, list):
            all_samples = data
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        del data  # Release original wrapper
        
        # Split by train_ratio
        total = len(all_samples)
        split_point = int(total * self.train_ratio)
        
        if self.mode == 'train':
            my_samples = all_samples[:split_point]
        else:
            my_samples = all_samples[split_point:]
        
        del all_samples  # Release all data
        
        LOGGER.info(f"Traditional mode processing {len(my_samples):,} samples...")
        
        # Process all samples
        self.data = []
        for sample in my_samples:
            processed = self._process_item_traditional(sample)
            if processed:
                self.data.append(processed)
        
        # Shuffle training set
        if self.mode == 'train':
            import random
            random.seed(42)
            random.shuffle(self.data)
        
        del my_samples  # Release original samples
        
        LOGGER.info(f"✅ Traditional mode loading complete: {len(self.data):,} samples")
    
    def _process_item_traditional(self, item):
        """Process a single sample (traditional mode, same as _process_item in ChunkedStreamingDataset)"""
        try:
            # Process APMDM format: x_k -> input_ids, y_star -> target_ids
            x_k = item['x_k'].tolist() if hasattr(item['x_k'], 'tolist') else list(item['x_k'])
            y_star = item['y_star'].tolist() if hasattr(item['y_star'], 'tolist') else list(item['y_star'])
            r_star = item['r_star'].tolist() if hasattr(item['r_star'], 'tolist') else list(item['r_star'])
            e_star = item['e_star'].tolist() if hasattr(item['e_star'], 'tolist') else list(item['e_star'])
            c_star = item['c_star'].tolist() if hasattr(item['c_star'], 'tolist') else list(item['c_star'])
            
            # Fix length inconsistency
            min_len = min(len(x_k), len(r_star), len(e_star), len(c_star))
            
            x_k = x_k[:min_len]
            r_star = r_star[:min_len]
            e_star = e_star[:min_len]
            c_star = c_star[:min_len]
            
            # y_star can be different length
            if abs(len(y_star) - min_len) > 10:
                y_star = y_star[:min_len + 3]
            
            # Check length limit
            if len(x_k) > self.max_length or len(y_star) > self.max_length:
                return None
            
            # Generate attention mask
            attention_mask = [1] * len(x_k)
            target_attention_mask = [1] * len(y_star)
            
            return {
                'input_ids': x_k,
                'target_ids': y_star,
                'attention_mask': attention_mask,
                'target_attention_mask': target_attention_mask,
                'remask_labels': r_star,
                'expansion_labels': e_star,
                'contraction_labels': c_star
            }
            
        except Exception as e:
            LOGGER.debug(f"Failed to process sample: {e}")
            return None

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        return 0
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_length=512):
    """
    Optimized batch processing function
    Use numpy intermediate storage to reduce torch.tensor calls
    """
    if not batch:
        return {}
    
    # Get batch size
    batch_size = len(batch)
    
    # Find maximum length
    max_input_len = max(len(item['input_ids']) for item in batch)
    max_target_len = max(len(item['target_ids']) for item in batch)
    
    # Limit maximum length
    max_input_len = min(max_input_len, max_length)
    max_target_len = min(max_target_len, max_length)
    
    # Create batch arrays using numpy
    pad_id = tokenizer.pad_token_id
    input_ids_np = np.full((batch_size, max_input_len), pad_id, dtype=np.int64)
    target_ids_np = np.full((batch_size, max_target_len), pad_id, dtype=np.int64)
    attention_mask_np = np.zeros((batch_size, max_input_len), dtype=np.int64)
    target_attention_mask_np = np.zeros((batch_size, max_target_len), dtype=np.int64)
    
    # APMDM-specific labels
    remask_labels_np = np.zeros((batch_size, max_input_len), dtype=np.int64)
    expansion_labels_np = np.zeros((batch_size, max_input_len), dtype=np.int64)
    contraction_labels_np = np.zeros((batch_size, max_input_len), dtype=np.int64)
    
    # Fill data
    for i, item in enumerate(batch):
        input_seq = item['input_ids'][:max_input_len]
        target_seq = item['target_ids'][:max_target_len]
        
        input_len = len(input_seq)
        target_len = len(target_seq)
        
        input_ids_np[i, :input_len] = input_seq
        target_ids_np[i, :target_len] = target_seq
        attention_mask_np[i, :input_len] = item['attention_mask'][:input_len]
        target_attention_mask_np[i, :target_len] = item['target_attention_mask'][:target_len]
        
        # Fill labels
        remask_labels_np[i, :input_len] = item['remask_labels'][:input_len]
        expansion_labels_np[i, :input_len] = item['expansion_labels'][:input_len]
        contraction_labels_np[i, :input_len] = item['contraction_labels'][:input_len]
    
    # Convert to torch tensor at once
    return {
        'input_ids': torch.from_numpy(input_ids_np),
        'target_ids': torch.from_numpy(target_ids_np), 
        'attention_mask': torch.from_numpy(attention_mask_np),
        'target_attention_mask': torch.from_numpy(target_attention_mask_np),
        'remask_labels': torch.from_numpy(remask_labels_np),
        'expansion_labels': torch.from_numpy(expansion_labels_np),
        'contraction_labels': torch.from_numpy(contraction_labels_np),
    }


# Backward compatibility
def get_apmdm_dataset(*args, **kwargs):
    """Universal APMDM dataset factory function"""
    return APMDMDataset(*args, **kwargs)

# Legacy function name for backward compatibility
def get_sudoku_dataset(*args, **kwargs):
    """Factory function for backward compatibility"""  
    return APMDMDataset(*args, **kwargs)
