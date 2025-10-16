#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📚 Parity Problem Data Loader

Provide convenient interface to load and use training and test datasets for Parity problem
"""

import pickle
import gzip
import json
from typing import List, Dict, Any, Tuple, Optional
from parity_generator import ParityTokens

class ParityDataLoader:
    """Parity problem data loader"""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.tokens = ParityTokens()
        self._training_samples = None
        self._test_samples = None
        self._vocab_cache = None
    
    def load_training_dataset(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load training dataset
        
        Args:
            file_path: Training data file path, defaults to parity_train.pkl.gz
            
        Returns:
            Training sample list
        """
        if self._training_samples is None:
            if file_path is None:
                file_path = f"{self.data_dir}/parity_train.pkl.gz"
            
            with gzip.open(file_path, 'rb') as f:
                self._training_samples = pickle.load(f)
                
        return self._training_samples
    
    def load_test_dataset(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load test dataset
        
        Args:
            file_path: Test data file path, defaults to parity_test.pkl.gz
            
        Returns:
            Test sample list
        """
        if self._test_samples is None:
            if file_path is None:
                file_path = f"{self.data_dir}/parity_test.pkl.gz"
            
            with gzip.open(file_path, 'rb') as f:
                self._test_samples = pickle.load(f)
                
        return self._test_samples
    
    def load_vocabulary(self, file_path: Optional[str] = None) -> Dict[int, str]:
        """
        Load vocabulary
        
        Args:
            file_path: Vocabulary file path, defaults to parity_vocab_cache.pkl
            
        Returns:
            Vocabulary dictionary {token_id: token_string}
        """
        if self._vocab_cache is None:
            if file_path is None:
                file_path = f"{self.data_dir}/parity_vocab_cache.pkl"
            
            with open(file_path, 'rb') as f:
                self._vocab_cache = pickle.load(f)
                
        return self._vocab_cache
    
    def get_sample_by_id(self, sample_id: int, dataset: str = 'train') -> Optional[Dict[str, Any]]:
        """
        Get specific sample by ID
        
        Args:
            sample_id: Sample ID
            dataset: Dataset type ('train' or 'test')
            
        Returns:
            Sample dictionary or None
        """
        if dataset == 'train':
            samples = self.load_training_dataset()
            for sample in samples:
                if sample['solver_metadata']['sample_id'] == sample_id:
                    return sample
        elif dataset == 'test':
            samples = self.load_test_dataset()
            if 0 <= sample_id < len(samples):
                return samples[sample_id]
        
        return None
    
    def get_samples_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """
        Get samples by operation type
        
        Args:
            operation: Operation name
            
        Returns:
            Matching sample list
        """
        samples = self.load_training_dataset()
        return [s for s in samples if s['solver_metadata']['operation'] == operation]
    
    def convert_tokens_to_strings(self, tokens: List[int]) -> List[str]:
        """
        Convert token ID list to string list
        
        Args:
            tokens: Token ID list
            
        Returns:
            Token string list
        """
        return [self.tokens.to_string(token) for token in tokens]
    
    def convert_strings_to_tokens(self, token_strings: List[str]) -> List[int]:
        """
        Convert string list to token ID list
        
        Args:
            token_strings: Token string list
            
        Returns:
            Token ID list
        """
        return [self.tokens.from_string(token_str) for token_str in token_strings]
    
    def print_sample_details(self, sample: Dict[str, Any], show_debug: bool = True):
        """
        Print detailed sample information
        
        Args:
            sample: Sample dictionary
            show_debug: Whether to show debug info
        """
        if 'solver_metadata' in sample:
            # Training sample
            meta = sample['solver_metadata']
            print(f"📋 Training Sample {meta['sample_id']}: {meta['operation']}")
            print(f"   x_k:       {self.convert_tokens_to_strings(sample['x_k'])}")
            print(f"   x_k_plus_1: {self.convert_tokens_to_strings(sample['x_k_plus_1'])}")
            print(f"   y_star:    {self.convert_tokens_to_strings(sample['y_star'])}")
            print(f"   r_star:    {sample['r_star']}")
            print(f"   e_star:    {sample['e_star']}")
            print(f"   c_star:    {sample['c_star']}")
        else:
            # Test sample
            print(f"🧪 Test Sample {sample['sample_id']}")
            print(f"   Sequence length:  {sample['length']}")
            print(f"   Number of 1s:   {sample['ones_count']}")
            print(f"   Parity:    {'Odd' if sample['parity'] == 1 else 'Even'}")
            print(f"   Expected result:  {self.convert_tokens_to_strings(sample['expected_final'])}")
            
            if show_debug and 'debug_info' in sample:
                seq_str = sample['debug_info']['sequence_str']
                print(f"   Sequence preview:  {seq_str[:10]}...{seq_str[-10:] if len(seq_str) > 20 else seq_str[10:]}")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Statistics dictionary
        """
        train_samples = self.load_training_dataset()
        test_samples = self.load_test_dataset()
        vocab = self.load_vocabulary()
        
        # Training set statistics
        train_operations = {}
        for sample in train_samples:
            op = sample['solver_metadata']['operation']
            train_operations[op] = train_operations.get(op, 0) + 1
        
        # Test set statistics
        test_lengths = [s['length'] for s in test_samples]
        test_parities = [s['parity'] for s in test_samples]
        
        return {
            'training': {
                'total_samples': len(train_samples),
                'operations': train_operations,
                'unique_operations': len(train_operations)
            },
            'test': {
                'total_samples': len(test_samples),
                'min_length': min(test_lengths),
                'max_length': max(test_lengths),
                'avg_length': sum(test_lengths) / len(test_lengths),
                'even_parity_samples': sum(1 for p in test_parities if p == 0),
                'odd_parity_samples': sum(1 for p in test_parities if p == 1)
            },
            'vocabulary': {
                'vocab_size': len(vocab),
                'tokens': list(vocab.values())
            }
        }
    
    def create_batch(self, samples: List[Dict[str, Any]], batch_size: int = 8) -> List[List[Dict[str, Any]]]:
        """
        Batch samples
        
        Args:
            samples: Sample list
            batch_size: Batch size
            
        Returns:
            Batch list
        """
        batches = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def get_apmdm_training_format(self, sample: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Get APMDM standard training format
        
        Args:
            sample: Original sample
            
        Returns:
            APMDM format sample
        """
        return {
            'x_k': sample['x_k'],
            'y_star': sample['y_star'],
            'r_star': sample['r_star'],
            'e_star': sample['e_star'],
            'c_star': sample['c_star'],
            'x_k_plus_1': sample['x_k_plus_1']
        }


def main():
    """Demonstrate data loader usage"""
    print("📚 Parity Data Loader Demo")
    print("=" * 40)
    
    # Create loader
    loader = ParityDataLoader()
    
    # Load datasets
    print("📂 Loading datasets...")
    train_samples = loader.load_training_dataset()
    test_samples = loader.load_test_dataset()
    vocab = loader.load_vocabulary()
    
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Test samples: {len(test_samples)}")
    print(f"   Vocabulary size: {len(vocab)} tokens")
    
    # Show statistics
    print(f"\n📊 Dataset Statistics:")
    stats = loader.get_dataset_statistics()
    
    print(f"   Training set:")
    for op, count in stats['training']['operations'].items():
        print(f"     {op}: {count} samples")
    
    print(f"   Test set:")
    print(f"     Average length: {stats['test']['avg_length']:.1f}")
    print(f"     Length range: {stats['test']['min_length']} - {stats['test']['max_length']}")
    print(f"     Even parity: {stats['test']['even_parity_samples']}")
    print(f"     Odd parity: {stats['test']['odd_parity_samples']}")
    
    print(f"   Vocabulary: {stats['vocabulary']['tokens']}")
    
    # Show some training samples
    print(f"\n🎓 Training Sample Examples:")
    for i in range(min(3, len(train_samples))):
        sample = loader.get_sample_by_id(i, 'train')
        loader.print_sample_details(sample)
        print()
    
    # Show one test sample
    print(f"🧪 Test Sample Example:")
    test_sample = loader.get_sample_by_id(0, 'test')
    loader.print_sample_details(test_sample, show_debug=False)
    
    # Demo APMDM format conversion
    print(f"\n🔄 APMDM Format Conversion Example:")
    sample = loader.get_sample_by_id(0, 'train')
    apmdm_format = loader.get_apmdm_training_format(sample)
    print(f"   APMDM format sample contains fields: {list(apmdm_format.keys())}")


if __name__ == "__main__":
    main()
