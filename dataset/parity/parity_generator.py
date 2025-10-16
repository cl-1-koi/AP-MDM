#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Parity Problem APMDM Training Data Generator

Problem Description:
- Solve parity problem by continuously eliminating pairs of same digits
- Elimination rules: 00 -> empty, 11 -> empty, 0X -> MASK X, 1X -> 1 MASK (X is any digit)
- Final goal: Simplify long sequence to only BOS (even number of 1s) or BOS 1 (odd number of 1s)

Token Vocabulary:
- BOS: Beginning of sequence
- EOS: End of sequence  
- MASK: Placeholder
- 0: Digit 0
- 1: Digit 1
"""

import random
import pickle
import gzip
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

class ParityTokens:
    """Token definitions for Parity problem"""
    BOS = 0
    EOS = 1  
    MASK = 2
    ZERO = 3
    ONE = 4
    
    @classmethod
    def to_string(cls, token_id: int) -> str:
        """Convert token ID to string"""
        mapping = {
            cls.BOS: "[BOS]",
            cls.EOS: "[EOS]", 
            cls.MASK: "[MASK]",
            cls.ZERO: "0",
            cls.ONE: "1"
        }
        return mapping.get(token_id, f"UNK_{token_id}")
    
    @classmethod
    def from_string(cls, token_str: str) -> int:
        """Convert string to token ID"""
        mapping = {
            "[BOS]": cls.BOS,
            "[EOS]": cls.EOS,
            "[MASK]": cls.MASK, 
            "0": cls.ZERO,
            "1": cls.ONE
        }
        return mapping.get(token_str, -1)
    
    @classmethod
    def get_vocab_size(cls) -> int:
        """Get vocabulary size"""
        return 5


class ParityDataGenerator:
    """APMDM training data generator for Parity problem"""
    
    def __init__(self):
        self.tokens = ParityTokens()
    
    def generate_training_samples(self) -> List[Dict[str, Any]]:
        """
        Generate fixed 7 training samples
        
        Sample 0: [BOS] 0 1 -> [BOS] [MASK] 1        (R operation: 0->MASK)
        Sample 1: [BOS] [MASK] 1 -> [BOS] 1          (C operation: delete MASK)  
        Sample 2: [BOS] 1 0 -> [BOS] 1 [MASK]        (R operation: 0->MASK)
        Sample 3: [BOS] 1 [MASK] -> [BOS] 1          (C operation: delete MASK)
        Sample 4: [BOS] 0 0 -> [BOS] [MASK] [MASK]   (R operation: two 0s->MASK)
        Sample 5: [BOS] 1 1 -> [BOS] [MASK] [MASK]   (R operation: two 1s->MASK) 
        Sample 6: [BOS] [MASK] [MASK] -> [BOS]       (C operation: delete two MASKs)
        """
        
        samples = []
        
        # Sample 0: [BOS] 0 1 -> [BOS] [MASK] 1
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ZERO, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.ONE],
            operation="eliminate_single_0",
            sample_id=0
        ))
        
        # Sample 1: [BOS] [MASK] 1 -> [BOS] 1  
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.MASK, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE],
            operation="delete_middle_mask",
            sample_id=1
        ))
        
        # Sample 2: [BOS] 1 0 -> [BOS] 1 [MASK]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.ZERO],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE, self.tokens.MASK],
            operation="eliminate_tail_0",
            sample_id=2
        ))
        
        # Sample 3: [BOS] 1 [MASK] -> [BOS] 1
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.MASK],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE],
            operation="delete_tail_mask",
            sample_id=3
        ))
        
        # Sample 4: [BOS] 0 0 -> [BOS] [MASK] [MASK]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ZERO, self.tokens.ZERO],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            operation="eliminate_pair_0",
            sample_id=4
        ))
        
        # Sample 5: [BOS] 1 1 -> [BOS] [MASK] [MASK] 
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            operation="eliminate_pair_1",
            sample_id=5
        ))
        
        # Sample 6: [BOS] [MASK] [MASK] -> [BOS]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            x_k_plus_1=[self.tokens.BOS],
            operation="delete_pair_mask",
            sample_id=6
        ))
        
        return samples
    
    @classmethod
    def get_vocab_size(cls) -> int:
        """Get vocabulary size"""
        return 5


class ParityDataGenerator:
    """APMDM training data generator for Parity problem"""
    
    def __init__(self):
        self.tokens = ParityTokens()
    
    def generate_training_samples(self) -> List[Dict[str, Any]]:
        """
        Generate fixed 7 training samples
        
        Sample 0: [BOS] 0 1 -> [BOS] [MASK] 1        (R operation: 0->MASK)
        Sample 1: [BOS] [MASK] 1 -> [BOS] 1          (C operation: delete MASK)  
        Sample 2: [BOS] 1 0 -> [BOS] 1 [MASK]        (R operation: 0->MASK)
        Sample 3: [BOS] 1 [MASK] -> [BOS] 1          (C operation: delete MASK)
        Sample 4: [BOS] 0 0 -> [BOS] [MASK] [MASK]   (R operation: two 0s->MASK)
        Sample 5: [BOS] 1 1 -> [BOS] [MASK] [MASK]   (R operation: two 1s->MASK) 
        Sample 6: [BOS] [MASK] [MASK] -> [BOS]       (C operation: delete two MASKs)
        """
        
        samples = []
        
        # Sample 0: [BOS] 0 1 -> [BOS] [MASK] 1
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ZERO, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.ONE],
            operation="eliminate_single_0",
            sample_id=0
        ))
        
        # Sample 1: [BOS] [MASK] 1 -> [BOS] 1  
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.MASK, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE],
            operation="delete_middle_mask",
            sample_id=1
        ))
        
        # Sample 2: [BOS] 1 0 -> [BOS] 1 [MASK]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.ZERO],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE, self.tokens.MASK],
            operation="eliminate_tail_0",
            sample_id=2
        ))
        
        # Sample 3: [BOS] 1 [MASK] -> [BOS] 1
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.MASK],
            x_k_plus_1=[self.tokens.BOS, self.tokens.ONE],
            operation="delete_tail_mask",
            sample_id=3
        ))
        
        # Sample 4: [BOS] 0 0 -> [BOS] [MASK] [MASK]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ZERO, self.tokens.ZERO],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            operation="eliminate_pair_0",
            sample_id=4
        ))
        
        # Sample 5: [BOS] 1 1 -> [BOS] [MASK] [MASK] 
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.ONE, self.tokens.ONE],
            x_k_plus_1=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            operation="eliminate_pair_1",
            sample_id=5
        ))
        
        # Sample 6: [BOS] [MASK] [MASK] -> [BOS]
        samples.append(self._create_sample(
            x_k=[self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
            x_k_plus_1=[self.tokens.BOS],
            operation="delete_pair_mask",
            sample_id=6
        ))
        
        return samples
    
    def _create_sample(self, x_k: List[int], x_k_plus_1: List[int], 
                      operation: str, sample_id: int) -> Dict[str, Any]:
        """
        Create single APMDM sample, manually compute Y/R/E/C signals
        """
        # Compute APMDM operation signals
        y_star, r_star, e_star, c_star = self._compute_apmdm_signals(x_k, x_k_plus_1)
        
        sample = {
            "x_k": x_k,
            "y_star": y_star,
            "r_star": r_star,
            "e_star": e_star, 
            "c_star": c_star,
            "x_k_plus_1": x_k_plus_1,
            "solver_metadata": {
                "operation": operation,
                "sample_id": sample_id,
                "x_k_length": len(x_k),
                "x_k_plus_1_length": len(x_k_plus_1),
                "algorithm": "parity_elimination"
            }
        }
        
        # Add readable string representation for debugging
        sample["debug_info"] = {
            "x_k_str": [self.tokens.to_string(t) for t in x_k],
            "x_k_plus_1_str": [self.tokens.to_string(t) for t in x_k_plus_1],
            "y_star_str": [self.tokens.to_string(t) for t in y_star],
        }
        
        return sample
    
    def _compute_apmdm_signals(self, x_k: List[int], x_k_plus_1: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Manually compute APMDM Y/R/E/C signals
        
        According to APMDM specification:
        - C operation: delete MASK (highest priority)
        - R operation: change token to MASK
        - Y operation: change MASK to target value
        - E operation: insert MASK (not used in this problem, all 0)
        """
        seq_len = len(x_k)
        
        # Initialize signals
        y_star = x_k.copy()  # Default: keep unchanged
        r_star = [0] * seq_len
        e_star = [0] * seq_len  # Parity problem doesn't use expansion
        c_star = [0] * seq_len
        
        # Analyze change pattern
        if len(x_k_plus_1) < len(x_k):
            # Sequence becomes shorter -> use C operation to delete MASK
            self._compute_contraction_signals(x_k, x_k_plus_1, c_star)
            
        elif len(x_k_plus_1) == len(x_k):
            # Sequence length unchanged -> use R operation or Y operation
            self._compute_remask_unmask_signals(x_k, x_k_plus_1, y_star, r_star)
        
        return y_star, r_star, e_star, c_star
    
    def _compute_contraction_signals(self, x_k: List[int], x_k_plus_1: List[int], c_star: List[int]):
        """Compute C operation signals - delete MASK"""
        # Find deleted MASK positions
        x_k_plus_1_idx = 0
        
        for i in range(len(x_k)):
            if x_k_plus_1_idx < len(x_k_plus_1) and x_k[i] == x_k_plus_1[x_k_plus_1_idx]:
                # Position matches, continue
                x_k_plus_1_idx += 1
            else:
                # Position doesn't match, x_k[i] was deleted
                if x_k[i] == self.tokens.MASK:
                    c_star[i] = 1  # Delete this MASK
                # If not MASK, there's a problem (shouldn't happen)
    
    def _compute_remask_unmask_signals(self, x_k: List[int], x_k_plus_1: List[int], 
                                     y_star: List[int], r_star: List[int]):
        """Compute R operation and Y operation signals"""
        for i in range(len(x_k)):
            if x_k[i] != x_k_plus_1[i]:
                if x_k_plus_1[i] == self.tokens.MASK:
                    # token -> MASK = R operation
                    r_star[i] = 1
                elif x_k[i] == self.tokens.MASK:
                    # MASK -> token = Y operation
                    y_star[i] = x_k_plus_1[i]
    
    def generate_test_dataset(self, num_samples: int = 1000, 
                            min_length: int = 100, max_length: int = 10000) -> List[Dict[str, Any]]:
        """
        Generate large-scale test dataset
        
        Format: BOS + random 01 sequence + EOS
        Length: Very long, for testing model generalization
        """
        test_samples = []
        
        for i in range(num_samples):
            # Random sequence length
            seq_length = random.randint(min_length, max_length)
            
            # Generate random 01 sequence
            sequence = [self.tokens.BOS]
            for _ in range(seq_length):
                sequence.append(random.choice([self.tokens.ZERO, self.tokens.ONE]))
            sequence.append(self.tokens.EOS)
            
            # Calculate correct parity result
            ones_count = sum(1 for t in sequence if t == self.tokens.ONE)
            parity = ones_count % 2
            
            test_sample = {
                "sequence": sequence,
                "length": len(sequence),
                "ones_count": ones_count,
                "parity": parity,  # 0=even number of 1s, 1=odd number of 1s
                "expected_final": [self.tokens.BOS] if parity == 0 else [self.tokens.BOS, self.tokens.ONE],
                "sample_id": i
            }
            
            # Add debug info
            test_sample["debug_info"] = {
                "sequence_str": [self.tokens.to_string(t) for t in sequence],
                "expected_final_str": [self.tokens.to_string(t) for t in test_sample["expected_final"]]
            }
            
            test_samples.append(test_sample)
        
        return test_samples
    
    def expand_training_samples(self, base_samples: List[Dict[str, Any]], 
                               target_count: int = 1000) -> List[Dict[str, Any]]:
        """
        Expand training set by randomly copying base samples
        
        Args:
            base_samples: Base 7 samples
            target_count: Target sample count
            
        Returns:
            Expanded training sample list
        """
        expanded_samples = []
        
        for i in range(target_count):
            # Randomly select a base sample to copy
            base_sample = random.choice(base_samples)
            
            # Create sample copy
            expanded_sample = {
                "x_k": base_sample["x_k"].copy(),
                "y_star": base_sample["y_star"].copy(),
                "r_star": base_sample["r_star"].copy(),
                "e_star": base_sample["e_star"].copy(),
                "c_star": base_sample["c_star"].copy(),
                "x_k_plus_1": base_sample["x_k_plus_1"].copy(),
                "solver_metadata": {
                    "operation": base_sample["solver_metadata"]["operation"],
                    "sample_id": i,  # New sample ID
                    "x_k_length": base_sample["solver_metadata"]["x_k_length"],
                    "x_k_plus_1_length": base_sample["solver_metadata"]["x_k_plus_1_length"],
                    "algorithm": base_sample["solver_metadata"]["algorithm"],
                    "base_sample_id": base_sample["solver_metadata"]["sample_id"],  # Record original sample ID
                    "is_expanded": True  # Mark as expanded sample
                },
                "debug_info": base_sample["debug_info"].copy()
            }
            
            expanded_samples.append(expanded_sample)
        
        return expanded_samples
    
    def save_dataset(self, training_samples: List[Dict[str, Any]], 
                    test_samples: List[Dict[str, Any]], output_dir: str = ".", 
                    expand_training: bool = False, target_train_count: int = 1000):
        """Save training and test datasets"""
        
        # Expand training set if needed
        if expand_training:
            print(f"🔄 Expanding training set from {len(training_samples)} to {target_train_count} samples...")
            training_samples = self.expand_training_samples(training_samples, target_train_count)
        
        # Save training dataset (pickle format, compatible with existing system)
        train_file = f"{output_dir}/parity_train.pkl.gz"
        with gzip.open(train_file, 'wb') as f:
            pickle.dump(training_samples, f)
        
        # Save test dataset
        test_file = f"{output_dir}/parity_test.pkl.gz"
        with gzip.open(test_file, 'wb') as f:
            pickle.dump(test_samples, f)
        
        # Save JSON format for human reading
        train_json_file = f"{output_dir}/parity_train.json"
        with open(train_json_file, 'w', encoding='utf-8') as f:
            json.dump(training_samples, f, ensure_ascii=False, indent=2)
        
        # Save vocabulary cache
        vocab_cache = {}
        for i in range(self.tokens.get_vocab_size()):
            vocab_cache[i] = self.tokens.to_string(i)
        
        vocab_file = f"{output_dir}/parity_vocab_cache.pkl"
        with open(vocab_file, 'wb') as f:
            pickle.dump(vocab_cache, f)
        
        print(f"✅ Dataset saved:")
        print(f"   Training samples: {len(training_samples)} -> {train_file}")
        print(f"   Test samples: {len(test_samples)} -> {test_file}")
        print(f"   Vocabulary: {len(vocab_cache)} tokens -> {vocab_file}")
        
        return {
            "train_file": train_file,
            "test_file": test_file,
            "vocab_file": vocab_file,
            "train_samples": len(training_samples),
            "test_samples": len(test_samples)
        }
    
    def print_training_samples_analysis(self, samples: List[Dict[str, Any]], show_expanded: bool = False):
        """Print detailed analysis of training samples"""
        
        # If expanded sample set, only show statistics and a few examples
        if len(samples) > 7 and not show_expanded:
            print("🔍 Expanded Training Set Analysis:")
            print("=" * 60)
            
            # Count each operation type
            operation_counts = {}
            for sample in samples:
                op = sample["solver_metadata"]["operation"]
                operation_counts[op] = operation_counts.get(op, 0) + 1
            
            print(f"📊 Sample Distribution:")
            for op, count in operation_counts.items():
                print(f"   {op}: {count} samples")
            
            print(f"\n📋 Random Sample Display (first 3):")
            for i in range(min(3, len(samples))):
                sample = samples[i]
                meta = sample["solver_metadata"]
                debug = sample["debug_info"]
                
                base_id = meta.get('base_sample_id', 'original')
                print(f"   Sample {meta['sample_id']}: {meta['operation']} (based on original Sample {base_id})")
                print(f"      x_k: {debug['x_k_str']} → {debug['x_k_plus_1_str']}")
            
            return
        
        # Original detailed analysis (for 7 base samples)
        print("🔍 Training Sample Detailed Analysis:")
        print("=" * 80)
        
        for sample in samples:
            meta = sample["solver_metadata"]
            debug = sample["debug_info"]
            
            print(f"\n📋 Sample {meta['sample_id']}: {meta['operation']}")
            print(f"   x_k:       {debug['x_k_str']}")
            print(f"   x_k_plus_1: {debug['x_k_plus_1_str']}")
            print(f"   y_star:    {debug['y_star_str']}")
            print(f"   r_star:    {sample['r_star']}")
            print(f"   e_star:    {sample['e_star']}")
            print(f"   c_star:    {sample['c_star']}")
            
            # Verify APMDM inference
            predicted = self.simulate_apmdm_inference(
                sample['x_k'], sample['y_star'], sample['r_star'], 
                sample['e_star'], sample['c_star']
            )
            
            is_correct = predicted == sample['x_k_plus_1']
            status = "✅ Correct" if is_correct else "❌ Wrong"
            print(f"   APMDM inference:  {[self.tokens.to_string(t) for t in predicted]} {status}")
    
    def simulate_apmdm_inference(self, x_k: List[int], y_star: List[int], 
                               r_star: List[int], e_star: List[int], c_star: List[int]) -> List[int]:
        """
        Simulate APMDM inference process
        Execution order: C -> R -> Y -> E
        """
        x_k_plus_1_sequence = []
        
        for i in range(len(x_k)):
            # 1. Contraction: if c_star[i]==1 and x_k[i]==MASK, skip this position
            if i < len(c_star) and c_star[i] == 1 and x_k[i] == self.tokens.MASK:
                continue
            
            # 2. Decide current position's token: Remask > Unmask > Keep
            if i < len(r_star) and r_star[i] == 1:
                token_to_add = self.tokens.MASK
            elif x_k[i] == self.tokens.MASK:
                token_to_add = y_star[i] if i < len(y_star) else self.tokens.MASK
            else:
                token_to_add = x_k[i]
            
            x_k_plus_1_sequence.append(token_to_add)
            
            # 3. Expansion: if e_star[i]==1, insert MASK after current position
            # (Parity problem doesn't use expansion, so this won't execute)
            if i < len(e_star) and e_star[i] == 1:
                x_k_plus_1_sequence.append(self.tokens.MASK)
        
        return x_k_plus_1_sequence


def main():
    """Main function: Generate training dataset for Parity problem"""
    print("🎯 Parity Problem APMDM Training Data Generator")
    print("=" * 50)
    
    generator = ParityDataGenerator()
    
    # Generate training dataset (fixed 7 samples)
    print("📚 Generating training dataset...")
    training_samples = generator.generate_training_samples()
    
    # Print training sample analysis
    generator.print_training_samples_analysis(training_samples)
    
    # Expand training set to 1000 samples
    print(f"\n🔄 Expanding training set to 1000 samples...")
    expanded_samples = generator.expand_training_samples(training_samples, target_count=1000)
    
    # Save training dataset
    train_file = "parity_train.pkl.gz"
    with gzip.open(train_file, 'wb') as f:
        pickle.dump(expanded_samples, f)
    
    # Save JSON format for inspection
    train_json_file = "parity_train.json"
    with open(train_json_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_samples, f, ensure_ascii=False, indent=2)
    
    # Save vocabulary cache
    vocab_cache = {}
    for i in range(generator.tokens.get_vocab_size()):
        vocab_cache[i] = generator.tokens.to_string(i)
    
    vocab_file = "parity_vocab_cache.pkl"
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab_cache, f)
    
    print(f"\n✅ Dataset saved:")
    print(f"   Training samples: {len(expanded_samples)} -> {train_file}")
    print(f"   Vocabulary: {len(vocab_cache)} tokens -> {vocab_file}")
    
    print(f"\n🎉 Generation complete!")
    print(f"   Training set: {len(expanded_samples)} samples")
    print(f"   Vocabulary size: {len(vocab_cache)} tokens")


if __name__ == "__main__":
    main()
