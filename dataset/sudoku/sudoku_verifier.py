#!/usr/bin/env python3
"""
🔍 APMDM Dataset Validator
Validate 5 core metrics of generated data to ensure compliance with APMDM training requirements
"""

import numpy as np
from typing import List, Dict, Any

class APMDMValidator:
    """APMDM dataset validator - Validate 5 core metrics"""
    
    def __init__(self):
        self.MASK_TOKEN = 10
        self.EOS_TOKEN = 31  # Update: Changed from 28 to 31, accommodating 15 branch color extension
    
    def simulate_apmdm_inference(self, x_k: List[int], y_star: List[int], 
                               r_star: List[int], e_star: List[int], c_star: List[int]) -> List[int]:
        """Simulate APMDM inference process - Completely consistent with backup version"""
        x_k_plus_1_sequence = []
        seq_len = len(x_k)
        
        for i in range(seq_len):
            # 1. Contraction: Ifc_star[i]==1，Skip this position
            if i < len(c_star) and c_star[i] == 1:
                continue
            
            # 2. Decide current positiontoken: Remask > Unmask > Keep
            if i < len(r_star) and r_star[i] == 1:
                token_to_add = self.MASK_TOKEN
            elif x_k[i] == self.MASK_TOKEN:
                token_to_add = y_star[i] if i < len(y_star) else self.MASK_TOKEN
            else:
                token_to_add = x_k[i]
            
            x_k_plus_1_sequence.append(token_to_add)
            
            # 3. Expansion: Ife_star[i]==1，Insert after current positionMASK
            if i < len(e_star) and e_star[i] == 1:
                x_k_plus_1_sequence.append(self.MASK_TOKEN)
        
        return x_k_plus_1_sequence
    
    def _convert_to_list(self, data):
        """Unified data format conversion"""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, list):
            return data
        elif isinstance(data, str):
            return [int(x) for x in data.split()]
        else:
            return list(data)
    
    def validate_5_metrics(self, samples: List[Dict[str, Any]], puzzle_samples: Dict[int, List[Dict]] = None) -> Dict[str, Any]:
        """Validate 5 core metrics"""
        print(f"🔍 Starting validation of 5 core metrics for {len(samples)} samples...")
        print("=" * 50)
        
        # 1. APMDM inference compatibility
        print("1️⃣ APMDM inference compatibility validation...")
        apmdm_result = self._validate_apmdm_inference(samples)
        print(f"   APMDM success rate: {apmdm_result['success_rate']:.1%}")
        
        # 2. Sequence continuity
        print("2️⃣ Sequence continuity validation...")
        if puzzle_samples:
            continuity_result = self._validate_sequence_continuity(puzzle_samples)
        else:
            # 🔥 Fix: Group samples by instance_id to avoid cross-instance continuity validation
            instance_grouped = {}
            for sample in samples:
                instance_id = sample.get('solver_metadata', {}).get('instance_id', 0)
                if instance_id not in instance_grouped:
                    instance_grouped[instance_id] = []
                instance_grouped[instance_id].append(sample)
            continuity_result = self._validate_sequence_continuity(instance_grouped)
        print(f"   Continuity: {continuity_result['success_rate']:.1%}")
        
        # 3. Sequence uniqueness
        print("3️⃣ Sequence uniqueness validation...")
        uniqueness_result = self._validate_sequence_uniqueness(samples)
        print(f"   Uniqueness: {uniqueness_result['uniqueness_rate']:.1%}")
        
        # 4. Length consistency
        print("4️⃣ Length consistency validation...")
        length_result = self._validate_length_consistency(samples)
        print(f"   Length consistency: {length_result['consistency_rate']:.1%}")
        
        # 5. y_star no MASK
        print("5️⃣ y_star no MASK validation...")
        y_star_result = self._validate_y_star_no_mask(samples)
        print(f"   y_star no MASK: {y_star_result['clean_rate']:.1%}")
        
        # Overall results
        all_perfect = (
            apmdm_result['success_rate'] >= 1.0 and
            continuity_result['success_rate'] >= 1.0 and
            uniqueness_result['uniqueness_rate'] >= 1.0 and
            length_result['consistency_rate'] >= 1.0 and
            y_star_result['clean_rate'] >= 1.0
        )
        
        print(f"\n🎯 Validation Results:")
        print("=" * 30)
        print(f"   1️⃣ APMDM Inference Compatibility: {'✅' if apmdm_result['success_rate'] >= 1.0 else '❌'} {apmdm_result['success_rate']:.1%}")
        print(f"   2️⃣ Sequence Continuity: {'✅' if continuity_result['success_rate'] >= 1.0 else '❌'} {continuity_result['success_rate']:.1%}")
        print(f"   3️⃣ Sequence Uniqueness: {'✅' if uniqueness_result['uniqueness_rate'] >= 1.0 else '❌'} {uniqueness_result['uniqueness_rate']:.1%}")
        print(f"   4️⃣ Length Consistency: {'✅' if length_result['consistency_rate'] >= 1.0 else '❌'} {length_result['consistency_rate']:.1%}")
        print(f"   5️⃣ y_star No MASK: {'✅' if y_star_result['clean_rate'] >= 1.0 else '❌'} {y_star_result['clean_rate']:.1%}")
        print(f"   🏆 Overall Status: {'✅ Perfect' if all_perfect else '❌ Issues exist'}")
        
        return {
            'overall_success': all_perfect,
            'metrics': {
                'apmdm_inference': apmdm_result['success_rate'],
                'sequence_continuity': continuity_result['success_rate'],
                'sequence_uniqueness': uniqueness_result['uniqueness_rate'],
                'length_consistency': length_result['consistency_rate'],
                'y_star_clean': y_star_result['clean_rate']
            }
        }
    
    def _validate_apmdm_inference(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate APMDM inference compatibility"""
        correct_count = 0
        total_count = len(samples)
        
        for sample in samples:
            try:
                x_k = self._convert_to_list(sample['x_k'])
                x_k_plus_1 = self._convert_to_list(sample['x_k_plus_1'])
                y_star = self._convert_to_list(sample['y_star'])
                r_star = self._convert_to_list(sample['r_star'])
                e_star = self._convert_to_list(sample['e_star'])
                c_star = self._convert_to_list(sample['c_star'])
                
                predicted_x_k_plus_1 = self.simulate_apmdm_inference(x_k, y_star, r_star, e_star, c_star)
                
                if (predicted_x_k_plus_1 is not None and 
                    len(predicted_x_k_plus_1) == len(x_k_plus_1) and
                    all(a == b for a, b in zip(predicted_x_k_plus_1, x_k_plus_1))):
                    correct_count += 1
                    
            except Exception:
                pass  # Error samples not counted as correct
        
        return {
            'total_samples': total_count,
            'correct_samples': correct_count,
            'success_rate': correct_count / total_count if total_count > 0 else 0
        }
    
    def _validate_sequence_continuity(self, puzzle_samples: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Validate sequence continuity"""
        continuity_errors = 0
        total_transitions = 0
        
        for puzzle_idx, samples in puzzle_samples.items():
            for i in range(1, len(samples)):
                prev_x_k_plus_1 = self._convert_to_list(samples[i-1]['x_k_plus_1'])
                curr_x_k = self._convert_to_list(samples[i]['x_k'])
                
                if prev_x_k_plus_1 != curr_x_k:
                    continuity_errors += 1
                    
            total_transitions += len(samples) - 1
        
        return {
            'total_transitions': total_transitions,
            'errors': continuity_errors,
            'success_rate': (total_transitions - continuity_errors) / total_transitions if total_transitions > 0 else 1.0
        }
    
    def _validate_sequence_uniqueness(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate sequence uniqueness"""
        unique_sequences = set()
        duplicates = 0
        
        for sample in samples:
            x_k = self._convert_to_list(sample['x_k'])
            x_k_tuple = tuple(x_k)
            
            if x_k_tuple in unique_sequences:
                duplicates += 1
            else:
                unique_sequences.add(x_k_tuple)
        
        return {
            'total_samples': len(samples),
            'duplicates': duplicates,
            'uniqueness_rate': (len(samples) - duplicates) / len(samples) if len(samples) > 0 else 1.0
        }
    
    def _validate_length_consistency(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate length consistency"""
        length_errors = 0
        
        for sample in samples:
            x_k = self._convert_to_list(sample['x_k'])
            y_star = self._convert_to_list(sample['y_star'])
            r_star = self._convert_to_list(sample['r_star'])
            e_star = self._convert_to_list(sample['e_star'])
            c_star = self._convert_to_list(sample['c_star'])
            
            expected_len = len(x_k)
            
            if not (len(r_star) == len(e_star) == len(y_star) == len(c_star) == expected_len):
                length_errors += 1
        
        return {
            'total_samples': len(samples),
            'errors': length_errors,
            'consistency_rate': (len(samples) - length_errors) / len(samples) if len(samples) > 0 else 1.0
        }
    
    def _validate_y_star_no_mask(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate y_star does not contain MASK token"""
        mask_errors = 0
        
        for sample in samples:
            y_star = self._convert_to_list(sample['y_star'])
            
            if self.MASK_TOKEN in y_star:
                mask_errors += 1
        
        return {
            'total_samples': len(samples),
            'mask_errors': mask_errors,
            'clean_rate': (len(samples) - mask_errors) / len(samples) if len(samples) > 0 else 1.0
        }


def validate_dataset(data_file: str = 'data/sudoku-tiny-data.npy', max_puzzles: int = None) -> Dict[str, Any]:
    """Main entry function for dataset validation"""
    print(f"🎯 Validating dataset: {data_file}")
    
    from sudoku_generator import APMDMDataManager
    from sudoku_loader import SudokuLoader
    
    # Load data
    data = np.load(data_file)
    total_puzzles = data.shape[0]
    process_puzzles = min(total_puzzles, max_puzzles) if max_puzzles else total_puzzles
    
    print(f"   Total puzzles: {total_puzzles}")
    print(f"   Validating puzzles: {process_puzzles}")
    
    # Generate samples
    manager = APMDMDataManager()
    validator = APMDMValidator()
    
    all_samples = []
    puzzle_samples = {}
    
    for i in range(process_puzzles):
        try:
            print(f"📊 Processing Sudoku {i+1}/{process_puzzles}...", end=" ", flush=True)
            puzzle, _ = SudokuLoader.read_sudoku(data_file, i)
            samples = manager.generate_from_puzzle(puzzle)
            
            all_samples.extend(samples)
            puzzle_samples[i] = samples
            print(f"✅ {len(samples)} samples")
            
        except Exception as e:
            print(f"❌ {str(e)}")
    
    print(f"\n🎯 Data generation complete: {len(all_samples)} samples")
    
    # Execute validation
    result = validator.validate_5_metrics(all_samples, puzzle_samples)
    
    return result


if __name__ == "__main__":
    print("🔍 APMDM Dataset Validator - 5 Core Metrics Validation")
    print("=" * 50)
    
    # Validate first 5 Sudoku puzzles
    result = validate_dataset('data/sudoku-tiny-data.npy', max_puzzles=5)
    
    if result['overall_success']:
        print(f"\n🎉 Validation successful! All 5 metrics reached 100%!")
        print(f"✅ Data can be directly used for APMDM training!")
    else:
        print(f"\n⚠️  Issues found, please check specific errors!")