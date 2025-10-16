#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 Parity Problem APMDM Dataset Validator

Validate whether generated training data conforms to APMDM specification:
1. APMDM inference compatibility: Can correctly infer from x_k to x_k_plus_1
2. Sequence uniqueness: Whether x_k sequences are unique
3. Length consistency: Whether all label array lengths match x_k
4. y_star consistency: Whether y_star matches x_k at non-MASK positions
5. Operation logic correctness: Verify parity elimination logic is correct
"""

import pickle
import gzip
import json
from typing import List, Dict, Any
from parity_generator import ParityTokens, ParityDataGenerator

class ParityAPMDMValidator:
    """APMDM data validator for Parity problem"""
    
    def __init__(self):
        self.tokens = ParityTokens()
        self.generator = ParityDataGenerator()
    
    def validate_training_dataset(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate training dataset integrity"""
        print("🔍 Starting Parity training dataset validation...")
        print("=" * 60)
        
        results = {}
        
        # 1. APMDM inference compatibility validation
        print("1️⃣ APMDM inference compatibility validation...")
        apmdm_result = self._validate_apmdm_inference(samples)
        results['apmdm_compatibility'] = apmdm_result
        print(f"   Success rate: {apmdm_result['success_rate']:.1%}")
        
        # 2. Sequence uniqueness validation
        print("2️⃣ Sequence uniqueness validation...")
        uniqueness_result = self._validate_sequence_uniqueness(samples)
        results['sequence_uniqueness'] = uniqueness_result
        print(f"   Uniqueness: {uniqueness_result['uniqueness_rate']:.1%}")
        
        # 3. Length consistency validation
        print("3️⃣ Length consistency validation...")
        length_result = self._validate_length_consistency(samples)
        results['length_consistency'] = length_result
        print(f"   Consistency: {length_result['consistency_rate']:.1%}")
        
        # 4. y_star consistency validation
        print("4️⃣ y_star consistency validation...")
        y_star_result = self._validate_y_star_consistency(samples)
        results['y_star_consistency'] = y_star_result
        print(f"   Consistency: {y_star_result['consistency_rate']:.1%}")
        
        # 5. Parity logic correctness validation
        print("5️⃣ Parity logic correctness validation...")
        parity_result = self._validate_parity_logic(samples)
        results['parity_logic'] = parity_result
        print(f"   Correctness: {parity_result['correctness_rate']:.1%}")
        
        # Overall evaluation
        all_perfect = all(
            result.get('success_rate', result.get('consistency_rate', result.get('uniqueness_rate', result.get('correctness_rate', 0)))) >= 1.0
            for result in results.values()
        )
        
        print(f"\n🎯 Overall Evaluation Results:")
        print(f"   1️⃣ APMDM Inference Compatibility: {'✅' if apmdm_result['success_rate'] >= 1.0 else '❌'} {apmdm_result['success_rate']:.1%}")
        print(f"   2️⃣ Sequence Uniqueness: {'✅' if uniqueness_result['uniqueness_rate'] >= 1.0 else '❌'} {uniqueness_result['uniqueness_rate']:.1%}")
        print(f"   3️⃣ Length Consistency: {'✅' if length_result['consistency_rate'] >= 1.0 else '❌'} {length_result['consistency_rate']:.1%}")
        print(f"   4️⃣ y_star Consistency: {'✅' if y_star_result['consistency_rate'] >= 1.0 else '❌'} {y_star_result['consistency_rate']:.1%}")
        print(f"   5️⃣ Parity Logic Correctness: {'✅' if parity_result['correctness_rate'] >= 1.0 else '❌'} {parity_result['correctness_rate']:.1%}")
        print(f"   🏆 Overall: {'✅ Perfect' if all_perfect else '❌ Needs improvement'}")
        
        results['overall_perfect'] = all_perfect
        return results
    
    def _validate_apmdm_inference(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate APMDM inference compatibility"""
        successful = 0
        failed_samples = []
        
        for i, sample in enumerate(samples):
            try:
                x_k = sample['x_k']
                x_k_plus_1 = sample['x_k_plus_1']
                y_star = sample['y_star']
                r_star = sample['r_star']
                e_star = sample['e_star']
                c_star = sample['c_star']
                
                # Simulate APMDM inference
                predicted_x_k_plus_1 = self.generator.simulate_apmdm_inference(
                    x_k, y_star, r_star, e_star, c_star
                )
                
                if predicted_x_k_plus_1 == x_k_plus_1:
                    successful += 1
                else:
                    failed_samples.append({
                        'sample_id': i,
                        'expected': x_k_plus_1,
                        'predicted': predicted_x_k_plus_1,
                        'x_k': x_k
                    })
                    
            except Exception as e:
                failed_samples.append({
                    'sample_id': i,
                    'error': str(e)
                })
        
        return {
            'success_rate': successful / len(samples),
            'successful_samples': successful,
            'failed_samples': len(failed_samples),
            'failures': failed_samples
        }
    
    def _validate_sequence_uniqueness(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate sequence uniqueness"""
        unique_sequences = set()
        duplicates = []
        
        for i, sample in enumerate(samples):
            x_k = tuple(sample['x_k'])
            if x_k in unique_sequences:
                duplicates.append({
                    'sample_id': i,
                    'sequence': sample['x_k']
                })
            else:
                unique_sequences.add(x_k)
        
        return {
            'uniqueness_rate': (len(samples) - len(duplicates)) / len(samples),
            'unique_samples': len(unique_sequences),
            'duplicate_samples': len(duplicates),
            'duplicates': duplicates
        }
    
    def _validate_length_consistency(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate length consistency"""
        consistent = 0
        inconsistent_samples = []
        
        for i, sample in enumerate(samples):
            x_k = sample['x_k']
            y_star = sample['y_star']
            r_star = sample['r_star']
            e_star = sample['e_star']
            c_star = sample['c_star']
            
            expected_len = len(x_k)
            
            if (len(y_star) == len(r_star) == len(e_star) == len(c_star) == expected_len):
                consistent += 1
            else:
                inconsistent_samples.append({
                    'sample_id': i,
                    'expected_length': expected_len,
                    'actual_lengths': {
                        'y_star': len(y_star),
                        'r_star': len(r_star),
                        'e_star': len(e_star),
                        'c_star': len(c_star)
                    }
                })
        
        return {
            'consistency_rate': consistent / len(samples),
            'consistent_samples': consistent,
            'inconsistent_samples': len(inconsistent_samples),
            'inconsistencies': inconsistent_samples
        }
    
    def _validate_y_star_consistency(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate y_star consistency: non-MASK positions must match x_k"""
        consistent = 0
        inconsistent_samples = []
        
        for i, sample in enumerate(samples):
            x_k = sample['x_k']
            y_star = sample['y_star']
            
            is_consistent = True
            inconsistencies = []
            
            for j in range(min(len(x_k), len(y_star))):
                # If x_k[j] is not MASK, then y_star[j] must match x_k[j]
                if x_k[j] != self.tokens.MASK and y_star[j] != x_k[j]:
                    is_consistent = False
                    inconsistencies.append({
                        'position': j,
                        'x_k_value': x_k[j],
                        'y_star_value': y_star[j]
                    })
            
            if is_consistent:
                consistent += 1
            else:
                inconsistent_samples.append({
                    'sample_id': i,
                    'inconsistencies': inconsistencies
                })
        
        return {
            'consistency_rate': consistent / len(samples),
            'consistent_samples': consistent,
            'inconsistent_samples': len(inconsistent_samples),
            'inconsistencies': inconsistent_samples
        }
    
    def _validate_parity_logic(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate Parity elimination logic correctness"""
        correct = 0
        incorrect_samples = []
        
        # Define expected patterns based on operation type (not sample_id)
        expected_patterns = {
            "eliminate_single_0": {
                "expected_x_k": [self.tokens.BOS, self.tokens.ZERO, self.tokens.ONE],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.MASK, self.tokens.ONE]
            },
            "delete_middle_mask": {
                "expected_x_k": [self.tokens.BOS, self.tokens.MASK, self.tokens.ONE],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.ONE]
            },
            "eliminate_tail_0": {
                "expected_x_k": [self.tokens.BOS, self.tokens.ONE, self.tokens.ZERO],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.ONE, self.tokens.MASK]
            },
            "delete_tail_mask": {
                "expected_x_k": [self.tokens.BOS, self.tokens.ONE, self.tokens.MASK],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.ONE]
            },
            "eliminate_pair_0": {
                "expected_x_k": [self.tokens.BOS, self.tokens.ZERO, self.tokens.ZERO],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.MASK, self.tokens.MASK]
            },
            "eliminate_pair_1": {
                "expected_x_k": [self.tokens.BOS, self.tokens.ONE, self.tokens.ONE],
                "expected_x_k_plus_1": [self.tokens.BOS, self.tokens.MASK, self.tokens.MASK]
            },
            "delete_pair_mask": {
                "expected_x_k": [self.tokens.BOS, self.tokens.MASK, self.tokens.MASK],
                "expected_x_k_plus_1": [self.tokens.BOS]
            }
        }
        
        for i, sample in enumerate(samples):
            operation = sample['solver_metadata']['operation']
            x_k = sample['x_k']
            x_k_plus_1 = sample['x_k_plus_1']
            
            is_correct = True
            error_message = ""
            
            # Validate sample logic based on operation type
            if operation in expected_patterns:
                pattern = expected_patterns[operation]
                expected_x_k = pattern["expected_x_k"]
                expected_x_k_plus_1 = pattern["expected_x_k_plus_1"]
                
                if x_k != expected_x_k or x_k_plus_1 != expected_x_k_plus_1:
                    is_correct = False
                    error_message = f"{operation} pattern mismatch"
            else:
                # Unknown operation type
                is_correct = False
                error_message = f"Unknown operation type: {operation}"
            
            if is_correct:
                correct += 1
            else:
                incorrect_samples.append({
                    'sample_id': sample['solver_metadata']['sample_id'],
                    'operation': operation,
                    'error': error_message,
                    'actual_x_k': [self.tokens.to_string(t) for t in x_k],
                    'actual_x_k_plus_1': [self.tokens.to_string(t) for t in x_k_plus_1]
                })
        
        return {
            'correctness_rate': correct / len(samples),
            'correct_samples': correct,
            'incorrect_samples': len(incorrect_samples),
            'errors': incorrect_samples
        }
    
    def validate_test_dataset(self, test_samples: List[Dict[str, Any]], num_to_check: int = 100) -> Dict[str, Any]:
        """Validate test dataset quality"""
        print(f"🧪 Validating test dataset (checking first {num_to_check} samples)...")
        
        valid_samples = 0
        invalid_samples = []
        
        samples_to_check = test_samples[:num_to_check]
        
        for i, sample in enumerate(samples_to_check):
            is_valid = True
            errors = []
            
            # Check sequence format
            sequence = sample['sequence']
            if len(sequence) < 2:
                is_valid = False
                errors.append("Sequence too short")
            elif sequence[0] != self.tokens.BOS:
                is_valid = False
                errors.append("Sequence doesn't start with BOS")
            elif sequence[-1] != self.tokens.EOS:
                is_valid = False
                errors.append("Sequence doesn't end with EOS")
            
            # Check middle part only contains 0 and 1
            middle_part = sequence[1:-1]
            for j, token in enumerate(middle_part):
                if token not in [self.tokens.ZERO, self.tokens.ONE]:
                    is_valid = False
                    errors.append(f"Position {j+1} contains invalid token: {self.tokens.to_string(token)}")
            
            # Validate parity calculation
            ones_count = sum(1 for t in sequence if t == self.tokens.ONE)
            expected_parity = ones_count % 2
            if sample['parity'] != expected_parity:
                is_valid = False
                errors.append(f"Parity calculation error: expected {expected_parity}, actual {sample['parity']}")
            
            # Validate expected result
            expected_final = [self.tokens.BOS] if expected_parity == 0 else [self.tokens.BOS, self.tokens.ONE]
            if sample['expected_final'] != expected_final:
                is_valid = False
                errors.append(f"Expected result error")
            
            if is_valid:
                valid_samples += 1
            else:
                invalid_samples.append({
                    'sample_id': i,
                    'errors': errors
                })
        
        return {
            'valid_rate': valid_samples / len(samples_to_check),
            'valid_samples': valid_samples,
            'invalid_samples': len(invalid_samples),
            'total_checked': len(samples_to_check),
            'errors': invalid_samples
        }
    
    def load_and_validate_datasets(self, train_file: str, test_file: str) -> Dict[str, Any]:
        """Load and validate complete datasets"""
        print("📂 Loading datasets...")
        
        # Load training dataset
        with gzip.open(train_file, 'rb') as f:
            training_samples = pickle.load(f)
        
        # Load test dataset
        with gzip.open(test_file, 'rb') as f:
            test_samples = pickle.load(f)
        
        print(f"   Training samples: {len(training_samples)}")
        print(f"   Test samples: {len(test_samples)}")
        
        # Validate training dataset
        train_results = self.validate_training_dataset(training_samples)
        
        # Validate test dataset
        test_results = self.validate_test_dataset(test_samples)
        
        return {
            'training_validation': train_results,
            'test_validation': test_results,
            'training_samples': len(training_samples),
            'test_samples': len(test_samples)
        }


def main():
    """Main function: Validate generated datasets"""
    print("🔍 Parity Problem APMDM Dataset Validator")
    print("=" * 50)
    
    validator = ParityAPMDMValidator()
    
    # Validate datasets
    results = validator.load_and_validate_datasets(
        train_file="./parity_train.pkl.gz",
        test_file="./parity_test.pkl.gz"
    )
    
    print(f"\n📊 Final Validation Report:")
    print(f"=" * 50)
    
    train_results = results['training_validation']
    test_results = results['test_validation']
    
    print(f"🎓 Training Dataset:")
    print(f"   Sample count: {results['training_samples']}")
    print(f"   Overall quality: {'✅ Perfect' if train_results['overall_perfect'] else '❌ Needs improvement'}")
    
    print(f"\n🧪 Test Dataset:")
    print(f"   Sample count: {results['test_samples']}")
    print(f"   Validation quality: {'✅ Excellent' if test_results['valid_rate'] >= 0.99 else '❌ Needs improvement'} ({test_results['valid_rate']:.1%})")
    
    # Save validation report
    with open("parity_validation_report.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Validation report saved to: parity_validation_report.json")


if __name__ == "__main__":
    main()
