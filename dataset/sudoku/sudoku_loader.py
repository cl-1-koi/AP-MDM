#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudoku data loader, statistics analyzer and evaluation tools
Includes data reading, performance evaluation, statistical analysis, demo demonstrations, etc.
"""

import numpy as np
import time
from collections import Counter, defaultdict
from sudoku_solver import SudokuSolver

class SudokuLoader:
    """Sudoku data loader"""
    
    @staticmethod
    def read_sudoku(data_file, index=0):
        """Read single Sudoku: puzzle(9x9, 0=empty), solution(9x9, complete answer)"""
        # 🔥 Fix: Support PKL format data loading
        if data_file.endswith('.pkl') or data_file.endswith('.pkl.gz'):
            # PKL format: Assume this is training sample data, needs loading from APMDMDataManager
            from sudoku_generator import APMDMDataManager
            manager = APMDMDataManager()
            samples = manager.load_samples(data_file)
            if index >= len(samples):
                raise IndexError(f"Index {index} out of data range [0, {len(samples)-1}]")
            # Construct Sudoku from samples (reverse engineering needed here, temporarily throw error prompting user to use .npy file)
            raise ValueError("PKL data file contains APMDM training samples, please use original .npy Sudoku data file")
        else:
            # NPY format: Original Sudoku data
            data = np.load(data_file, allow_pickle=True)[index]
        
        puzzle, solution = np.zeros((9, 9), dtype=int), np.zeros((9, 9), dtype=int)
        
        for i in range(81):
            row, col, value, strategy = data[1+i*4:1+i*4+4]
            solution[row, col] = value
            if strategy == 0: puzzle[row, col] = value
        
        return puzzle, solution
    
    @staticmethod
    def read_batch_sudoku(data_file, num_samples=None):
        """Batch read Sudoku: puzzles[N,9,9], solutions[N,9,9]"""
        data = np.load(data_file)
        n = len(data) if num_samples is None else num_samples
        puzzles, solutions = np.zeros((n, 9, 9), dtype=int), np.zeros((n, 9, 9), dtype=int)
        
        for i in range(n):
            puzzles[i], solutions[i] = SudokuLoader.read_sudoku(data_file, i)
        
        return puzzles, solutions
    
    @staticmethod
    def get_dataset_info(data_file):
        """Get basic dataset information"""
        data = np.load(data_file)
        return {
            'total_count': len(data),
            'data_shape': data.shape,
            'data_size_mb': data.nbytes / (1024*1024),
            'sample_shape': data[0].shape if len(data) > 0 else None
        }

class SudokuEvaluator:
    """Sudoku solver evaluation tool"""
    
    @staticmethod
    def is_valid_solution(solution):
        """Verify if solution conforms to Sudoku rules"""
        if solution is None: return False
        
        # Check each row
        for row in solution:
            if len(set(row)) != 9 or set(row) != set(range(1, 10)):
                return False
        
        # Check each column
        for col in solution.T:
            if len(set(col)) != 9 or set(col) != set(range(1, 10)):
                return False
        
        for box_row in range(3):
            for box_col in range(3):
                box = solution[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                if len(set(box.flatten())) != 9 or set(box.flatten()) != set(range(1, 10)):
                    return False
        
        return True
    
    @staticmethod 
    def matches_ground_truth(solved, ground_truth):
        """Check if solution matches standard answer"""
        if solved is None or ground_truth is None: return False
        return np.array_equal(solved, ground_truth)
    
    @staticmethod
    def is_consistent_with_puzzle(puzzle, solution):
        """Check if solution is consistent with original puzzle (known positions unchanged)"""
        if solution is None: return False
        for i in range(9):
            for j in range(9):
                if puzzle[i, j] != 0 and puzzle[i, j] != solution[i, j]:
                    return False
        return True
    
    @staticmethod
    def evaluate_single(puzzle, solved_solution, ground_truth):
        """Evaluate solving result of single Sudoku"""
        results = {
            "valid_solution": SudokuEvaluator.is_valid_solution(solved_solution),
            "matches_ground_truth": SudokuEvaluator.matches_ground_truth(solved_solution, ground_truth),
            "consistent_with_puzzle": SudokuEvaluator.is_consistent_with_puzzle(puzzle, solved_solution),
            "solved": solved_solution is not None
        }
        results["correct"] = all([results["valid_solution"], 
                                results["matches_ground_truth"], 
                                results["consistent_with_puzzle"]])
        return results
    
    @staticmethod
    def batch_evaluate(solver, puzzles, ground_truths, verbose=True):
        """Batch evaluate solver performance"""
        n = len(puzzles)
        results = {
            "total": n,
            "solved": 0,
            "correct": 0,
            "valid": 0,
            "consistent": 0,
            "matches_gt": 0,
            "total_steps": 0,
            "total_backtracks": 0,
            "solve_times": [],
            "individual_stats": []
        }
        
        for i, (puzzle, gt) in enumerate(zip(puzzles, ground_truths)):
            if verbose and (i + 1) % 25 == 0:
                print(f"  Evaluation progress: {i+1}/{n}")
                
            solver.steps = 0
            solver.backtracks = 0
                
            # Solve
            start_time = time.time()
            solution = solver.solve(puzzle)
            solve_time = time.time() - start_time
            
            eval_result = SudokuEvaluator.evaluate_single(puzzle, solution, gt)
            stats = solver.get_stats()
            
            individual_stat = {
                "index": i,
                "empty_cells": np.count_nonzero(puzzle == 0),
                "steps": stats["steps"],
                "backtracks": stats["backtracks"],
                "solve_time": solve_time,
                "correct": eval_result["correct"]
            }
            results["individual_stats"].append(individual_stat)
            
            # Accumulate statistics
            results["solved"] += eval_result["solved"]
            results["correct"] += eval_result["correct"]
            results["valid"] += eval_result["valid_solution"]
            results["consistent"] += eval_result["consistent_with_puzzle"]
            results["matches_gt"] += eval_result["matches_ground_truth"]
            results["total_steps"] += stats["steps"]
            results["total_backtracks"] += stats["backtracks"]
            results["solve_times"].append(solve_time)
        
        results["avg_steps"] = results["total_steps"] / n
        results["avg_backtracks"] = results["total_backtracks"] / n  
        results["avg_time"] = np.mean(results["solve_times"])
        results["accuracy"] = results["correct"] / n * 100
        
        return results

class SudokuStatistics:
    """Sudoku dataset statistical analysis tool"""
    
    @staticmethod
    def analyze_dataset_statistics(data_file, max_samples=None):
        """Detailed analysis of dataset statistics"""
        print(f"=" * 60)
        print(f"📊 Detailed Sudoku Dataset Statistical Analysis")
        print(f"=" * 60)
        
        # Basic information
        info = SudokuLoader.get_dataset_info(data_file)
        print(f"\n🔍 Basic Information:")
        print(f"  File: {data_file}")
        print(f"  Total Sudoku count: {info['total_count']:,}")
        print(f"  Data shape: {info['data_shape']}")
        print(f"  File size: {info['data_size_mb']:.2f} MB")
        print(f"  Single Sudoku: {info['sample_shape']}")
        
        # Sampling analysis
        sample_size = min(max_samples or info['total_count'], info['total_count'], 2000)
        print(f"\n📈 Statistical Analysis (sampling {sample_size:,} puzzles):")
        
        puzzles, solutions = SudokuLoader.read_batch_sudoku(data_file, sample_size)
        
        # Difficulty analysis
        known_counts = [np.count_nonzero(puzzle) for puzzle in puzzles]
        empty_counts = [81 - known for known in known_counts]
        
        print(f"\n🎯 Difficulty Statistics:")
        print(f"  Known digits: avg={np.mean(known_counts):.2f}, range=[{np.min(known_counts)}, {np.max(known_counts)}]")
        print(f"  Empty cells: avg={np.mean(empty_counts):.2f}, range=[{np.min(empty_counts)}, {np.max(empty_counts)}]")
        
        # Difficulty distribution
        print(f"\n📊 Difficulty Distribution (by empty cell count):")
        empty_counter = Counter(empty_counts)
        for empty_count in sorted(empty_counter.keys()):
            count = empty_counter[empty_count]
            percentage = count / len(empty_counts) * 100
            print(f"  {empty_count:2d} empty: {count:4d} puzzles ({percentage:5.1f}%)")
        
        # Digit frequency analysis
        print(f"\n🔢 Digit Distribution Analysis:")
        all_given_digits = []
        for puzzle in puzzles:
            given_digits = puzzle[puzzle != 0]
            all_given_digits.extend(given_digits)
        
        digit_counter = Counter(all_given_digits)
        total_given = len(all_given_digits)
        print(f"  Total known digits: {total_given:,}")
        for digit in range(1, 10):
            count = digit_counter[digit]
            percentage = count / total_given * 100
            print(f"  Digit {digit}: {count:6d} times ({percentage:5.2f}%)")
        
        SudokuStatistics._analyze_strategies(data_file, sample_size)
        
        SudokuStatistics._check_data_quality(data_file, sample_size)
        
        return {
            'known_counts': known_counts,
            'empty_counts': empty_counts,
            'digit_distribution': dict(digit_counter),
            'sample_size': sample_size
        }
    
    @staticmethod
    def _analyze_strategies(data_file, sample_size):
        """Analyze solving strategy distribution"""
        print(f"\n🧠 Solving Strategy Analysis:")
        
        try:
            data = np.load(data_file)
            strategy_counter = Counter()
            abnormal_strategies = Counter()
            problem_count = 0
            
            for i in range(min(sample_size, len(data))):
                sudoku_data = data[i]
                has_abnormal = False
                
                for j in range(81):
                    strategy = sudoku_data[1 + j*4 + 3]  # 策略ID
                    
                    # Separate normal and abnormal strategies
                    if strategy <= 100:  # 正常策略范围
                        strategy_counter[strategy] += 1
                    else:  # 异常策略
                        abnormal_strategies[strategy] += 1
                        if not has_abnormal:
                            problem_count += 1
                            has_abnormal = True
            
            strategy_names = {
                0: "Given",
                2: "Naked Single", 
                3: "Hidden Single",
                4: "Naked Pair",
                5: "Naked Triple", 
                6: "Box-Line Reduction",
                7: "XY-Wing",
                8: "Unique Rectangle",
                12: "Advanced",
                13: "Advanced2"
            }
            
            total_normal = sum(strategy_counter.values())
            total_abnormal = sum(abnormal_strategies.values())
            
            print(f"  Normal cells: {total_normal:,}")
            print(f"  Abnormal cells: {total_abnormal:,}")
            print(f"  Problem Sudoku count: {problem_count}/{sample_size} ({problem_count/sample_size*100:.1f}%)")
            
            print(f"\n📋 Normal Strategy Distribution:")
            for strategy_id in sorted(strategy_counter.keys()):
                count = strategy_counter[strategy_id]
                percentage = count / total_normal * 100 if total_normal > 0 else 0
                name = strategy_names.get(strategy_id, f"Strategy{strategy_id}")
                print(f"  Strategy{strategy_id:2d} ({name:8s}): {count:6d} times ({percentage:5.2f}%)")
            
            if abnormal_strategies:
                print(f"\n⚠️  Abnormal Strategy IDs (possible data issues):")
                for strategy_id in sorted(abnormal_strategies.keys())[:10]:  # Show first 10 only
                    count = abnormal_strategies[strategy_id]
                    print(f"  Strategy {strategy_id}: {count} times")
                if len(abnormal_strategies) > 10:
                    print(f"  ... {len(abnormal_strategies)-10} more abnormal strategies")
                    
        except Exception as e:
            print(f"  Strategy analysis failed: {e}")
    
    @staticmethod
    def _check_data_quality(data_file, sample_size):
        """Check data quality"""
        print(f"\n🔍 Data Quality Check:")
        
        try:
            data = np.load(data_file)
            quality_issues = []
            valid_count = 0
            
            for i in range(min(sample_size, len(data))):
                sudoku_data = data[i]
                
                if len(sudoku_data) != 325:
                    quality_issues.append(f"{i}: Data length{len(sudoku_data)} != 325")
                    continue
                
                filled_count = sudoku_data[0]
                actual_given = 0
                has_error = False
                
                for j in range(81):
                    idx = 1 + j*4
                    if idx + 3 >= len(sudoku_data):
                        has_error = True
                        break
                        
                    row, col, val, strategy = sudoku_data[idx:idx+4]
                    
                    if not (0 <= row <= 8 and 0 <= col <= 8):
                        quality_issues.append(f"{i}: ({row},{col})")
                        has_error = True
                    
                    if not (1 <= val <= 9):
                        quality_issues.append(f"{i}: {val}1-9")
                        has_error = True
                    
                    if strategy == 0:
                        actual_given += 1
                
                if not has_error and filled_count != actual_given:
                    quality_issues.append(f"{i}: filled_count({filled_count}) != ({actual_given})")
                
                if not has_error:
                    valid_count += 1
            
            # Output quality report
            error_count = len(quality_issues)
            print(f"  Checked Sudoku count: {min(sample_size, len(data))}")
            print(f"  Valid Sudoku count: {valid_count}")
            print(f"  Problem Sudoku count: {min(sample_size, len(data)) - valid_count}")
            print(f"  Data integrity: {valid_count/min(sample_size, len(data))*100:.1f}%")
            
            if quality_issues:
                print(f"\n⚠️  Issues Found (showing first 5):")
                for issue in quality_issues[:5]:
                    print(f"    {issue}")
                if len(quality_issues) > 5:
                    print(f"    ... {len(quality_issues)-5} more issues")
            else:
                print(f"  ✅ No data quality issues found")
                
        except Exception as e:
            print(f"  Data quality check failed: {e}")

class SudokuDemo:
    """Sudoku demonstration tool"""
    
    @staticmethod
    def demo_single_solve(data_file='data/sudoku-test-data.npy', index=10):
        """Demonstrate single Sudoku solving"""
        print("=" * 50)
        print("🎯 Single Sudoku Solving Demo")
        print("=" * 50)
        
        # Read Sudoku
        puzzle, ground_truth = SudokuLoader.read_sudoku(data_file, index)
        
        print(f"📋 Original Puzzle (index {index}):")
        SudokuDemo._print_sudoku(puzzle)
        
        # Solve
        solver = SudokuSolver()
        start_time = time.time()
        solution = solver.solve(puzzle)
        solve_time = time.time() - start_time
        
        print(f"\n⏱️  Solving time: {solve_time*1000:.2f}ms")
        print(f"📊 Solving statistics: {solver.get_stats()}")
        
        if solution is not None:
            print(f"\n✅ Solution:")
            SudokuDemo._print_sudoku(solution)
        else:
            print(f"\n❌ No solution!")
            return
        
        # Evaluate
        eval_result = SudokuEvaluator.evaluate_single(puzzle, solution, ground_truth)
        print(f"\n🔍 Evaluation Result:")
        for key, value in eval_result.items():
            emoji = "✅" if value else "❌"
            print(f"  {emoji} {key}: {value}")
        
        print(f"\n🎯 Ground Truth:")
        SudokuDemo._print_sudoku(ground_truth)
    
    @staticmethod
    def demo_batch_performance(data_file='data/sudoku-test-data.npy', num_samples=100):
        """Demonstrate batch performance evaluation"""
        print("\n" + "=" * 50)
        print(f"⚡ Batch Solving Performance Evaluation ({num_samples} puzzles)")
        print("=" * 50)
        
        # Read multiple Sudoku for testing
        puzzles, ground_truths = SudokuLoader.read_batch_sudoku(data_file, num_samples)
        print(f"📦 Loaded {len(puzzles)} Sudoku puzzles")
        
        # Batch solve and evaluate
        solver = SudokuSolver()
        results = SudokuEvaluator.batch_evaluate(solver, puzzles, ground_truths)
        
        # Display results
        print(f"\n📊 Performance Report:")
        print(f"  Total Sudoku count: {results['total']}")
        print(f"  Solve success rate: {results['solved']}/{results['total']} ({results['solved']/results['total']*100:.1f}%)")
        print(f"  Completely correct rate: {results['correct']}/{results['total']} ({results['accuracy']:.1f}%)")
        print(f"  Valid solutions: {results['valid']}")
        print(f"  Puzzle consistent: {results['consistent']}")
        print(f"  Matches ground truth: {results['matches_gt']}")
        print(f"  Average algorithm steps: {results['avg_steps']:.1f}")
        print(f"  Average backtracks: {results['avg_backtracks']:.1f}")
        print(f"  Average solve time: {results['avg_time']*1000:.2f}ms")
        print(f"  Total solve time: {sum(results['solve_times']):.2f}s")
        
        # Performance distribution analysis
        SudokuDemo._analyze_performance_distribution(results['individual_stats'])
    
    @staticmethod
    def _analyze_performance_distribution(individual_stats):
        """Analyze performance distribution"""
        print(f"\n📈 Performance Distribution Analysis:")
        
        # Group by difficulty
        easy_stats = [s for s in individual_stats if s['empty_cells'] <= 55]
        medium_stats = [s for s in individual_stats if 55 < s['empty_cells'] <= 58]
        hard_stats = [s for s in individual_stats if s['empty_cells'] > 58]
        
        for name, stats in [("Easy", easy_stats), ("Medium", medium_stats), ("Hard", hard_stats)]:
            if not stats: continue
            
            avg_backtracks = np.mean([s['backtracks'] for s in stats])
            avg_time = np.mean([s['solve_time'] for s in stats]) * 1000
            success_rate = np.mean([s['correct'] for s in stats]) * 100
            
            print(f"  {name} puzzles ({len(stats)}): avg backtracks {avg_backtracks:.1f}, "
                  f"avg time {avg_time:.1f}ms, success rate {success_rate:.1f}%")
    
    @staticmethod
    def demo_quick_examples():
        """Quick usage example collection"""
        print("\n" + "=" * 50)
        print("📖 Quick Usage Examples")
        print("=" * 50)
        
        data_file = 'data/sudoku-tiny-data.npy'
        
        try:
            # Basic usage example
            print("\n🔹 Basic Usage Example:")
            puzzle, solution = SudokuLoader.read_sudoku(data_file, 0)
            print(f"  ✓ Read Sudoku: {puzzle.sum()} known digits, {(puzzle==0).sum()} empty cells")
            
            solver = SudokuSolver()
            solved = solver.solve(puzzle)
            stats = solver.get_stats()
            print(f"  ✓ Solve statistics: {stats['steps']} steps, {stats['backtracks']} backtracks")
            
            is_correct = SudokuEvaluator.matches_ground_truth(solved, solution)
            print(f"  ✓ Verification result: {'Correct' if is_correct else 'Incorrect'}")
            
            # Batch processing example
            print("\n🔹 Batch Processing Example:")
            puzzles, solutions = SudokuLoader.read_batch_sudoku(data_file, 10)
            print(f"  ✓ Batch read: {len(puzzles)} Sudoku puzzles")
            
            solver = SudokuSolver()
            results = SudokuEvaluator.batch_evaluate(solver, puzzles, solutions, verbose=False)
            print(f"  ✓ Batch solve: success rate {results['accuracy']:.1f}%, "
                  f"avg {results['avg_backtracks']:.1f} backtracks")
            
            # Statistical information example
            print("\n🔹 Statistical Information Example:")
            info = SudokuLoader.get_dataset_info(data_file)
            print(f"  ✓ Dataset info: {info['total_count']:,} Sudoku puzzles, {info['data_size_mb']:.1f}MB")
            
            puzzles, _ = SudokuLoader.read_batch_sudoku(data_file, 100)
            difficulty_levels = [(puzzle==0).sum() for puzzle in puzzles]
            avg_difficulty = sum(difficulty_levels) / len(difficulty_levels)
            print(f"  ✓ Average difficulty: {avg_difficulty:.1f} empty cells")
            
        except Exception as e:
            print(f"  ❌ Example error: {e}")
            print(f"  💡 Hint: Make sure data file exists at {data_file}")
    
    @staticmethod
    def _print_sudoku(grid):
        """Pretty print Sudoku"""
        for i in range(9):
            if i % 3 == 0 and i > 0:
                print("    ------+-------+------")
            row_str = "    "
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                cell_val = "_" if grid[i, j] == 0 else str(grid[i, j])
                row_str += cell_val + " "
            print(row_str)

def main():
    """Main function: Run complete Sudoku analysis and demonstration"""
    
    print("🧩 Sudoku Loader and Analysis Tool")
    print("=" * 60)
    data_file = 'data/sudoku-tiny-data.npy'
    
    try:
        # 0. Quick usage examples (new)
        SudokuDemo.demo_quick_examples()
        
        # 1. Dataset statistical analysis
        SudokuStatistics.analyze_dataset_statistics(data_file, max_samples=1000)
        
        # 2. Single solving demonstration  
        SudokuDemo.demo_single_solve(data_file, index=0)
        
        # 3. Batch performance evaluation
        SudokuDemo.demo_batch_performance(data_file, num_samples=50)
        
        print(f"\n🎉 Analysis complete!")
        print(f"\n💡 More features:")
        print(f"  - SudokuLoader: Data loading and basic information retrieval")
        print(f"  - SudokuEvaluator: Solution verification and batch evaluation")
        print(f"  - SudokuStatistics: Detailed statistical analysis")
        print(f"  - SudokuDemo: Demonstrations and example code")
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")

if __name__ == "__main__":
    main()