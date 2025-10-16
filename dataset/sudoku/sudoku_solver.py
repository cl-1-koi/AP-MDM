#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Sudoku Solver - Complete State Management System
Optimized for APMDM data generation, supports detailed state tracking and animation generation
"""

import numpy as np
from typing import Optional, Any, Tuple, List, Set
from dataclasses import dataclass, field
from enum import Enum

# 🔧 Configuration constants
GRID_SIZE = 9
BOX_SIZE = 3

class FillSource(Enum):
    """🏷️ Fill source type"""
    INITIAL = "initial"        # Initial puzzle given
    DETERMINISTIC = "det"      # Deterministic fill (only 1 candidate)
    BRANCH = "branch"          # Branch decision fill

@dataclass
class CellState:
    """📚 Complete cell state data structure"""
    value: int = 0                           # Current value: 0=empty, 1-9=filled
    candidates: List[int] = field(default_factory=list)  # Candidate list (sorted from small to large)
    failed_attempts: Set[int] = field(default_factory=set)  # Failure history
    fill_source: FillSource = FillSource.INITIAL  # Fill source
    branch_id: int = 0                       # Belongs to branch ID (0=non-branch)
    fill_step: int = 0                       # Step number when filled
    last_updated: int = 0                    # Last update step
    
    def is_empty(self) -> bool:
        """Empty cell check"""
        return self.value == 0
    
    def add_failed_attempt(self, value: int, step: int):
        """Record failed attempt"""
        self.failed_attempts.add(value)
        self.last_updated = step
    
    def set_value(self, value: int, source: FillSource, branch_id: int, step: int):
        """Set value and update state"""
        self.value = value
        self.fill_source = source
        self.branch_id = branch_id
        self.fill_step = step
        self.last_updated = step
        self.candidates.clear()  # Filled, clear candidates
    
    def clear_value(self, step: int):
        """Clear value (during backtrack)"""
        self.value = 0
        self.fill_source = FillSource.INITIAL
        self.branch_id = 0
        self.last_updated = step
        # Keep failure history, don't clear candidates (will be recalculated)

@dataclass
class BranchFrame:
    """📚 Branch stack frame data structure"""
    start_cell: Tuple[int, int]          # Branch start position (row, col)
    remaining_candidates: List[int]      # Untried candidates [3,5,7...] (sorted)
    branch_id: int                       # Branch depth ID (1,2,3...)
    filled_cells: List[Tuple[int, int]]  # Cells filled by this branch [(r1,c1),(r2,c2)...]
    state_snapshot: List[List[CellState]]  # Complete state snapshot before branch
    current_tried_value: int             # Current tried digit value
    original_candidates: List[int]       # 🔢 Original complete candidates (for display)

class SudokuSolver:
    """🎯 Sudoku Solver - Complete State Management System"""
    
    def __init__(self):
        self.steps = 0
        self.backtracks = 0
        self._observer: Optional[Any] = None
        
        # 🌟 Enhanced state management
        self.cell_states: List[List[CellState]] = []  # 9x9 CellState matrix
        self.branch_stack: List[BranchFrame] = []
        self.current_branch_cells: List[Tuple[int, int]] = []
        
        # 🚀 Performance optimization cache
        self._candidates_cache: dict = {}     # Candidate cache
        self._cache_valid: bool = False       # Cache validity flag
    
    def _init_cell_states(self, puzzle: np.ndarray):
        """🚀 Initialize complete cell state matrix"""
        self.cell_states = []
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                cell = CellState()
                if puzzle[i, j] != 0:
                    cell.set_value(puzzle[i, j], FillSource.INITIAL, 0, 0)
                row.append(cell)
            self.cell_states.append(row)
        
        # Initialize candidates for all empty cells (ensure sorted)
        self._update_all_candidates()
    
    def _update_all_candidates(self):
        """🔄 Update candidate sets for all empty cells"""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.cell_states[i][j].is_empty():
                    # 🔢 Ensure sorted candidates on each update
                    candidates = self._compute_candidates(i, j)  # Already sorted
                    self.cell_states[i][j].candidates = candidates
                    self.cell_states[i][j].last_updated = self.steps
    
    def _compute_candidates(self, row: int, col: int) -> List[int]:
        """🎯 Compute cell candidates, sorted from small to large, considering failure history"""
        if not self.cell_states[row][col].is_empty():
            return []
        
        candidates = []
        failed = self.cell_states[row][col].failed_attempts
        
        # 🔢 Ensure order from small to large: 1,2,3,4,5,6,7,8,9
        for num in range(1, GRID_SIZE + 1):
            if num not in failed and self._is_valid_enhanced(row, col, num):
                candidates.append(num)
        
        return candidates  # Naturally sorted: [1,2,4,7,9] not random
    
    def _is_valid_enhanced(self, row: int, col: int, num: int) -> bool:
        """✅ Enhanced validity check: based on CellState"""
        # Row check
        for c in range(GRID_SIZE):
            if self.cell_states[row][c].value == num:
                return False
        
        # Column check
        for r in range(GRID_SIZE):
            if self.cell_states[r][col].value == num:
                return False
        
        # Box check
        box_row, box_col = BOX_SIZE * (row // BOX_SIZE), BOX_SIZE * (col // BOX_SIZE)
        for r in range(box_row, box_row + BOX_SIZE):
            for c in range(box_col, box_col + BOX_SIZE):
                if self.cell_states[r][c].value == num:
                    return False
        
        return True
    
    def _find_best_cell_enhanced(self) -> Tuple[Optional[Tuple[int, int]], List[int]]:
        """🔍 Find best empty cell: based on sorted candidates"""
        best_cell, best_candidates, min_count = None, [], 10
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = self.cell_states[i][j]
                if cell.is_empty():
                    # 🔢 Force sort to ensure 1,2,3... order
                    candidates = sorted(cell.candidates)
                    if len(candidates) < min_count:
                        min_count = len(candidates)
                        best_cell, best_candidates = (i, j), candidates
                        if min_count <= 1: 
                            return best_cell, best_candidates
        
        return best_cell, best_candidates
    
    def _fill_and_record_enhanced(self, row: int, col: int, value: int, 
                                source: FillSource, branch_id: int = 0) -> None:
        """✏️ Fill and record: update complete state + recalculate related candidates"""
        # Set cell state
        self.cell_states[row][col].set_value(value, source, branch_id, self.steps)
        self.current_branch_cells.append((row, col))
        
        # 🔄 Update affected candidates: same row, column, box
        self._update_affected_candidates(row, col, value)
        
        # 📡 Notify observer - only non-branch fills call on_assign
        # Branch fills are handled by on_branch in _create_branch_enhanced
        if source != FillSource.BRANCH:
            grid = self._get_current_grid()
            self._notify('on_assign', row, col, value, grid)
    
    def _update_affected_candidates(self, filled_row: int, filled_col: int, filled_value: int):
        """🔄 Update affected cell candidates: same row, column, box"""
        affected_cells = set()
        
        # Same row
        for c in range(GRID_SIZE):
            if c != filled_col:
                affected_cells.add((filled_row, c))
        
        # Same column
        for r in range(GRID_SIZE):
            if r != filled_row:
                affected_cells.add((r, filled_col))
        
        # Same box
        box_row, box_col = BOX_SIZE * (filled_row // BOX_SIZE), BOX_SIZE * (filled_col // BOX_SIZE)
        for r in range(box_row, box_row + BOX_SIZE):
            for c in range(box_col, box_col + BOX_SIZE):
                if (r, c) != (filled_row, filled_col):
                    affected_cells.add((r, c))
        
        # Update candidates (maintain sorting)
        for r, c in affected_cells:
            if self.cell_states[r][c].is_empty():
                # Remove from sorted list
                if filled_value in self.cell_states[r][c].candidates:
                    self.cell_states[r][c].candidates.remove(filled_value)
                self.cell_states[r][c].last_updated = self.steps
    
    def _get_current_grid(self) -> np.ndarray:
        """🔄 Convert from CellState matrix to traditional grid format"""
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                grid[i, j] = self.cell_states[i][j].value
        return grid
    
    def _create_state_snapshot(self) -> List[List[CellState]]:
        """📸 Create complete state snapshot"""
        snapshot = []
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                # Deep copy CellState
                original = self.cell_states[i][j]
                copy_cell = CellState(
                    value=original.value,
                    candidates=original.candidates.copy(),  # List copy maintains order
                    failed_attempts=original.failed_attempts.copy(),
                    fill_source=original.fill_source,
                    branch_id=original.branch_id,
                    fill_step=original.fill_step,
                    last_updated=original.last_updated
                )
                row.append(copy_cell)
            snapshot.append(row)
        return snapshot
    
    def _restore_state_snapshot(self, snapshot: List[List[CellState]]):
        """🔄 Restore complete state snapshot"""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                original = snapshot[i][j]
                self.cell_states[i][j] = CellState(
                    value=original.value,
                    candidates=original.candidates.copy(),  # ListList copy maintains order
                    failed_attempts=original.failed_attempts.copy(),
                    fill_source=original.fill_source,
                    branch_id=original.branch_id,
                    fill_step=original.fill_step,
                    last_updated=original.last_updated
                )
    
    def get_complete_state(self) -> dict:
        """📊 Get complete state information (for debugging and analysis)"""
        state = {
            "grid": self._get_current_grid().tolist(),
            "candidates": {},
            "failed_attempts": {},
            "fill_sources": {},
            "branch_assignments": {},
            "fill_timeline": {}
        }
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = self.cell_states[i][j]
                key = f"({i},{j})"
                
                state["candidates"][key] = list(cell.candidates)
                state["failed_attempts"][key] = list(cell.failed_attempts)
                state["fill_sources"][key] = cell.fill_source.value
                state["branch_assignments"][key] = cell.branch_id
                state["fill_timeline"][key] = cell.fill_step
        
        return state
    
    # Compatibility interface
    def set_observer(self, observer: Optional[Any]) -> None:
        """🔗 Set observer"""
        self._observer = observer
    
    def _notify(self, event: str, *args) -> None:
        """📡 Notify observer, safe call to avoid exception interrupting solving"""
        if self._observer and hasattr(self._observer, event):
            try:
                getattr(self._observer, event)(*args)
            except Exception as e:
                print(f"⚠️ Observer notification error {event}: {e}")
    
    def _create_branch_enhanced(self, cell: Tuple[int, int], candidates: List[int]) -> None:
        """🌱 Create branch: complete state snapshot + failure history management"""
        row, col = cell
        
        # 🔢 Step 1: Force sort candidates to ensure 1,2,3... order
        sorted_candidates = sorted(candidates)
        
        snapshot = self._create_state_snapshot()              # 📸 Complete state snapshot
        branch_id = len(self.branch_stack) + 1                # 🆔 Branch depth ID
        
        if self.branch_stack:                                 # 📚 Transfer current branch record
            self.branch_stack[-1].filled_cells.extend(self.current_branch_cells)
        
        frame = BranchFrame(                                  # 🌳 Create new stack frame
            start_cell=cell,
            remaining_candidates=sorted_candidates[1:],       # 🔢 Remaining candidates (sorted)
            branch_id=branch_id,
            filled_cells=[],
            state_snapshot=snapshot,                          # Complete state snapshot
            current_tried_value=sorted_candidates[0],         # 🔢 Minimum candidate
            original_candidates=sorted_candidates             # 🔢 Save original complete candidates
        )
        self.branch_stack.append(frame)
        
        self.current_branch_cells = []                        # 🔄 Fill first candidate (minimum)
        self._fill_and_record_enhanced(row, col, sorted_candidates[0], FillSource.BRANCH, branch_id)
        
        # 🔢 Pass sorted candidates to trace_generator
        self._notify('on_branch', row, col, sorted_candidates, sorted_candidates[0], branch_id, self._get_current_grid())
    
    def _instant_batch_backtrack_enhanced(self) -> bool:
        """🔥 Batch backtrack: record failure history + restore complete state"""
        if not self.branch_stack:
            return False                                      # 💀 Stack empty, no solution
        
        # 🔥 New: Find contradiction position before backtracking (position with empty candidates)
        contradiction_pos = self._find_contradiction_position()
        if contradiction_pos:
            contradiction_row, contradiction_col = contradiction_pos
            current_branch_id = self.branch_stack[-1].branch_id if self.branch_stack else 0
            # 📡 Notify contradiction position
            self._notify('on_contradiction', contradiction_row, contradiction_col, current_branch_id, self._get_current_grid())
        
        frame = self.branch_stack.pop()                       # 🎯 Directly pop top of stack
        assert frame.remaining_candidates, f"Branch degradation logic guarantee: branch_id={frame.branch_id} has remaining candidates"
        
        # 📋 Collect cleared cells + record failure history
        cleared_total = []
        cleared_total.extend(frame.filled_cells)
        cleared_total.extend(self.current_branch_cells)
        unique_cleared = list(set(cleared_total))
        cleared_positions = [r * 9 + c for r, c in unique_cleared]
        
        # 💀 Record failed attempt for start cell
        start_row, start_col = frame.start_cell
        failed_value = frame.current_tried_value
        self.cell_states[start_row][start_col].add_failed_attempt(failed_value, self.steps)
        
        self.backtracks += 1                                  # 📡 Notify backtrack event
        self._notify('on_batch_backtrack', start_row, start_col, 
                   len(unique_cleared), cleared_positions, frame.branch_id, self._get_current_grid(), failed_value, contradiction_pos)
        
        # 🔄 Restore complete state snapshot
        self._restore_state_snapshot(frame.state_snapshot)
        next_candidate = frame.remaining_candidates.pop(0)
        frame.filled_cells = []
        frame.current_tried_value = next_candidate
        
        if len(frame.remaining_candidates) == 0:              # 🎯 Branch degradation handling
            self.current_branch_cells = []                    # Deterministic fill: don't push to stack
            self._fill_and_record_enhanced(start_row, start_col, next_candidate, 
                                         FillSource.DETERMINISTIC, frame.branch_id)
        else:
            self.branch_stack.append(frame)                   # 🌳 True branch: push to stack
            self.current_branch_cells = []
            self._fill_and_record_enhanced(start_row, start_col, next_candidate, 
                                         FillSource.BRANCH, frame.branch_id)
            # After backtrack retry branch, also need to notify on_branch
            remaining_plus_current = [next_candidate] + frame.remaining_candidates
            self._notify('on_branch', start_row, start_col, remaining_plus_current, 
                        next_candidate, frame.branch_id, self._get_current_grid())
        
        return True
    
    def _find_contradiction_position(self) -> Optional[Tuple[int, int]]:
        """🔥 New: Find contradiction position (empty cell with empty candidates)"""
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = self.cell_states[i][j]
                if cell.is_empty() and len(cell.candidates) == 0:
                    # Found empty cell with empty candidates - this is the contradiction position
                    return (i, j)
        return None
    
    def solve(self, puzzle: np.ndarray) -> Optional[np.ndarray]:
        """🎯 Main solve loop: complete state management"""
        self.steps = self.backtracks = 0                      # 🚀 Initialize
        self._init_cell_states(puzzle)
        self.branch_stack.clear()
        self.current_branch_cells.clear()
        
        self._notify('on_start', self._get_current_grid())
        
        while True:                                           # 🔄 Main loop
            self.steps += 1
            cell, candidates = self._find_best_cell_enhanced()
            
            if cell is None:                                  # 🎉 Solving complete
                if self.branch_stack:
                    self.branch_stack[-1].filled_cells.extend(self.current_branch_cells)
                self._notify('on_end', True, self._get_current_grid())
                return self._get_current_grid()
            
            if not candidates:                                # 💀 Dead end backtrack
                if self.branch_stack:
                    self.branch_stack[-1].filled_cells.extend(self.current_branch_cells)
                
                if not self._instant_batch_backtrack_enhanced():
                    self._notify('on_end', False, self._get_current_grid())
                    return None
                continue
            
            row, col = cell                                   # 🎯 Process candidates
            
            if len(candidates) == 1:
                self._fill_and_record_enhanced(row, col, candidates[0], FillSource.DETERMINISTIC)
            else:
                # 🔢 Force sort candidates to ensure 1,2,3... order
                sorted_candidates = sorted(candidates)
                self._create_branch_enhanced(cell, sorted_candidates)
    
    def get_stats(self) -> dict:
        """📊 Get statistics"""
        return {"steps": self.steps, "backtracks": self.backtracks}


# Compatibility aliases (maintain backward compatibility)
EnhancedSudokuSolver = SudokuSolver
CorrectSudokuSolver = SudokuSolver

# Simplified demonstration version
def demo_solver():
    """🎪 Demonstrate solver functionality"""
    solver = SudokuSolver()
    
    # Example Sudoku (complete 9x9)
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    solver._init_cell_states(puzzle)
    
    print("=== 🌟 Sudoku Solver Demo ===")
    print(f"Cell (0,2) state:")
    cell = solver.cell_states[0][2]
    print(f"  Value: {cell.value}")
    print(f"  Candidates: {cell.candidates}")
    print(f"  Failed attempts: {cell.failed_attempts}")
    print(f"  Source: {cell.fill_source}")
    
    return solver

if __name__ == "__main__":
    demo_solver()
