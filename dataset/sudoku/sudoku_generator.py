#!/usr/bin/env python3
"""
🎯 APMDM Sudoku Training Data Generator - Complete Modified Version
- 4-token format: [digit, color, special flag, separator] (maintains visualization compatibility)
- Complete implementation of all operations
- Modified backtrack and skull_to_normal operations
"""

import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BranchInfo:
    """Branch information"""
    id: int
    color: int
    start_cell: Tuple[int, int]
    cells: set


class APMDMSampleGenerator:
    """Complete APMDM sample generator - 3-token format"""
    
    def __init__(self):
        # Token definitions (32 total vocab tokens) - Extended to support 15 branch colors
        self.EMPTY_TOKEN = 0
        self.MASK_TOKEN = 10
        self.WHITE_TOKEN = 11
        # Color tokens: 12-26 (support 15 branch colors)
        self.NORMAL_TOKEN = 27  # Moved from 24 to 27 to avoid color token conflicts
        self.SKULL_TOKEN = 28   # Moved from 25 to 28
        self.BRANCH_TOKEN = 29  # Moved from 26 to 29
        self.SEPARATOR_TOKEN = 30  # Moved from 27 to 30
        self.EOS_TOKEN = 31     # Moved from 28 to 31
        self.VOCAB_SIZE = 32    # Increased from 29 to 32
        
        # State
        self.samples: List[Dict[str, Any]] = []
        self.sample_counter = 0
        self.instance_id = 0  # 🔥 New: Sudoku puzzle ID
        self.current_grid = None
        self.color_grid = None
        self.branch_stack: List[BranchInfo] = []
        self.failed_branch_starts = set()
        self.branch_color_counter = 0
        
        # Backtrack state tracking
        self.skull_positions: Set[Tuple[int, int]] = set()
        
        # Sequence state maintenance
        self.current_sequence_state: List[int] = []
        self.last_failed_value = None
        
        # 🔥 New: Contradiction position tracking
        self.contradiction_positions: Set[Tuple[int, int]] = set()
    
    def reset(self):
        """Reset generator state"""
        self.samples.clear()
        self.sample_counter = 0
        self.branch_stack.clear()
        self.failed_branch_starts.clear()
        self.branch_color_counter = 0
        self.skull_positions.clear()
        self.current_sequence_state.clear()
        self.last_failed_value = None
        # 🔥 New: Initialize contradiction position tracking
        self.contradiction_positions = set()
    
    def _update_sequence_state(self, x_k_plus_1: List[int]):
        """Update current sequence state - ensure continuity"""
        self.current_sequence_state = x_k_plus_1.copy()
    
    def _get_color_token(self, color_id: int) -> int:
        """Get color token"""
        return self.WHITE_TOKEN if color_id == 0 else (self.WHITE_TOKEN + color_id)
    
    def _get_branch_color(self, branch_id: int) -> int:
        """Get branch color: calculate fixed color index based on branch_id"""
        palette_size = 16  # Increased to 16, support 15 branch colors (1-15), sufficient for max branch depth 14
        return (branch_id - 1) % (palette_size - 1) + 1
    
    def _get_cell_index(self, row: int, col: int) -> int:
        """Get cell start position in 324-token sequence - maintain 4-token format for visualization compatibility"""
        return (row * 9 + col) * 4
    
    def _encode_current_state_from_grid(self, grid: np.ndarray) -> List[int]:
        """Encode from given grid to 324-token sequence - maintain 4-token format for visualization compatibility"""
        sequence = []
        
        for r in range(9):
            for c in range(9):
                value = grid[r, c]
                color = self.color_grid[r, c] if hasattr(self, 'color_grid') and self.color_grid is not None else 0
                
                # Determine marker token
                marker = self.NORMAL_TOKEN
                if hasattr(self, 'failed_branch_starts') and (r, c) in self.failed_branch_starts:
                    marker = self.SKULL_TOKEN
                elif hasattr(self, 'branch_stack') and any(b.start_cell == (r, c) for b in self.branch_stack):
                    marker = self.BRANCH_TOKEN
                
                # Encode cell: [value, color, marker, separator] - maintain 4-token format
                value_token = value if value != 0 else self.EMPTY_TOKEN
                color_token = self._get_color_token(color)
                
                sequence.extend([value_token, color_token, marker, self.SEPARATOR_TOKEN])
        
        return sequence
    
    def _create_sample(self, x_k: List[int], y_star: List[int], r_star: List[int],
                      e_star: List[int], c_star: List[int], x_k_plus_1: List[int],
                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create APMDM training sample"""
        self.sample_counter += 1
        
        # Verify length consistency (except expansion/contraction operations)
        gdlm_op = metadata.get('gdlm_operation', '')
        if gdlm_op not in ['expansion', 'contraction']:
            expected_len = len(x_k)
            if not (len(y_star) == len(r_star) == len(e_star) == len(c_star) == expected_len):
                print(f"⚠️  Length inconsistency: x_k={len(x_k)}, y_star={len(y_star)}, r_star={len(r_star)}, e_star={len(e_star)}, c_star={len(c_star)}")
        
        return {
            "x_k": np.array(x_k, dtype=np.uint8),
            "y_star": np.array(y_star, dtype=np.uint8),
            "r_star": np.array(r_star, dtype=np.uint8),
            "e_star": np.array(e_star, dtype=np.uint8),
            "c_star": np.array(c_star, dtype=np.uint8),
            "x_k_plus_1": np.array(x_k_plus_1, dtype=np.uint8),
            "solver_metadata": {
                "sample_id": np.uint32(self.sample_counter),
                "instance_id": np.uint32(self.instance_id),  # 🔥 New: Sudoku puzzle ID
                "sequence_length": np.uint16(len(x_k)),
                "branch_id": np.uint8(metadata.get("branch_id", 0)),
                **{k: v for k, v in metadata.items() if k != "branch_id"}
            }
        }
    
    def _generate_assign_samples(self, row: int, col: int, value: int) -> None:
        """Generate APMDM samples for assign operation - 🔥 Modified: also remask special markers"""
        x_k = self.current_sequence_state.copy()
        cell_start = self._get_cell_index(row, col)
        
        # Determine target color
        target_color = self.branch_stack[-1].color if self.branch_stack else 0
        target_color_token = self._get_color_token(target_color)
        
        # 2-step operation: remask + unmask
        # Step 1: 🔥 Modified: Remask [value, color, marker] - now also remask special marker
        r_star1 = [0] * len(x_k)
        r_star1[cell_start] = 1      # remask value
        r_star1[cell_start + 1] = 1  # remask color  
        r_star1[cell_start + 2] = 1  # 🔥 New: remask marker
        
        x_k_plus_1_1 = x_k.copy()
        x_k_plus_1_1[cell_start] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 1] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN  # 🔥 New: marker also becomes MASK
        
        y_star1 = x_k.copy()
        # For assign_remask, if there's MASK in x_k, should fill with target value
        if x_k[cell_start] == self.MASK_TOKEN:
            y_star1[cell_start] = value
        if x_k[cell_start + 1] == self.MASK_TOKEN:
            y_star1[cell_start + 1] = target_color_token
        if x_k[cell_start + 2] == self.MASK_TOKEN:
            y_star1[cell_start + 2] = self.NORMAL_TOKEN
        
        self.samples.append(self._create_sample(
            x_k, y_star1, r_star1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "assign_remask", "gdlm_operation": "remask"}
        ))
        
        # Step 2: 🔥 Modified: Unmask to new value - now also unmask marker to NORMAL
        y_star2 = x_k_plus_1_1.copy()
        y_star2[cell_start] = value
        y_star2[cell_start + 1] = target_color_token
        y_star2[cell_start + 2] = self.NORMAL_TOKEN  # 🔥 New: unmask marker to NORMAL
        
        x_k_plus_1_2 = x_k_plus_1_1.copy()
        x_k_plus_1_2[cell_start] = value
        x_k_plus_1_2[cell_start + 1] = target_color_token
        x_k_plus_1_2[cell_start + 2] = self.NORMAL_TOKEN  # 🔥 New: set marker to NORMAL
        
        self.samples.append(self._create_sample(
            x_k_plus_1_1, y_star2, [0]*len(x_k_plus_1_1), [0]*len(x_k_plus_1_1), [0]*len(x_k_plus_1_1), x_k_plus_1_2,
            {"operation": "assign_unmask", "gdlm_operation": "unmask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
    
    def _generate_branch_samples(self, row: int, col: int, value: int, branch_id: int) -> None:
        """Generate APMDM samples for branch operation - Complete implementation"""
        x_k = self.current_sequence_state.copy()
        cell_start = self._get_cell_index(row, col)
        
        # Use fixed color based on branch_id
        branch_color = self._get_branch_color(branch_id)
        branch_color_token = self._get_color_token(branch_color)
        
        # Step 1: Remask [value, color, state]
        r_star1 = [0] * len(x_k)
        r_star1[cell_start] = 1
        r_star1[cell_start + 1] = 1
        r_star1[cell_start + 2] = 1  # State token also needs remask
        
        x_k_plus_1_1 = x_k.copy()
        x_k_plus_1_1[cell_start] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 1] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN
        
        y_star1 = x_k.copy()
        # For branch_remask, if there's MASK in x_k, should fill with target value
        if x_k[cell_start] == self.MASK_TOKEN:
            y_star1[cell_start] = value
        if x_k[cell_start + 1] == self.MASK_TOKEN:
            y_star1[cell_start + 1] = branch_color_token
        if x_k[cell_start + 2] == self.MASK_TOKEN:
            y_star1[cell_start + 2] = self.BRANCH_TOKEN
        
        self.samples.append(self._create_sample(
            x_k, y_star1, r_star1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "branch_remask", "branch_id": branch_id, "gdlm_operation": "remask"}
        ))
        
        # Step 2: Unmask to branch state
        y_star2 = x_k_plus_1_1.copy()
        y_star2[cell_start] = value
        y_star2[cell_start + 1] = branch_color_token
        y_star2[cell_start + 2] = self.BRANCH_TOKEN
        
        x_k_plus_1_2 = x_k_plus_1_1.copy()
        x_k_plus_1_2[cell_start] = value
        x_k_plus_1_2[cell_start + 1] = branch_color_token
        x_k_plus_1_2[cell_start + 2] = self.BRANCH_TOKEN
        
        self.samples.append(self._create_sample(
            x_k_plus_1_1, y_star2, [0]*len(x_k_plus_1_1), [0]*len(x_k_plus_1_1), [0]*len(x_k_plus_1_1), x_k_plus_1_2,
            {"operation": "branch_unmask", "branch_id": branch_id, "gdlm_operation": "unmask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
    
    def _generate_modified_combined_backtrack(self, x_k: List[int], ordinary_cells: List[Tuple[int, int]], 
                                            failed_pos: Tuple[int, int], branch_id: int, contradiction_pos: Tuple[int, int] = None):
        """🔥 Modified version: new behavior for backtrack operation - 4-token format for visualization compatibility"""
        
        # ==== Step 1: Modified Remask operation ====
        r_star_1 = [0] * len(x_k)
        x_k_plus_1_1 = x_k.copy()
        
        # 🔥 Modified: Handle ordinary cells: now remask all three tokens
        for row, col in ordinary_cells:
            cell_start = self._get_cell_index(row, col)
            r_star_1[cell_start] = 1      # remask value token
            r_star_1[cell_start + 1] = 1  # remask color token
            r_star_1[cell_start + 2] = 1  # 🔥 New: remask marker token
            x_k_plus_1_1[cell_start] = self.MASK_TOKEN
            x_k_plus_1_1[cell_start + 1] = self.MASK_TOKEN
            x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN  # 🔥 New: marker also becomes MASK
        
        # 🔥 Modified: Handle failed point (branch start): only remask special marker token
        if failed_pos:
            row, col = failed_pos
            cell_start = self._get_cell_index(row, col)
            # 🔥 New behavior: only remask special marker token (position+2), keep digit and color
            r_star_1[cell_start + 2] = 1  # Only remask marker token
            x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN
        
        # 🔥 New: Handle contradiction position: remask all three tokens
        if contradiction_pos:
            row, col = contradiction_pos
            cell_start = self._get_cell_index(row, col)
            
            r_star_1[cell_start] = 1      # remask value
            r_star_1[cell_start + 1] = 1  # remask color  
            r_star_1[cell_start + 2] = 1  # remask marker
            x_k_plus_1_1[cell_start] = self.MASK_TOKEN
            x_k_plus_1_1[cell_start + 1] = self.MASK_TOKEN
            x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN
        
        y_star_1 = x_k.copy()
        
        # Create step 1 sample (modified remask)
        self.samples.append(self._create_sample(
            x_k, y_star_1, r_star_1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "backtrack_modified_remask", "branch_id": branch_id, "gdlm_operation": "remask"}
        ))
        
        # ==== Step 2: Modified operation (simultaneously remask and unmask) ====
        x_k_2 = x_k_plus_1_1.copy()
        y_star_2 = x_k_2.copy()  # Copy current state at start
        r_star_2 = [0] * len(x_k_2)     # Remask labels
        x_k_plus_1_2 = x_k_2.copy()
        
        # 🔥 Modified: Handle ordinary cells: now unmask to EMPTY+WHITE+NORMAL
        for row, col in ordinary_cells:
            cell_start = self._get_cell_index(row, col)
            y_star_2[cell_start] = self.EMPTY_TOKEN
            y_star_2[cell_start + 1] = self.WHITE_TOKEN
            y_star_2[cell_start + 2] = self.NORMAL_TOKEN  # 🔥 New: unmask marker to NORMAL
            x_k_plus_1_2[cell_start] = self.EMPTY_TOKEN
            x_k_plus_1_2[cell_start + 1] = self.WHITE_TOKEN
            x_k_plus_1_2[cell_start + 2] = self.NORMAL_TOKEN  # 🔥 New: set marker to NORMAL
        
        # 🔥 Modified: Handle failed point (branch start): simultaneously remask first two and unmask third
        if failed_pos:
            row, col = failed_pos
            cell_start = self._get_cell_index(row, col)
            old_value = x_k[cell_start]  # Save original digit
            
            # 🔥 Fix: Correctly set remask and unmask operations
            # Remask first two positions
            r_star_2[cell_start] = 1      # remask value position
            r_star_2[cell_start + 1] = 1  # remask color position
            
            # y_star: provide target value for unmasking third position
            y_star_2[cell_start + 2] = old_value  # unmask third position to original digit
            
            # x_k_plus_1: final result
            x_k_plus_1_2[cell_start] = self.MASK_TOKEN      # remask result
            x_k_plus_1_2[cell_start + 1] = self.MASK_TOKEN  # remask result
            x_k_plus_1_2[cell_start + 2] = old_value        # unmask result
        
        # 🔥 New: Handle contradiction position: unmask to original state [EMPTY, WHITE, NORMAL]
        if contradiction_pos:
            row, col = contradiction_pos
            cell_start = self._get_cell_index(row, col)
            
            # Unmask contradiction position to original state
            y_star_2[cell_start] = self.EMPTY_TOKEN      # unmask to EMPTY
            y_star_2[cell_start + 1] = self.WHITE_TOKEN  # unmask to WHITE
            y_star_2[cell_start + 2] = self.NORMAL_TOKEN # unmask to NORMAL
            
            x_k_plus_1_2[cell_start] = self.EMPTY_TOKEN
            x_k_plus_1_2[cell_start + 1] = self.WHITE_TOKEN
            x_k_plus_1_2[cell_start + 2] = self.NORMAL_TOKEN
        
        # Create step 2 sample (modified: simultaneously remask and unmask)
        self.samples.append(self._create_sample(
            x_k_2, y_star_2, r_star_2, [0]*len(x_k_2), [0]*len(x_k_2), x_k_plus_1_2,
            {"operation": "backtrack_modified_unmask", "branch_id": branch_id, "gdlm_operation": "remask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
        
        # Record SKULL position (if there's a failed point)
        if failed_pos:
            self.skull_positions.add(failed_pos)
    
    def _generate_contradiction_samples(self, row: int, col: int, branch_id: int):
        """🔥 New: Generate marking samples for contradiction position"""
        x_k = self.current_sequence_state.copy()
        cell_start = self._get_cell_index(row, col)
        
        # Get current branch color
        branch_color = self._get_branch_color(branch_id) if branch_id > 0 else 0
        branch_color_token = self._get_color_token(branch_color)
        
        # ==== Step 1: Remask three tokens at contradiction position ====
        r_star_1 = [0] * len(x_k)
        r_star_1[cell_start] = 1      # remask value
        r_star_1[cell_start + 1] = 1  # remask color
        r_star_1[cell_start + 2] = 1  # remask marker
        
        x_k_plus_1_1 = x_k.copy()
        x_k_plus_1_1[cell_start] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 1] = self.MASK_TOKEN
        x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN
        
        y_star_1 = x_k.copy()
        
        self.samples.append(self._create_sample(
            x_k, y_star_1, r_star_1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "contradiction_remask", "branch_id": branch_id, "gdlm_operation": "remask"}
        ))
        
        # ==== Step 2: Unmask to [EMPTY, branch color, SKULL] ====
        x_k_2 = x_k_plus_1_1.copy()
        y_star_2 = x_k_2.copy()
        y_star_2[cell_start] = self.EMPTY_TOKEN       # unmask to EMPTY
        y_star_2[cell_start + 1] = branch_color_token # unmask to branch color
        y_star_2[cell_start + 2] = self.SKULL_TOKEN   # unmask to SKULL
        
        x_k_plus_1_2 = x_k_2.copy()
        x_k_plus_1_2[cell_start] = self.EMPTY_TOKEN
        x_k_plus_1_2[cell_start + 1] = branch_color_token
        x_k_plus_1_2[cell_start + 2] = self.SKULL_TOKEN
        
        self.samples.append(self._create_sample(
            x_k_2, y_star_2, [0]*len(x_k_2), [0]*len(x_k_2), [0]*len(x_k_2), x_k_plus_1_2,
            {"operation": "contradiction_unmask", "branch_id": branch_id, "gdlm_operation": "unmask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
        
        # Record contradiction position for later handling in backtrack
        if not hasattr(self, 'contradiction_positions'):
            self.contradiction_positions = set()
        self.contradiction_positions.add((row, col))
    
    def _generate_backtrack_samples(self, branch_id: int, cleared_positions: List[Tuple[int, int]], failed_pos: Tuple[int, int], contradiction_pos: Tuple[int, int] = None):
        """Generate modified version of backtrack operation"""
        x_k = self.current_sequence_state.copy()
        ordinary_cells = [pos for pos in cleared_positions if pos != failed_pos]
        
        # 🔥 New: If there's a contradiction position, also exclude it from ordinary cells
        if contradiction_pos:
            ordinary_cells = [pos for pos in ordinary_cells if pos != contradiction_pos]
        
        self._generate_modified_combined_backtrack(x_k, ordinary_cells, failed_pos, branch_id, contradiction_pos)
    
    def _generate_skull_to_normal_samples_modified(self, row: int, col: int, value: int):
        """🔥 Modified: SKULL→NORMAL conversion 2-step samples"""
        x_k = self.current_sequence_state.copy()
        cell_start = self._get_cell_index(row, col)
        
        # Determine correct color based on branch stack state
        if self.branch_stack:
            target_color = self.branch_stack[-1].color
            target_color_token = self._get_color_token(target_color)
        else:
            target_color_token = self.WHITE_TOKEN
        
        # Expected current state should be: [MASK, MASK, old_number] (obtained through backtrack)
        old_number = x_k[cell_start + 2]  # Third position should be the original digit
        
        # ==== Step 1: Unmask first two, remask third ====
        # [MASK] [MASK] old_number → new_number, new_color, [MASK]
        
        # 🔥 Fix: Simultaneously unmask and remask operations, y_star cannot contain MASK
        r_star_1 = [0] * len(x_k)
        r_star_1[cell_start + 2] = 1  # remask third position
        
        y_star_1 = x_k.copy()
        y_star_1[cell_start] = value                # unmask first MASK to new digit
        y_star_1[cell_start + 1] = target_color_token  # unmask second MASK to new color
        # 🔥 Fix: y_star cannot contain MASK! For remask position, keep original reasonable value
        y_star_1[cell_start + 2] = old_number  # Keep original digit (this position will be remasked)
        
        x_k_plus_1_1 = x_k.copy()
        x_k_plus_1_1[cell_start] = value
        x_k_plus_1_1[cell_start + 1] = target_color_token
        x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN  # Actual result: remask to MASK
        
        self.samples.append(self._create_sample(
            x_k, y_star_1, r_star_1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "skull_to_normal_step1", "branch_id": 0, "gdlm_operation": "remask"}
        ))
        
        # ==== Step 2: Unmask third to new marker ====
        # new_number, new_color, [MASK] → new_number, new_color, new_symbol
        
        x_k_2 = x_k_plus_1_1.copy()
        y_star_2 = x_k_2.copy()
        y_star_2[cell_start + 2] = self.NORMAL_TOKEN  # unmask third MASK to NORMAL marker
        
        x_k_plus_1_2 = x_k_2.copy()
        x_k_plus_1_2[cell_start + 2] = self.NORMAL_TOKEN
        
        self.samples.append(self._create_sample(
            x_k_2, y_star_2, [0]*len(x_k_2), [0]*len(x_k_2), [0]*len(x_k_2), x_k_plus_1_2,
            {"operation": "skull_to_normal_step2", "branch_id": 0, "gdlm_operation": "unmask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
    
    def _generate_skull_to_branch_samples_modified(self, row: int, col: int, value: int, branch_id: int):
        """🔥 Modified: SKULL→BRANCH conversion 2-step samples"""
        x_k = self.current_sequence_state.copy()
        cell_start = self._get_cell_index(row, col)
        
        # Get fixed color based on branch_id
        branch_color = self._get_branch_color(branch_id) 
        target_color_token = self._get_color_token(branch_color)
        
        # Expected current state should be: [MASK, MASK, old_number] (obtained through backtrack)
        old_number = x_k[cell_start + 2]  # Third position should be the original digit
        
        # ==== Step 1: Unmask first two, remask third ====
        # [MASK] [MASK] old_number → new_number, new_color, [MASK]
        
        r_star_1 = [0] * len(x_k)
        r_star_1[cell_start + 2] = 1  # remask third position
        
        y_star_1 = x_k.copy()
        y_star_1[cell_start] = value                # unmask first MASK to new digit
        y_star_1[cell_start + 1] = target_color_token  # unmask second MASK to new color
        y_star_1[cell_start + 2] = old_number       # keep original digit (this position will be remasked)
        
        x_k_plus_1_1 = x_k.copy()
        x_k_plus_1_1[cell_start] = value
        x_k_plus_1_1[cell_start + 1] = target_color_token
        x_k_plus_1_1[cell_start + 2] = self.MASK_TOKEN
        
        self.samples.append(self._create_sample(
            x_k, y_star_1, r_star_1, [0]*len(x_k), [0]*len(x_k), x_k_plus_1_1,
            {"operation": "skull_to_branch_step1", "branch_id": branch_id, "gdlm_operation": "remask"}
        ))
        
        # ==== Step 2: Unmask third to branch marker ====
        # new_number, new_color, [MASK] → new_number, new_color, BRANCH
        
        x_k_2 = x_k_plus_1_1.copy()
        y_star_2 = x_k_2.copy()
        y_star_2[cell_start + 2] = self.BRANCH_TOKEN  # unmask third MASK to BRANCH marker
        
        x_k_plus_1_2 = x_k_2.copy()
        x_k_plus_1_2[cell_start + 2] = self.BRANCH_TOKEN
        
        self.samples.append(self._create_sample(
            x_k_2, y_star_2, [0]*len(x_k_2), [0]*len(x_k_2), [0]*len(x_k_2), x_k_plus_1_2,
            {"operation": "skull_to_branch_step2", "branch_id": branch_id, "gdlm_operation": "unmask"}
        ))
        
        # Update sequence state
        self._update_sequence_state(x_k_plus_1_2)
    
    # === Observer interface methods ===
    
    def on_start(self, grid: np.ndarray) -> None:
        """Start solving"""
        self.current_grid = grid.copy()
        self.color_grid = np.zeros((9, 9), dtype=int)
        
        # Initialize sequence state - 4-token format, total length 324, compatible with visualization
        x_k_initial = self._encode_current_state_from_grid(grid)
        self.current_sequence_state = x_k_initial.copy()
    
    def on_assign(self, row: int, col: int, value: int, grid: np.ndarray) -> None:
        """Deterministic filling - Complete implementation"""
        self.current_grid = grid.copy()
        self.failed_branch_starts.discard((row, col))
        
        # Check if this is a SKULL→NORMAL conversion
        if (row, col) in self.skull_positions:
            self._generate_skull_to_normal_samples_modified(row, col, value)
            self.skull_positions.remove((row, col))
            return
        
        if self.branch_stack:
            branch = self.branch_stack[-1]
            self.color_grid[row, col] = branch.color
            branch.cells.add((row, col))
            if (row, col) == branch.start_cell:
                return
        
        self._generate_assign_samples(row, col, value)
    
    def on_branch(self, row: int, col: int, candidates: List[int], value: int, branch_id: int, grid: np.ndarray) -> None:
        """Branch creation - Complete implementation"""
        self.current_grid = grid.copy()
        self.failed_branch_starts.discard((row, col))
        
        # Check if this is a SKULL→BRANCH conversion
        if (row, col) in self.skull_positions:
            self._generate_skull_to_branch_samples_modified(row, col, value, branch_id)
            self.skull_positions.remove((row, col))
            # Reuse existing color, don't increment new color counter
            existing_color = next((b.color for b in self.branch_stack if b.start_cell == (row, col)), None)
            if existing_color:
                self.color_grid[row, col] = existing_color
            return
        
        # Normal branch creation
        branch_color = self._get_branch_color(branch_id)
        branch = BranchInfo(
            id=branch_id,
            color=branch_color,
            start_cell=(row, col),
            cells={(row, col)}
        )
        self.branch_stack.append(branch)
        self.color_grid[row, col] = branch.color
        
        self._generate_branch_samples(row, col, value, branch_id)
    
    def on_contradiction(self, row: int, col: int, branch_id: int, grid: np.ndarray) -> None:
        """🔥 New: Handle contradiction position marking"""
        self.current_grid = grid.copy()
        self._generate_contradiction_samples(row, col, branch_id)
    
    def on_batch_backtrack(self, row: int, col: int, cleared_count: int, cleared_positions: List[int], 
                          branch_id: int, grid: np.ndarray, failed_value: int, contradiction_pos: Tuple[int, int] = None) -> None:
        """Batch backtrack - Complete implementation"""
        self.current_grid = grid.copy()
        
        cleared_coords = [(pos // 9, pos % 9) for pos in cleared_positions]
        self.failed_branch_starts.add((row, col))
        self.branch_stack = [b for b in self.branch_stack if b.id != branch_id]
        
        for pos in cleared_positions:
            r, c = pos // 9, pos % 9
            if (r, c) != (row, col):
                self.color_grid[r, c] = 0
        
        self.last_failed_value = failed_value
        self._generate_backtrack_samples(branch_id, cleared_coords, (row, col), contradiction_pos)
    
    def on_end(self, solved: bool, grid: np.ndarray) -> None:
        """End solving - 🔥 Modified: Remove final expand+eos operation"""
        self.current_grid = grid.copy()
        # 🔥 New behavior: end directly, don't add EOS token
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get generated samples"""
        return self.samples.copy()
    
    def create_vocabulary(self) -> Dict[int, str]:
        """Create token vocabulary mapping"""
        vocab = {}
        
        # Basic tokens
        vocab[0] = "[EMPTY]"
        vocab[self.MASK_TOKEN] = "[MASK]"
        vocab[self.WHITE_TOKEN] = "[WHITE]"
        vocab[self.NORMAL_TOKEN] = "[NORMAL]"
        vocab[self.SKULL_TOKEN] = "[SKULL]"
        vocab[self.BRANCH_TOKEN] = "[BRANCH]"
        vocab[self.SEPARATOR_TOKEN] = "[SEP]"
        vocab[self.EOS_TOKEN] = "[EOS]"
        
        # Digit tokens (1-9)
        for i in range(1, 10):
            vocab[i] = f"{i}"
        
        # Color tokens (COLOR_1 to COLOR_15, support 15 branch colors)
        for i in range(1, 16):
            color_token = self.WHITE_TOKEN + i
            vocab[color_token] = f"[COLOR_{i}]"
        
        return vocab


# ================================
# High-level data management
# ================================

class APMDMDataManager:
    """🎯 APMDM Data Manager - Compact version"""
    
    def __init__(self):
        self.generator = APMDMSampleGenerator()
    
    def generate_from_puzzle(self, puzzle: np.ndarray, instance_id: int = 0) -> List[Dict[str, Any]]:
        """Generate APMDM training samples from Sudoku"""
        from sudoku_solver import SudokuSolver
        self.generator.reset()
        self.generator.instance_id = instance_id  # 🔥 New: Set puzzle ID
        solver = SudokuSolver()
        solver.set_observer(self.generator)
        result = solver.solve(puzzle)
        if not (isinstance(result, np.ndarray) and np.all(result > 0)):
            raise RuntimeError("Sudoku solving failed")
        return self.generator.get_samples()
    
    def generate_with_initial_for_visualization(self, puzzle: np.ndarray) -> List[Dict[str, Any]]:
        """Generate samples for visualization"""
        return self.generate_from_puzzle(puzzle)
    
    def save_samples(self, samples: List[Dict[str, Any]], output_file: str, format='pkl', compress=True) -> None:
        """Save samples to specified format"""
        import pickle, gzip, json
        
        if format.lower() == 'pkl':
            # PKL format: use gzip compression
            data = {'samples': samples, 'total_samples': len(samples), 'format_version': '3.0'}
            
            if compress or output_file.endswith('.gz'):
                with gzip.open(output_file, 'wb', compresslevel=6) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            file_size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"💾 Saved {len(samples)} samples: {output_file} ({file_size_mb:.1f}MB)")
            
        elif format.lower() == 'jsonl':
            # JSONL format
            def convert_numpy(obj):
                # Handle numpy scalars
                if hasattr(obj, 'item') and hasattr(obj, 'shape') and obj.shape == ():
                    return obj.item()
                # Handle numpy arrays
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                # Handle dictionaries
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                # Handle lists
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                # Handle numpy integer types
                elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
                    return int(obj)
                # Handle numpy float types
                elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
                    return float(obj)
                else:
                    return obj
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    json.dump(convert_numpy(sample), f, ensure_ascii=False)
                    f.write('\n')
            
            file_size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"💾 Saved {len(samples)} samples: {output_file} ({file_size_mb:.1f}MB)")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_samples(self, input_file: str) -> List[Dict[str, Any]]:
        """Load sample data"""
        import pickle, gzip, json
        
        if input_file.endswith('.pkl') or input_file.endswith('.pkl.gz'):
            # PKL format
            try:
                opener = gzip.open if input_file.endswith('.gz') else open
                mode = 'rb'
                with opener(input_file, mode) as f:
                    data = pickle.load(f)
                return data['samples'] if isinstance(data, dict) and 'samples' in data else data
            except:
                # Try another format
                opener = open if input_file.endswith('.gz') else gzip.open
                with opener(input_file, 'rb') as f:
                    data = pickle.load(f)
                return data['samples'] if isinstance(data, dict) and 'samples' in data else data
        else:
            # JSONL format
            samples = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
    
    def generate_complete_dataset(self, data_file: str, output_file: str = "sudoku.pkl.gz") -> Dict[str, Any]:
        """Generate complete dataset"""
        try:
            from sudoku_loader import SudokuLoader
            if not Path(data_file).exists():
                raise FileNotFoundError(f"File not found: {data_file}")
            
            print(f"🚀 Generating: {data_file} → {output_file}")
            data = np.load(data_file)
            total = data.shape[0]
            
            # Generate vocabulary
            if not Path('vocab_cache.pkl').exists():
                vocab = self.generator.create_vocabulary()
                import pickle
                with open('vocab_cache.pkl', 'wb') as f:
                    pickle.dump(vocab, f)
                print(f"📚 Vocabulary: {len(vocab)} tokens")
            
            all_samples, success, instance_sample_counts = [], 0, []
            for i in range(total):
                try:
                    if i % 50 == 0:
                        print(f"📊 {i+1}/{total}")
                    puzzle, _ = SudokuLoader.read_sudoku(data_file, i)
                    samples = self.generate_from_puzzle(puzzle, instance_id=i)  # 🔥 Pass instance_id
                    all_samples.extend(samples)
                    instance_sample_counts.append(len(samples))
                    success += 1
                except Exception as e:
                    instance_sample_counts.append(0)
                    if i % 100 == 0:
                        print(f"⚠️  Puzzle{i}: {str(e)}")
            
            self.save_samples(all_samples, output_file, format='pkl')
            
            # 🔥 Print sample count for each puzzle
            if len(instance_sample_counts) <= 100:  # Only print when puzzle count is not too large
                print(f"\n📊 Sample count per puzzle:")
                for i, count in enumerate(instance_sample_counts):
                    if i % 10 == 0:
                        print()
                    print(f"{i}:{count}", end=" ")
                print()
            
            return {
                'success': True, 'total_puzzles': total, 'successful_puzzles': success,
                'total_samples': len(all_samples), 'output_file': output_file,
                'instance_sample_counts': instance_sample_counts
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    

    
    def generate_from_file_with_format(self, data_file: str, index: int, output_file: str, format: str = 'jsonl') -> Dict[str, Any]:
        """🔥 New: Generate data in specified format from file (serve.py compatibility method)"""
        try:
            from sudoku_loader import SudokuLoader
            
            if not Path(data_file).exists():
                raise FileNotFoundError(f"File not found: {data_file}")
            
            print(f"🎯 Generating visualization data: {data_file}[{index}] → {output_file} ({format})")
            
            # Read Sudoku puzzle at specified index
            puzzle, _ = SudokuLoader.read_sudoku(data_file, index)
            
            # Generate samples including initial state (for visualization)
            samples = self.generate_with_initial_for_visualization(puzzle)
            
            # Save to specified format
            self.save_samples(samples, output_file, format=format, compress=False)
            
            return {
                'success': True,
                'sample_count': len(samples),
                'output_file': output_file,
                'message': f'Successfully generated {len(samples)} samples'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Generation failed: {str(e)}'
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🎯 APMDM Sudoku Dataset Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sudoku_generator.py sudoku-100.npy
  python sudoku_generator.py sudoku-100.npy sudoku-test.npy
  python sudoku_generator.py sudoku-*.npy
  python sudoku_generator.py sudoku-100.npy --format jsonl
        """
    )
    parser.add_argument('inputs', nargs='*', help='Input NPY file(s) (e.g., sudoku-100.npy sudoku-test.npy)')
    parser.add_argument('--format', choices=['pkl', 'jsonl'], default='pkl', help='Output format (default: pkl)')
    
    args = parser.parse_args()
    
    print("🎯 APMDM Sudoku Dataset Generator")
    print("=" * 50)
    
    manager = APMDMDataManager()
    
    # No input files
    if not args.inputs:
        parser.print_help()
        return
    
    # Process each input file
    for input_file in args.inputs:
        # Auto-generate output filename
        base_name = Path(input_file).stem
        ext = '.pkl.gz' if args.format == 'pkl' else '.jsonl'
        output_file = base_name + ext
        
        # Check input file exists
        if not Path(input_file).exists():
            print(f"❌ Skipping: {input_file} (file not found)")
            continue
        
        # Generate dataset
        print(f"\n📊 Processing: {input_file}")
        print(f"📦 Output: {output_file}")
        
        res = manager.generate_complete_dataset(input_file, output_file)
        
        if res['success']:
            print(f"✅ Success: {res['successful_puzzles']}/{res['total_puzzles']} puzzles, {res['total_samples']:,} samples")
        else:
            print(f"❌ Failed: {res['error']}")
    
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
