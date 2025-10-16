#!/usr/bin/env python3
"""
🎨 Token Visual Decoder - Visualization module completely based on token sequences

Design goals:
1. Completely decode all visualization information from x_k and x_k_plus_1 token sequences
2. Completely separated from sample generation module
3. Support visualization reconstruction of arbitrary token sequences
"""

from typing import List, Dict, Any, Tuple, Optional
import json


class TokenVisualDecoder:
    """Token visual decoder - Decode visualization info from 324-token sequence"""
    
    def __init__(self):
        # Token definitions (consistent with generator)
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
        
        # Token mappings
        self.token_vocab = self._build_token_vocab()
        self.color_palette = self._build_color_palette()
    
    def _build_token_vocab(self) -> Dict[str, str]:
        """Build token vocabulary"""
        vocab = {
            '0': '[EMPTY]', '10': '[MASK]', '11': '[WHITE]',
            '27': '[NORMAL]', '28': '[SKULL]', '29': '[BRANCH]', '30': '|',
            '31': '[EOS]'  # 🆕 Update EOS token position
        }
        
        # Digit tokens
        for i in range(1, 10):
            vocab[str(i)] = str(i)
        
        # Color tokens (support 15 branch colors)
        for i in range(12, 27):
            vocab[str(i)] = f'[COLOR_{i-11}]'
        
        return vocab
    
    def _split_tokens_by_separator(self, tokens: List[int]) -> List[List[int]]:
        """🔥 Fix: Dynamically split cells based on SEPARATOR token, keep separator as 4th token of cell"""
        cells = []
        current_cell = []
        
        for token in tokens:
            if token == self.SEPARATOR_TOKEN:
                # Met separator, add it to current cell and complete cell
                current_cell.append(token)
                if current_cell:
                    cells.append(current_cell)
                    current_cell = []
            elif token == self.EOS_TOKEN:
                # Met EOS, end processing
                break
            else:
                # Regular token, add to current cell
                current_cell.append(token)
        
        # Handle last cell (if not ending with separator)
        if current_cell:
            cells.append(current_cell)
        
        return cells
    
    def _build_color_palette(self) -> List[str]:
        """Build color palette"""
        return [
            '#ffffff',  # White (COLOR_0/WHITE)
            '#ffe4e6', '#e0f2fe', '#dcfce7', '#ede9fe', '#fef9c3', '#f1f5f9',
            '#f5d0fe', '#bbf7d0', '#fde68a', '#bae6fd', '#fecaca', '#ddd6fe',
            '#fef3c7', '#c7d2fe', '#bfdbfe', '#fecdd3', '#d9f99d'  # COLOR_1-12
        ]
    
    def decode_token_sequence(self, token_sequence: str) -> Dict[str, Any]:
        """Decode complete visualization info from token sequence string"""
        tokens = [int(t) for t in token_sequence.split()]
        return self.decode_tokens(tokens)
    
    def decode_sample_data(self, sample_data: dict) -> Dict[str, Any]:
        """🔥 New: Decode visualization info from sample data, support multiple formats (numpy array, list, string)"""
        # Check data format
        x_k_plus_1 = sample_data.get('x_k_plus_1', [])
        
        if hasattr(x_k_plus_1, 'tolist'):  # numpy array
            # PKL format: numpy array -> convert to list
            return self.decode_tokens(x_k_plus_1.tolist())
        elif isinstance(x_k_plus_1, list):
            # JSONL format: integer array
            return self.decode_tokens(x_k_plus_1)
        elif isinstance(x_k_plus_1, str):
            # Old format: string
            return self.decode_token_sequence(x_k_plus_1)
        else:
            raise ValueError(f"Unsupported x_k_plus_1 format: {type(x_k_plus_1)}")
    
    @staticmethod
    def sample_to_string_format(sample_data: dict) -> dict:
        """🔥 New: Convert sample data to string format (compatible with old visualization)"""
        sample_copy = sample_data.copy()
        
        # Convert arrays to strings - 🚀 Support numpy arrays and lists
        for key in ['x_k', 'y_star', 'y_star_text', 'x_k_plus_1']:
            if key in sample_copy:
                value = sample_copy[key]
                if hasattr(value, 'tolist'):  # numpy array
                    sample_copy[key] = ' '.join(map(str, value.tolist()))
                elif isinstance(value, list):  # list
                    sample_copy[key] = ' '.join(map(str, value))
        
        return sample_copy
    
    @staticmethod
    def sample_to_array_format(sample_data: dict) -> dict:
        """🔥 New: Convert sample data to array format (for training)"""
        sample_copy = sample_data.copy()
        
        # Convert to integer arrays - 🚀 Support string, numpy array and list
        for key in ['x_k', 'y_star', 'y_star_text', 'x_k_plus_1']:
            if key in sample_copy:
                value = sample_copy[key]
                if isinstance(value, str):  # String format
                    sample_copy[key] = [int(t) for t in value.split()]
                elif hasattr(value, 'tolist'):  # numpy array
                    sample_copy[key] = value.tolist()
                # List format remains unchanged
        
        return sample_copy
    
    def decode_tokens(self, tokens: List[int]) -> Dict[str, Any]:
        """Decode complete visualization info from token list"""
        
        # Basic validation
        if len(tokens) < 324:
            # Handle variable length sequences, pad to 324
            tokens = tokens + [self.SEPARATOR_TOKEN] * (324 - len(tokens))
        
        # Extract three dimensions of information
        grid = self._extract_grid(tokens)
        color_grid = self._extract_color_grid(tokens) 
        marker_grid = self._extract_marker_grid(tokens)
        
        # Analyze branch and marker information
        analysis = self._analyze_grid_state(grid, color_grid, marker_grid)
        
        return {
            'grid': grid,
            'colorGrid': color_grid,
            'markerGrid': marker_grid,
            'activeBranches': analysis['active_branches'],
            'branchStartMarkers': analysis['branch_start_markers'],
            'failedPositions': analysis['failed_positions'],
            'palette': self.color_palette,
            'tokenVocab': self.token_vocab,
            'gridStats': analysis['stats']
        }
    
    def _extract_grid(self, tokens: List[int]) -> List[int]:
        """🔥 Fix: Dynamically extract 9x9 Sudoku grid based on separator"""
        grid = []
        
        # 🔥 Key fix: Split cells based on SEPARATOR token
        cells = self._split_tokens_by_separator(tokens)
        
        for cell_tokens in cells[:81]:  # Only take first 81 cells
            if len(cell_tokens) >= 1:
                value_token = cell_tokens[0]  # First token of each cell is value
                # Convert token to actual value
                if value_token == self.EMPTY_TOKEN:
                    grid.append(0)
                elif value_token == self.MASK_TOKEN:
                    grid.append(0)  # MASK displays as empty
                elif 1 <= value_token <= 9:
                    grid.append(value_token)
                else:
                    grid.append(0)  # Other cases display as empty
            else:
                grid.append(0)  # Empty cell
        
        # Ensure 81 elements (9x9 grid)
        while len(grid) < 81:
            grid.append(0)
        
        return grid[:81]
    
    def _extract_color_grid(self, tokens: List[int]) -> List[int]:
        """🔥 Fix: Dynamically extract 9x9 color grid based on separator"""
        color_grid = []
        
        # 🔥 Key fix: Split cells based on SEPARATOR token
        cells = self._split_tokens_by_separator(tokens)
        
        for cell_tokens in cells[:81]:  # Only take first 81 cells
            if len(cell_tokens) >= 2:
                color_token = cell_tokens[1]  # Second token of each cell is color
                # Convert color token to color ID
                if color_token == self.WHITE_TOKEN:
                    color_grid.append(0)  # White
                elif self.WHITE_TOKEN < color_token < self.NORMAL_TOKEN:
                    color_id = color_token - self.WHITE_TOKEN
                    color_grid.append(color_id)
                else:
                    color_grid.append(0)  # Default white
            else:
                color_grid.append(0)  # Default white
        
        # Ensure 81 elements
        while len(color_grid) < 81:
            color_grid.append(0)
        
        return color_grid[:81]
    
    def _extract_marker_grid(self, tokens: List[int]) -> List[int]:
        """🔥 Fix: Dynamically extract 9x9 marker grid based on separator"""
        marker_grid = []
        
        # 🔥 Key fix: Split cells based on SEPARATOR token
        cells = self._split_tokens_by_separator(tokens)
        
        for cell_tokens in cells[:81]:  # Only take first 81 cells
            if len(cell_tokens) >= 3:
                marker_token = cell_tokens[2]  # Third token of each cell is marker
                if marker_token in [self.NORMAL_TOKEN, self.SKULL_TOKEN, self.BRANCH_TOKEN]:
                    marker_grid.append(marker_token)
                else:
                    marker_grid.append(self.NORMAL_TOKEN)  # Default NORMAL
            else:
                marker_grid.append(self.NORMAL_TOKEN)  # Default NORMAL
        
        # Ensure 81 elements
        while len(marker_grid) < 81:
            marker_grid.append(self.NORMAL_TOKEN)
        
        return marker_grid[:81]
    
    def _analyze_grid_state(self, grid: List[int], color_grid: List[int], 
                           marker_grid: List[int]) -> Dict[str, Any]:
        """Analyze grid state, extract branch and marker information"""
        
        # Find active branches
        active_branches = {}
        branch_start_markers = {}
        failed_positions = []
        
        for i in range(81):
            row, col = i // 9, i % 9
            color_id = color_grid[i]
            marker = marker_grid[i]
            
            # Branch start point (BRANCH marker)
            if marker == self.BRANCH_TOKEN:
                branch_id = color_id  # Assume branch ID = color ID
                key = f"{row}-{col}"
                
                branch_start_markers[key] = {
                    "branchId": branch_id,
                    "status": "active"
                }
                
                # Count information for this branch
                if str(branch_id) not in active_branches:
                    active_branches[str(branch_id)] = {
                        "color": color_id,
                        "start_cell": [row, col],
                        "cell_count": 0
                    }
            
            # Failed position (SKULL marker)
            elif marker == self.SKULL_TOKEN:
                failed_positions.append((row, col))
                key = f"{row}-{col}"
                branch_start_markers[key] = {
                    "branchId": -1,
                    "status": "failed"
                }
            
            # Count cells included in branch
            if color_id > 0 and str(color_id) in active_branches:
                active_branches[str(color_id)]["cell_count"] += 1
        
        # Statistical information
        filled_cells = sum(1 for v in grid if v > 0)
        colored_cells = sum(1 for c in color_grid if c > 0)
        special_markers = sum(1 for m in marker_grid if m != self.NORMAL_TOKEN)
        
        stats = {
            "filled_cells": filled_cells,
            "colored_cells": colored_cells,
            "special_markers": special_markers,
            "active_branch_count": len(active_branches),
            "failed_position_count": len(failed_positions)
        }
        
        return {
            'active_branches': active_branches,
            'branch_start_markers': branch_start_markers,
            'failed_positions': failed_positions,
            'stats': stats
        }
    
    def format_decoded_sequence(self, tokens: List[int]) -> str:
        """Format display of decoded token meanings"""
        if not tokens:
            return ""
        
        formatted = []
        for i, token in enumerate(tokens):
            vocab_name = self.token_vocab.get(str(token), f"UNK_{token}")
            formatted.append(f"{token}:{vocab_name}")
        
        return " ".join(formatted)
    
    def compare_sequences(self, x_k: str, x_k_plus_1: str) -> Dict[str, Any]:
        """Compare differences between two sequences, for debugging"""
        
        tokens_k = [int(t) for t in x_k.split()]
        tokens_k1 = [int(t) for t in x_k_plus_1.split()]
        
        visual_k = self.decode_tokens(tokens_k)
        visual_k1 = self.decode_tokens(tokens_k1)
        
        # Find changed positions
        changes = []
        min_len = min(len(tokens_k), len(tokens_k1))
        
        for i in range(min_len):
            if tokens_k[i] != tokens_k1[i]:
                cell_idx = i // 4
                token_type = ["value", "color", "marker", "separator"][i % 4]
                row, col = cell_idx // 9, cell_idx % 9
                
                changes.append({
                    "position": i,
                    "cell": (row, col),
                    "token_type": token_type,
                    "from": f"{tokens_k[i]}:{self.token_vocab.get(str(tokens_k[i]), 'UNK')}",
                    "to": f"{tokens_k1[i]}:{self.token_vocab.get(str(tokens_k1[i]), 'UNK')}"
                })
        
        # Length change
        length_change = len(tokens_k1) - len(tokens_k)
        
        return {
            'sequence_changes': changes,
            'length_change': length_change,
            'visual_k': visual_k,
            'visual_k1': visual_k1,
            'stats_comparison': {
                'filled_cells': (visual_k['gridStats']['filled_cells'], 
                               visual_k1['gridStats']['filled_cells']),
                'colored_cells': (visual_k['gridStats']['colored_cells'],
                                visual_k1['gridStats']['colored_cells']),
                'active_branches': (visual_k['gridStats']['active_branch_count'],
                                  visual_k1['gridStats']['active_branch_count'])
            }
        }


# ================================
# Convenient interface functions
# ================================

def decode_visual_from_tokens(token_sequence: str) -> Dict[str, Any]:
    """Convenient function: Decode visualization info from token sequence"""
    decoder = TokenVisualDecoder()
    return decoder.decode_token_sequence(token_sequence)


def compare_token_sequences(x_k: str, x_k_plus_1: str) -> Dict[str, Any]:
    """Convenient function: Compare two token sequences"""
    decoder = TokenVisualDecoder()
    return decoder.compare_sequences(x_k, x_k_plus_1)


def main():
    """Demonstrate token decoding functionality"""
    
    print("🎨 Token Visual Decoder Demo")
    print("=" * 40)
    
    # Test from existing samples
    try:
        with open("sudoku_gdlm_with_visual.jsonl", "r") as f:
            lines = f.readlines()
            
        if lines:
            sample = json.loads(lines[10])  # Take 10th sample for testing
            
            decoder = TokenVisualDecoder()
            
            print("📊 Original Sample Info:")
            print(f"   Sample ID: {sample.get('solver_metadata', {}).get('sample_id', 'N/A')}")
            print(f"   Operation: {sample.get('solver_metadata', {}).get('operation', 'N/A')}")
            
            print("\n🔍 Visualization Info Based on Token Decoding:")
            visual_info = decoder.decode_token_sequence(sample['x_k_plus_1'])
            
            print(f"   Filled cells: {visual_info['gridStats']['filled_cells']}")
            print(f"   Colored cells: {visual_info['gridStats']['colored_cells']}")
            print(f"   Special markers: {visual_info['gridStats']['special_markers']}")
            print(f"   Active branches: {visual_info['gridStats']['active_branch_count']}")
            print(f"   Failed positions: {visual_info['gridStats']['failed_position_count']}")
            
            print("\n🔄 Sequence Change Analysis:")
            comparison = decoder.compare_sequences(sample['x_k'], sample['x_k_plus_1'])
            print(f"   Sequence length change: {comparison['length_change']}")
            print(f"   Token changes: {len(comparison['sequence_changes'])}")
            
            if comparison['sequence_changes']:
                print("   Main changes:")
                for change in comparison['sequence_changes'][:3]:  # Show first 3 changes
                    print(f"     Cell({change['cell'][0]},{change['cell'][1]}) {change['token_type']}: {change['from']} → {change['to']}")
            
            print("\n✅ Token decoding test successful!")
            print("💡 Visualization now completely based on token sequences, fully separated from generation module")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Please generate some sample data first")


if __name__ == "__main__":
    main()
