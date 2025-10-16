#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Max-Flow (Edmonds-Karp) Solver - APMDM Sampling Process Framework
Based on pure Token sequence state + atomic operations + APMDM R/Y/E/C micro-step specification

Design:
1. State = flat Token sequence containing all graph info and algorithm state
2. Each step executes one atomic operation, strictly following APMDM R/Y/E/C specification
3. All state decisions based solely on current Token sequence, no external variables
4. 12 atomic operations cover complete Edmonds-Karp algorithm
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

# =============================================================================
# 📚 Token Vocabulary Definition
# =============================================================================

class TokenVocab:
    """Token vocabulary - all possible token types"""
    
    # Structure Tokens
    PROMPT = "PROMPT"
    SRC = "SRC" 
    TGT = "TGT"
    GRAPH = "GRAPH"
    NODES = "NODES"
    EOS = "EOS"
    EOA = "EOA"  # End of Algorithm
    
    # Parentheses delimiters
    LPAREN = "("
    RPAREN = ")"
    
    # Edge feature tokens (dual-slot design)
    FB = "FB"          # Forward/Backward - edge direction availability
    MASK = "MASK"      # Placeholder for APMDM R/Y/E/C operations
    
    # Node feature tokens
    LVL0 = "LVL0"      # Source node level
    LVL1 = "LVL1"      # BFS level 1
    LVL2 = "LVL2"      # BFS level 2  
    LVL3 = "LVL3"      # BFS level 3
    LVL4 = "LVL4"      # BFS level 4
    LVL5 = "LVL5"      # BFS level 5
    LVL6 = "LVL6"      # BFS level 6
    LVL7 = "LVL7"      # BFS level 7
    LVL8 = "LVL8"      # BFS level 8
    LVL9 = "LVL9"      # BFS level 9 (>9 use LVL9)
    INF = "INF"        # Infinite level (unvisited)
    
    PAR = "PAR"        # Parent node marker
    NIL = "NIL"        # Null parent
    
    # Node IDs use integer strings: "0", "1", "2", "3", ...


# =============================================================================
# 🏗️ State Data Structures
# =============================================================================

@dataclass
class EdgeData:
    """Edge data - dual-slot state for each edge
    
    Design: Two slots with mutually exclusive FB/MASK states
    - slot1: Initialize to FB (baseline direction u→v available)
    - slot2: Initialize to MASK (reverse direction v→u unavailable)  
    - On augmentation: flip slots (slot1→MASK, slot2→FB)
    """
    u: int                        # Start node ID
    v: int                        # End node ID
    slot1: str = "NONE"           # First slot: "FB" | "MASK" | "NONE"
    slot2: str = "NONE"           # Second slot: "FB" | "MASK" | "NONE"
    is_initialized: bool = False  # Whether features initialized
    marked_for_deletion: bool = False  # Whether marked for deletion (for cut edge removal)
    
    def is_baseline_direction(self) -> bool:
        """Check if baseline direction (slot1=FB, slot2=MASK)"""
        return self.slot1 == "FB" and self.slot2 == "MASK"
    
    def is_reversed_direction(self) -> bool:
        """Check if reversed (slot1=MASK, slot2=FB)"""
        return self.slot1 == "MASK" and self.slot2 == "FB"
    
    def has_slots(self) -> bool:
        """Check if has slots"""
        return self.slot1 != "NONE" and self.slot2 != "NONE"
    
    def has_both_masks(self) -> bool:
        """Check if both slots are MASK (when just inserted)"""
        return self.slot1 == "MASK" and self.slot2 == "MASK"

@dataclass  
class NodeData:
    """Node data - complete state for each node"""
    node_id: int              # Node ID
    level: str = "NONE"       # BFS level: "NONE", "MASK", "LVL0"-"LVL9", "INF"
    parent: str = "NONE"      # Parent: "NONE", "MASK", "NIL", or node ID like "1","2","3"
    is_initialized: bool = False  # Whether features initialized

@dataclass
class GraphState:
    """Graph state - complete algorithm state representation
    
    Three-part design:
    1. Graph data: edge list with features
    2. Node data: node list with features  
    3. Global variables: global algorithm state
    """
    
    edges: List[EdgeData]         # All edge states
    nodes: List[NodeData]         # All node states
    src_node: int                 # Source node ID
    tgt_node: int                 # Target node ID
    algorithm_phase: str = "INIT" # Algorithm phase: INIT, BFS, AUG, TERM
    algorithm_ended: bool = False # Whether algorithm ended
    augment_count: int = 0        # Augmentation counter (actual max flow value)
    tail_tokens: List[str] = None # Tail tokens like ["EOA", "MASK"] or ["EOA", "EOS"]
    
    def __post_init__(self):
        if self.tail_tokens is None:
            self.tail_tokens = []
    
    def copy(self) -> 'GraphState':
        """Create state copy"""
        new_state = GraphState(
            edges=[EdgeData(u=e.u, v=e.v, slot1=e.slot1, slot2=e.slot2, 
                          is_initialized=e.is_initialized, marked_for_deletion=e.marked_for_deletion) 
                   for e in self.edges],
            nodes=[NodeData(node_id=n.node_id, level=n.level, parent=n.parent, is_initialized=n.is_initialized) 
                   for n in self.nodes],
            src_node=self.src_node,
            tgt_node=self.tgt_node,
            algorithm_phase=self.algorithm_phase,
            algorithm_ended=self.algorithm_ended,
            augment_count=self.augment_count,
            tail_tokens=self.tail_tokens.copy()
        )
        if hasattr(self, 'eos_written'):
            new_state.eos_written = self.eos_written
        return new_state
    
    def get_node_by_id(self, node_id: int) -> Optional[NodeData]:
        """Get node data by ID"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    def get_edge_by_nodes(self, u: int, v: int) -> Optional[EdgeData]:
        """Get edge data by endpoints"""
        for edge in self.edges:
            if (edge.u == u and edge.v == v) or (edge.u == v and edge.v == u):
                return edge
        return None


# =============================================================================
# 🎯 MaxFlow Solver - Main Class
# =============================================================================

class MaxFlowSolver:
    """Max-Flow (Edmonds-Karp) Solver
    
    Core Design Principles:
    1. Single wire loop: while True
    2. State-driven: determine next operation solely from current GraphState  
    3. Atomic: execute one operation per step
    4. APMDM compatible: all modifications follow R/Y/E/C micro-step specification
    5. Clear I/O: each operation current_state -> next_state
    """
    
    def __init__(self):
        """Initialize solver"""
        self.vocab = TokenVocab()
    
    # =========================================================================
    # 🚀 Main Solve Entry
    # =========================================================================
    
    def solve(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main solve function - single wire loop
        
        Args:
            graph_data: Graph data from maxflow_loader
            
        Returns:
            Solve result including final state and max flow value
        """
        if graph_data['source'] == graph_data['target']:
            return {
                'max_flow_value': 0,
                'algorithm_phase': 'FINISHED',
                'edges_count': len(graph_data['edges']),
                'nodes_count': len(graph_data['nodes']),
                'source': graph_data['source'],
                'target': graph_data['target'],
                'error': 'Source equals target, flow is 0'
            }
        
        current_state = self._convert_from_loader(graph_data)
        
        while True:
            operation = self._decide_next_operation(current_state)
            
            if operation is None:
                break
                
            current_state = self._execute_atomic_operation(current_state, operation)
        
        return self._convert_to_result(current_state)
    
    # =========================================================================
    # 🔄 Data Conversion
    # =========================================================================
    
    def _convert_from_loader(self, graph_data: Dict[str, Any]) -> GraphState:
        """Convert from maxflow_loader graph data to algorithm state"""
        edges = []
        for u, v in graph_data['edges']:
            edges.append(EdgeData(
                u=u, v=v,
                slot1="NONE", slot2="NONE",
                is_initialized=False
            ))
        
        nodes = []
        for node_id in graph_data['nodes']:
            nodes.append(NodeData(
                node_id=node_id,
                level="NONE", parent="NONE",
                is_initialized=False
            ))
        
        return GraphState(
            edges=edges,
            nodes=nodes,
            src_node=graph_data['source'],
            tgt_node=graph_data['target'],
            algorithm_phase="INIT",
            algorithm_ended=False
        )
    
    def _convert_to_result(self, state: GraphState) -> Dict[str, Any]:
        """Convert final state to result format"""
        max_flow_value = state.augment_count
        
        return {
            'max_flow_value': max_flow_value,
            'algorithm_phase': state.algorithm_phase,
            'edges_count': len(state.edges),
            'nodes_count': len(state.nodes),
            'source': state.src_node,
            'target': state.tgt_node
        }
    
    # =========================================================================
    # 🧠 Scheduler - State-Driven Operation Decision
    # =========================================================================
    
    def _decide_next_operation(self, state: GraphState) -> Optional[str]:
        """Scheduler - determine next operation based on current state
        
        Core idea: each state has exactly one clear next operation
        """
        
        # Priority 1: Reset MASK nodes after augmentation
        if self._has_augment_mask_nodes(state):
            return "AUG_RESET_NODES"
        
        # Priority 2: Process MASK nodes in BFS phase
        if self._has_bfs_mask_nodes(state):
            return "BFS_LAYER_Y"
        
        # Priority 3: Initialization phase
        if self._is_in_init_phase(state):
            return self._get_next_init_operation(state)
        
        # Priority 4: BFS expansion
        if self._can_do_bfs_expand(state):
            return "BFS_LAYER_R"
        
        # Priority 5: Augmentation
        if self._can_do_augment(state):
            return "AUG_FLIP_EDGES"
        
        # Priority 6: Termination
        return self._get_next_termination_operation(state)
    
    def _is_in_init_phase(self, state: GraphState) -> bool:
        """Check if in initialization phase"""
        return any(not edge.is_initialized for edge in state.edges) or \
               any(not node.is_initialized for node in state.nodes)
    
    def _get_next_init_operation(self, state: GraphState) -> str:
        """Get next initialization operation"""
        if self._has_edges_or_nodes_without_any_slots(state):
            return "INIT_SLOT1_E"
        elif self._has_edges_or_nodes_with_one_mask(state):
            return "INIT_SLOT2_E_AND_Y"
        elif self._has_nodes_with_lvl_and_mask_par(state):
            return "INIT_COMMIT_Y"
        else:
            return None
    
    def _can_do_bfs_expand(self, state: GraphState) -> bool:
        """Check if can perform BFS expansion"""
        all_init = all(node.is_initialized for node in state.nodes)
        current_level = self._get_max_level(state)
        has_relaxable = self._has_relaxable_edges_at_level(state, current_level)
        return all_init and has_relaxable
    
    def _can_do_augment(self, state: GraphState) -> bool:
        """Check if can perform augmentation"""
        target_reachable = self._is_target_reachable(state)
        no_mask_nodes = not any(node.level == "MASK" or node.parent == "MASK" for node in state.nodes)
        return target_reachable and no_mask_nodes
    
    def _get_next_termination_operation(self, state: GraphState) -> Optional[str]:
        """Get next termination operation"""
        # Step 1: Add EOA marker
        if not self._has_eoa_marker(state):
            return "APPEND_EOA_E" if not self._tail_is_mask(state) else "APPEND_EOA_Y"
        
        # Step 2: Delete min-cut edges
        if self._has_cut_edges_to_mark(state):
            return "CUT_MARK_R"
        elif self._has_marked_cut_edges(state):
            return "CUT_DELETE_AND_EXPAND"
        
        # Step 3: Unmask EOS
        elif not self._has_eos_marker(state):
            if self._tail_is_mask(state):
                return "APPEND_EOS_Y"
            else:
                return None
        else:
            return None
    
    # =========================================================================
    # ⚛️ Atomic Operation Executor
    # =========================================================================
    
    def _execute_atomic_operation(self, state: GraphState, operation: str) -> GraphState:
        """Execute specified atomic operation"""
        next_state = state.copy()
        
        if operation == "INIT_SLOT1_E":
            next_state = self._op_init_slot1_e(next_state)
        elif operation == "INIT_SLOT2_E_AND_Y":
            next_state = self._op_init_slot2_e_and_y(next_state)
        elif operation == "INIT_COMMIT_Y":
            next_state = self._op_init_commit_y(next_state)
        elif operation == "BFS_LAYER_R":
            next_state = self._op_bfs_layer_r(next_state)
        elif operation == "BFS_LAYER_Y":
            next_state = self._op_bfs_layer_y(next_state)
        elif operation == "AUG_FLIP_EDGES":
            next_state = self._op_aug_flip_edges(next_state)
        elif operation == "AUG_RESET_NODES":
            next_state = self._op_aug_reset_nodes(next_state)
        elif operation == "AUG_FLIP_AND_RESET":
            next_state = self._op_aug_flip_and_reset(next_state)
        elif operation == "APPEND_EOA_E":
            next_state = self._op_append_eoa_e(next_state)
        elif operation == "APPEND_EOA_Y":
            next_state = self._op_append_eoa_y(next_state)
        elif operation == "CUT_MARK_R":
            next_state = self._op_cut_mark_r(next_state)
        elif operation == "CUT_DELETE_C":
            next_state = self._op_cut_delete_c(next_state)
        elif operation == "CUT_DELETE_AND_EXPAND":
            next_state = self._op_cut_delete_and_expand(next_state)
        elif operation == "APPEND_EOS_E":
            next_state = self._op_append_eos_e(next_state)
        elif operation == "APPEND_EOS_Y":
            next_state = self._op_append_eos_y(next_state)
        else:
            raise ValueError(f"Unknown atomic operation: {operation}")
        
        return next_state
    
    # =========================================================================
    # ⚛️ Atomic Operations - Follow APMDM Two-Step Specification
    # =========================================================================
    
    def _op_init_slot1_e(self, state: GraphState) -> GraphState:
        """A1. INIT_SLOT1_E - Insert first MASK slot for edges and nodes
        
        Guard: Has edges or nodes without slots
        Writes(E): ( u v ) → ( u v MASK ), ( v ) → ( v MASK )
        """
        new_state = state.copy()
        
        for edge in new_state.edges:
            if not edge.has_slots():
                edge.slot1 = "MASK"
        
        for node in new_state.nodes:
            if not node.is_initialized:
                node.level = "MASK"
        
        return new_state
    
    def _op_init_slot2_e_and_y(self, state: GraphState) -> GraphState:
        """A2. INIT_SLOT2_E_AND_Y - Insert second MASK slot + first MASK becomes FB/LVL
        
        Guard: Has edges or nodes with only one MASK slot
        Writes(E+Y): Insert second MASK + first MASK→FB or LVL
        """
        new_state = state.copy()
        
        for edge in new_state.edges:
            if edge.slot1 == "MASK" and edge.slot2 == "NONE":
                edge.slot2 = "MASK"
                edge.slot1 = "FB"
        
        for node in new_state.nodes:
            if node.level == "MASK" and node.parent == "NONE":
                node.parent = "MASK"
                if node.node_id == new_state.src_node:
                    node.level = "LVL0"
                else:
                    node.level = "INF"
        
        return new_state
    
    def _op_init_commit_y(self, state: GraphState) -> GraphState:
        """A3. INIT_COMMIT_Y - Second MASK becomes final value
        
        Guard: Has nodes with second slot as MASK
        Writes(Y): Second MASK→NIL
        """
        new_state = state.copy()
        
        for node in new_state.nodes:
            if node.parent == "MASK":
                node.parent = "NIL"
                node.is_initialized = True
        
        for edge in new_state.edges:
            if edge.slot1 == "FB" and edge.slot2 == "MASK":
                edge.is_initialized = True
        
        return new_state
    
    def _op_bfs_layer_r(self, state: GraphState) -> GraphState:
        """B1. BFS_LAYER_R - Mark expandable nodes at current layer
        
        Guard: Has relaxable edges at current layer k
        Writes(R): For newly discovered nodes: LVL INF→MASK, PAR NIL→MASK
        Parent selection rule: choose node with smallest ID
        """
        new_state = state.copy()
        current_level = self._get_max_level(state)
        
        node_levels = {}
        for node in state.nodes:
            if node.level.startswith("LVL"):
                level_num = int(node.level[3:])
                node_levels[node.node_id] = level_num
            else:
                node_levels[node.node_id] = None
        
        discoveries = {}
        
        for edge in state.edges:
            if not edge.is_initialized:
                continue
                
            u_level = node_levels.get(edge.u, None)
            v_level = node_levels.get(edge.v, None)
            
            if edge.slot1 == "FB" and edge.slot2 == "MASK":
                if u_level == current_level and v_level is None:
                    if edge.v not in discoveries or edge.u < discoveries[edge.v]:
                        discoveries[edge.v] = edge.u
            
            if edge.slot1 == "MASK" and edge.slot2 == "FB":
                if v_level == current_level and u_level is None:
                    if edge.u not in discoveries or edge.v < discoveries[edge.u]:
                        discoveries[edge.u] = edge.v
        
        for node in new_state.nodes:
            if node.node_id in discoveries:
                node.level = "MASK"
                node.parent = "MASK"
        
        return new_state
    
    def _op_bfs_layer_y(self, state: GraphState) -> GraphState:
        """B2. BFS_LAYER_Y - Set level and parent for marked nodes
        
        Guard: Has nodes with LVL=MASK or PAR=MASK
        Writes(Y): LVL MASK→LVL(k+1), PAR MASK→parent_id
        """
        new_state = state.copy()
        current_level = self._get_max_level(state)
        
        node_levels = {}
        for node in state.nodes:
            if node.level == "MASK":
                node_levels[node.node_id] = None
            elif node.level.startswith("LVL"):
                level_num = int(node.level[3:])
                node_levels[node.node_id] = level_num
            else:
                node_levels[node.node_id] = None
        
        discoveries = {}
        for edge in state.edges:
            if not edge.is_initialized:
                continue
                
            u_level = node_levels.get(edge.u, None)
            v_level = node_levels.get(edge.v, None)
            
            if edge.slot1 == "FB" and edge.slot2 == "MASK":
                if u_level == current_level and v_level is None:
                    if edge.v not in discoveries or edge.u < discoveries[edge.v]:
                        discoveries[edge.v] = edge.u
            
            if edge.slot1 == "MASK" and edge.slot2 == "FB":
                if v_level == current_level and u_level is None:
                    if edge.u not in discoveries or edge.v < discoveries[edge.u]:
                        discoveries[edge.u] = edge.v
        
        for node in new_state.nodes:
            if node.level == "MASK" and node.parent == "MASK":
                if node.node_id in discoveries:
                    node.level = f"LVL{current_level + 1}"
                    node.parent = str(discoveries[node.node_id])
        
        return new_state
    
    def _op_aug_flip_edges(self, state: GraphState) -> GraphState:
        """C1. AUG_FLIP_EDGES - Flip edges and mark nodes as MASK
        
        Guard: Target reachable, need to flip path edges
        Writes(R+Y): Flip path edges + mark visited nodes as MASK
        """
        new_state = state.copy()
        
        path = self._trace_augmenting_path(state)
        new_state.augment_count += 1
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = new_state.get_edge_by_nodes(u, v)
            
            if edge:
                if edge.slot1 == "FB" and edge.slot2 == "MASK":
                    edge.slot1 = "MASK"
                    edge.slot2 = "FB"
                elif edge.slot1 == "MASK" and edge.slot2 == "FB":
                    edge.slot1 = "FB"
                    edge.slot2 = "MASK"
        
        for node in new_state.nodes:
            node.level = "MASK"
            node.parent = "MASK"
        
        return new_state
    
    def _op_aug_reset_nodes(self, state: GraphState) -> GraphState:
        """C2. AUG_RESET_NODES - Reset all MASK nodes to initial values
        
        Guard: All nodes marked as MASK
        Writes(Y): MASK→initial values
        """
        new_state = state.copy()
        
        for node in new_state.nodes:
            if node.level == "MASK":
                if node.node_id == new_state.src_node:
                    node.level = "LVL0"
                else:
                    node.level = "INF"
            
            if node.parent == "MASK":
                node.parent = "NIL"
        
        return new_state
    
    def _op_aug_flip_and_reset(self, state: GraphState) -> GraphState:
        """C1. AUG_FLIP_AND_RESET - Augment and reset (merged operation)
        
        Guard: Target reachable, need augmentation
        Writes(R+Y): Flip path edges + reset visited nodes
        """
        new_state = state.copy()
        
        path = self._trace_augmenting_path(state)
        new_state.augment_count += 1
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = new_state.get_edge_by_nodes(u, v)
            
            if edge:
                if edge.slot1 == "FB" and edge.slot2 == "MASK":
                    edge.slot1 = "MASK"
                    edge.slot2 = "FB"
                elif edge.slot1 == "MASK" and edge.slot2 == "FB":
                    edge.slot1 = "FB"
                    edge.slot2 = "MASK"
        
        for node in new_state.nodes:
            node.level = "MASK"
            node.parent = "MASK"
        
        return new_state
    
    def _trace_augmenting_path(self, state: GraphState) -> List[int]:
        """Trace back augmenting path from target to source"""
        path = []
        current = state.tgt_node
        
        while current != -1:
            path.append(current)
            node = state.get_node_by_id(current)
            if node and node.parent.isdigit():
                parent_id = int(node.parent)
                current = parent_id
            else:
                break
        
        path.reverse()
        return path
    
    def _op_append_eoa_e(self, state: GraphState) -> GraphState:
        """D1. APPEND_EOA_E - Insert MASK at end
        
        Writes(E): Insert MASK at tail
        """
        new_state = state.copy()
        new_state.tail_tokens.append("MASK")
        return new_state
    
    def _op_append_eoa_y(self, state: GraphState) -> GraphState:
        """D2. APPEND_EOA_Y - Write algorithm end marker
        
        Writes(Y): MASK→EOA
        """
        new_state = state.copy()
        if new_state.tail_tokens and new_state.tail_tokens[-1] == "MASK":
            new_state.tail_tokens[-1] = "EOA"
        return new_state
    
    def _op_cut_mark_r(self, state: GraphState) -> GraphState:
        """D3. CUT_MARK_R - Mark all min-cut edges as MASK
        
        Guard: Has cut edges (u∈S, v∉S) where S={v|LVL[v]≠INF}
        Writes(R): Mark all cut edges for deletion
        """
        new_state = state.copy()
        
        reachable_nodes = set()
        for node in state.nodes:
            if node.level != "INF" and node.level != "NONE":
                reachable_nodes.add(node.node_id)
        
        cut_edges = []
        for edge in state.edges:
            if edge.is_initialized and not edge.marked_for_deletion:
                u_reachable = edge.u in reachable_nodes
                v_reachable = edge.v in reachable_nodes
                
                if u_reachable and not v_reachable:
                    cut_edges.append(edge)
        
        if cut_edges:
            cut_edges.sort(key=lambda e: (e.u, e.v))
            
            for cut_edge in cut_edges:
                for edge in new_state.edges:
                    if edge.u == cut_edge.u and edge.v == cut_edge.v:
                        edge.marked_for_deletion = True
                        break
        
        return new_state
    
    def _op_cut_delete_c(self, state: GraphState) -> GraphState:
        """D4. CUT_DELETE_C - Delete marked min-cut edges (deprecated)"""
        new_state = state.copy()
        new_state.edges = [edge for edge in new_state.edges if not edge.marked_for_deletion]
        return new_state
    
    def _op_cut_delete_and_expand(self, state: GraphState) -> GraphState:
        """D4. CUT_DELETE_AND_EXPAND - Delete cut edges and expand MASK for EOS
        
        Writes(C+E): Delete marked cut edges + expand MASK at tail
        """
        new_state = state.copy()
        
        new_state.edges = [
            edge for edge in new_state.edges 
            if not edge.marked_for_deletion
        ]
        
        new_state.tail_tokens.append("MASK")
        
        return new_state
    
    def _op_append_eos_e(self, state: GraphState) -> GraphState:
        """D5. APPEND_EOS_E - Insert MASK at end
        
        Writes(E): Insert MASK at tail
        """
        new_state = state.copy()
        new_state.tail_tokens.append("MASK")
        return new_state
    
    def _op_append_eos_y(self, state: GraphState) -> GraphState:
        """D6. APPEND_EOS_Y - Write sequence end marker
        
        Writes(Y): MASK→EOS
        """
        new_state = state.copy()
        if new_state.tail_tokens and new_state.tail_tokens[-1] == "MASK":
            new_state.tail_tokens[-1] = "EOS"
        new_state.algorithm_phase = "FINISHED"
        return new_state
    
    # =========================================================================
    # 🔍 State Query Functions
    # =========================================================================
    
    def _has_edges_or_nodes_without_any_slots(self, state: GraphState) -> bool:
        """Check if has edges or nodes without any slots"""
        has_edges_without_slots = any(
            edge.slot1 == "NONE" and edge.slot2 == "NONE" 
            for edge in state.edges
        )
        has_nodes_without_slots = any(
            node.level == "NONE" and node.parent == "NONE" and not node.is_initialized
            for node in state.nodes
        )
        return has_edges_without_slots or has_nodes_without_slots
    
    def _has_edges_or_nodes_with_one_mask(self, state: GraphState) -> bool:
        """Check if has edges or nodes with only one MASK slot"""
        has_edges_with_one_mask = any(
            edge.slot1 == "MASK" and edge.slot2 == "NONE" 
            for edge in state.edges
        )
        has_nodes_with_one_mask = any(
            node.level == "MASK" and node.parent == "NONE"
            for node in state.nodes
        )
        return has_edges_with_one_mask or has_nodes_with_one_mask
    
    def _has_nodes_with_lvl_and_mask_par(self, state: GraphState) -> bool:
        """Check if has nodes with LVL set but PAR still MASK"""
        return any(
            node.parent == "MASK"
            for node in state.nodes
        )
    
    def _has_nodes_with_mask_slots(self, state: GraphState) -> bool:
        """Check if has nodes with LVL or PAR as MASK"""
        return any(
            node.level == "MASK" and node.parent == "MASK"
            for node in state.nodes
        )
    
    def _get_max_level(self, state: GraphState) -> int:
        """Get current maximum BFS level"""
        max_level = 0
        for node in state.nodes:
            if node.level.startswith("LVL"):
                level_num = int(node.level[3:])
                max_level = max(max_level, level_num)
        return max_level
    
    def _has_relaxable_edges_at_level(self, state: GraphState, level: int) -> bool:
        """Check if has relaxable edges at specified level"""
        node_levels = {}
        for node in state.nodes:
            if node.level == "MASK":
                node_levels[node.node_id] = None
            elif node.level.startswith("LVL"):
                level_num = int(node.level[3:])
                node_levels[node.node_id] = level_num
            else:
                node_levels[node.node_id] = None
        
        for edge in state.edges:
            if not edge.is_initialized:
                continue
                
            u_level = node_levels.get(edge.u, None)
            v_level = node_levels.get(edge.v, None)
            
            if edge.slot1 == "FB" and edge.slot2 == "MASK":
                if u_level == level and v_level is None:
                    return True
            
            if edge.slot1 == "MASK" and edge.slot2 == "FB":
                if v_level == level and u_level is None:
                    return True
        
        return False
    
    def _is_target_reachable(self, state: GraphState) -> bool:
        """Check if target node is reachable"""
        target_node = state.get_node_by_id(state.tgt_node)
        return target_node is not None and target_node.level.startswith("LVL")
    
    def _has_nodes_to_reset(self, state: GraphState) -> bool:
        """Check if has nodes to reset (after edge flip)"""
        return any(
            node.level.startswith("LVL") and node.node_id != state.src_node
            for node in state.nodes
        )
    
    def _has_edges_been_flipped(self, state: GraphState) -> bool:
        """Check if has edges been flipped (need to mark nodes as MASK)"""
        has_flipped_edges = any(
            edge.slot1 == "MASK" and edge.slot2 == "FB"
            for edge in state.edges
        )
        
        has_unmasked_nodes = any(
            node.level.startswith("LVL") and node.node_id != state.src_node
            for node in state.nodes
        )
        
        return has_flipped_edges and has_unmasked_nodes
    
    def _has_augment_mask_nodes(self, state: GraphState) -> bool:
        """Check if has MASK nodes after augmentation that need reset"""
        has_flipped_edges = any(edge.slot1 == "MASK" and edge.slot2 == "FB" for edge in state.edges)
        
        if not has_flipped_edges:
            return False
        
        all_nodes_masked = all(node.level == "MASK" and node.parent == "MASK" for node in state.nodes)
        
        return all_nodes_masked
    
    def _has_bfs_mask_nodes(self, state: GraphState) -> bool:
        """Check if has MASK nodes in BFS phase to process"""
        has_mask_nodes = any(node.level == "MASK" and node.parent == "MASK" for node in state.nodes)
        all_nodes_masked = all(node.level == "MASK" and node.parent == "MASK" for node in state.nodes)
        has_flipped_edges = any(edge.slot1 == "MASK" and edge.slot2 == "FB" for edge in state.edges)
        
        return has_mask_nodes and not all_nodes_masked
    
    def _has_masked_nodes_to_reset(self, state: GraphState) -> bool:
        """Check if has MASK nodes to reset"""
        return any(
            node.level == "MASK" or node.parent == "MASK"
            for node in state.nodes
        )
    
    def _has_path_edges_to_flip(self, state: GraphState) -> bool:
        """Check if has path edges to flip"""
        return self._is_target_reachable(state)
    
    def _has_eoa_marker(self, state: GraphState) -> bool:
        """Check if has EOA marker"""
        return "EOA" in state.tail_tokens
    
    def _tail_is_mask(self, state: GraphState) -> bool:
        """Check if tail is MASK"""
        return state.tail_tokens and state.tail_tokens[-1] == "MASK"
    
    def _has_marked_cut_edges(self, state: GraphState) -> bool:
        """Check if has marked cut edges"""
        return any(
            hasattr(edge, 'marked_for_deletion') and edge.marked_for_deletion
            for edge in state.edges
        )
    
    def _has_cut_edges_to_mark(self, state: GraphState) -> bool:
        """Check if has cut edges to mark"""
        reachable_nodes = set()
        for node in state.nodes:
            if node.level != "INF" and node.level != "NONE":
                reachable_nodes.add(node.node_id)
        
        for edge in state.edges:
            if edge.is_initialized and not edge.marked_for_deletion:
                u_reachable = edge.u in reachable_nodes
                v_reachable = edge.v in reachable_nodes
                
                if u_reachable and not v_reachable:
                    return True
        
        return False
    
    def _has_eos_marker(self, state: GraphState) -> bool:
        """Check if has EOS marker"""
        return "EOS" in state.tail_tokens


# =============================================================================
# 🎯 MaxFlow APMDM Sample Generator
# =============================================================================

class MaxFlowGenerator:
    """MaxFlow APMDM training data generator
    
    Listen to solver state transitions and translate atomic operations to APMDM R/Y/E/C
    """
    
    def __init__(self):
        self.samples: List[Dict[str, Any]] = []
        self.sample_counter = 0
        self.current_instance_id = 0
        self.current_step_id = 0
        self.vocab_cache: Optional[Dict[int, str]] = None
        self.token_to_id: Optional[Dict[str, int]] = None
        
    def reset_for_new_instance(self, instance_id: int):
        """Reset generator for new graph instance"""
        self.current_instance_id = instance_id
        self.current_step_id = 0
    
    def state_to_tokens(self, state: GraphState) -> List[int]:
        """Convert GraphState to token sequence"""
        if self.token_to_id is None:
            self.create_vocabulary()
        
        tokens = []
        
        tokens.extend([
            self.token_to_id["PROMPT"],
            self.token_to_id["SRC"],
            self.token_to_id[str(state.src_node)],
            self.token_to_id["TGT"], 
            self.token_to_id[str(state.tgt_node)]
        ])
        
        tokens.append(self.token_to_id["GRAPH"])
        
        for edge in state.edges:
            if edge.marked_for_deletion:
                tokens.append(self.token_to_id["[MASK]"])
                tokens.append(self.token_to_id["[MASK]"])
                tokens.append(self.token_to_id["[MASK]"])
                tokens.append(self.token_to_id["[MASK]"])
                tokens.append(self.token_to_id["[MASK]"])
                tokens.append(self.token_to_id["[MASK]"])
            else:
                tokens.append(self.token_to_id["("])
                tokens.append(self.token_to_id[str(edge.u)])
                tokens.append(self.token_to_id[str(edge.v)])
                
                if edge.slot1 != "NONE":
                    slot1_token = f"[{edge.slot1}]" if edge.slot1 == "MASK" else edge.slot1
                    tokens.append(self.token_to_id[slot1_token])
                if edge.slot2 != "NONE":
                    slot2_token = f"[{edge.slot2}]" if edge.slot2 == "MASK" else edge.slot2
                    tokens.append(self.token_to_id[slot2_token])
                
                tokens.append(self.token_to_id[")"]) 
        
        tokens.append(self.token_to_id["NODES"])
        
        for node in state.nodes:
            tokens.append(self.token_to_id["("])
            tokens.append(self.token_to_id[str(node.node_id)])
            
            if node.level != "NONE":
                if node.level in ["MASK", "INF", "NIL"]:
                    level_token = f"[{node.level}]"
                elif node.level.startswith("LVL"):
                    level_token = f"[{node.level}]"
                else:
                    level_token = node.level
                tokens.append(self.token_to_id[level_token])
            if node.parent != "NONE":
                if node.parent in ["MASK", "NIL"]:
                    parent_token = f"[{node.parent}]"
                elif node.parent.isdigit():
                    parent_token = node.parent
                else:
                    parent_token = f"[{node.parent}]"
                tokens.append(self.token_to_id[parent_token])
            
            tokens.append(self.token_to_id[")"]) 
        
        for tail_token in state.tail_tokens:
            if tail_token == "MASK":
                token_key = "[MASK]"
            elif tail_token == "EOS":
                token_key = "[EOS]"
            elif tail_token == "EOA":
                token_key = "[EOA]"
            else:
                token_key = tail_token
            tokens.append(self.token_to_id[token_key])
        
        return tokens
    
    def generate_gdlm_sample(self, 
                           state_before: GraphState, 
                           operation: str,
                           state_after: GraphState) -> Dict[str, Any]:
        """Generate single APMDM training sample"""
        x_k = self.state_to_tokens(state_before)
        x_k_plus_1 = self.state_to_tokens(state_after)
        
        y_star, r_star, e_star, c_star = self._translate_operation_to_gdlm(
            state_before, operation, state_after, x_k
        )
        
        self.sample_counter += 1
        self.current_step_id += 1
        
        sample = {
            "x_k": x_k,
            "y_star": y_star,
            "r_star": r_star, 
            "e_star": e_star,
            "c_star": c_star,
            "x_k_plus_1": x_k_plus_1,
            "solver_metadata": {
                "operation": operation,
                "x_k_length": len(x_k),
                "x_k_plus_1_length": len(x_k_plus_1),
                "instance_id": self.current_instance_id,
                "step_id": self.current_step_id,
                "sample_id": self.sample_counter,
                "algorithm_phase": state_before.algorithm_phase,
            }
        }
        
        self.samples.append(sample)
        return sample
    
    def _translate_operation_to_gdlm(self, 
                                   state_before: GraphState,
                                   operation: str, 
                                   state_after: GraphState,
                                   x_k: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Translate solver atomic operation to APMDM R/Y/E/C operation labels
        
        APMDM specification:
        - e_star[i]=1: Insert MASK after position i
        - r_star[i]=1: Position i becomes MASK (highest priority)
        - y_star[i]: If position i is MASK and r_star[i]=0, becomes y_star[i]
        """
        seq_len = len(x_k)
        y_star = x_k.copy()
        r_star = [0] * seq_len
        e_star = [0] * seq_len  
        c_star = [0] * seq_len
        
        x_k_plus_1 = self.state_to_tokens(state_after)
        mask_token_id = self.token_to_id["[MASK]"]
        
        if operation == "INIT_SLOT1_E":
            graph_token_id = self.token_to_id["GRAPH"]
            nodes_token_id = self.token_to_id["NODES"]
            lparen_token_id = self.token_to_id["("]
            rparen_token_id = self.token_to_id[")"]
            
            graph_start = -1
            nodes_start = -1
            for i in range(len(x_k)):
                if x_k[i] == graph_token_id:
                    graph_start = i
                elif x_k[i] == nodes_token_id:
                    nodes_start = i
                    break
            
            if graph_start != -1 and nodes_start != -1:
                i = graph_start + 1
                while i < nodes_start:
                    if x_k[i] == lparen_token_id:
                        if i + 3 < nodes_start and x_k[i + 3] == rparen_token_id:
                            e_star[i + 2] = 1  
                        i += 4  
                    else:
                        i += 1
            
            if nodes_start != -1:
                i = nodes_start + 1
                while i < len(x_k):
                    if x_k[i] == lparen_token_id:
                        if i + 2 < len(x_k) and x_k[i + 2] == rparen_token_id:
                            e_star[i + 1] = 1  
                        i += 3  
                    else:
                        i += 1
        
        elif operation == "INIT_SLOT2_E_AND_Y":
            graph_token_id = self.token_to_id["GRAPH"]
            nodes_token_id = self.token_to_id["NODES"]
            lparen_token_id = self.token_to_id["("]
            
            graph_start = -1
            nodes_start = -1
            for i in range(len(x_k)):
                if x_k[i] == graph_token_id:
                    graph_start = i
                elif x_k[i] == nodes_token_id:
                    nodes_start = i
                    break
            
            if graph_start != -1 and nodes_start != -1:
                for i in range(graph_start + 1, nodes_start):
                    if x_k[i] == mask_token_id:
                        e_star[i] = 1  
                        y_star[i] = self.token_to_id["FB"]  
            
            if nodes_start != -1:
                for i in range(nodes_start + 1, len(x_k)):
                    if x_k[i] == mask_token_id:
                        e_star[i] = 1  
                        node_id = None
                        for j in range(i-1, max(nodes_start, i-5), -1):
                            if x_k[j] == lparen_token_id:
                                if j + 1 < len(x_k):
                                    node_token = x_k[j + 1]
                                    for nid in range(100):
                                        if self.token_to_id.get(str(nid)) == node_token:
                                            node_id = nid
                                            break
                                break
                        
                        if node_id == state_before.src_node:
                            y_star[i] = self.token_to_id["[LVL0]"]  
                        else:
                            y_star[i] = self.token_to_id["[INF]"]   
        
        elif operation == "INIT_COMMIT_Y":
            for i in range(min(len(x_k), len(x_k_plus_1))):
                if x_k[i] == mask_token_id and x_k_plus_1[i] != mask_token_id:
                    y_star[i] = x_k_plus_1[i]  
        
        elif operation == "BFS_LAYER_R":
            
            for i in range(len(x_k)):
                if i < len(x_k_plus_1) and x_k[i] != mask_token_id and x_k_plus_1[i] == mask_token_id:
                    r_star[i] = 1
        
        elif operation == "BFS_LAYER_Y":
            
            for i in range(len(x_k)):
                if x_k[i] == mask_token_id and i < len(x_k_plus_1):
                    y_star[i] = x_k_plus_1[i]
        
        elif operation == "AUG_FLIP_EDGES":
            for i in range(min(len(x_k), len(x_k_plus_1))):
                if x_k[i] != x_k_plus_1[i]:
                    if x_k_plus_1[i] == mask_token_id:
                        r_star[i] = 1
                    else:
                        y_star[i] = x_k_plus_1[i]
        
        elif operation == "AUG_RESET_NODES":
            for i in range(min(len(x_k), len(x_k_plus_1))):
                if x_k[i] == mask_token_id and x_k_plus_1[i] != mask_token_id:
                    y_star[i] = x_k_plus_1[i]
        
        elif operation == "AUG_FLIP_AND_RESET":
            for i in range(min(len(x_k), len(x_k_plus_1))):
                if x_k[i] != x_k_plus_1[i]:
                    if x_k_plus_1[i] == mask_token_id:
                        r_star[i] = 1
                    else:
                        y_star[i] = x_k_plus_1[i]
        
        elif operation.startswith("APPEND") and operation.endswith("_E"):
            if len(x_k_plus_1) > len(x_k):
                e_star[len(x_k)-1] = 1
        
        elif operation.startswith("APPEND") and operation.endswith("_Y"):
            for i in range(len(x_k)):
                if x_k[i] == mask_token_id and i < len(x_k_plus_1):
                    y_star[i] = x_k_plus_1[i]
        
        elif operation == "CUT_MARK_R":
            graph_token_id = self.token_to_id["GRAPH"]
            nodes_token_id = self.token_to_id["NODES"]
            lparen_token_id = self.token_to_id["("]
            rparen_token_id = self.token_to_id[")"]
            
            graph_start = -1
            nodes_start = -1
            for i in range(len(x_k)):
                if x_k[i] == graph_token_id:
                    graph_start = i
                elif x_k[i] == nodes_token_id:
                    nodes_start = i
                    break
            
            if graph_start != -1 and nodes_start != -1:
                i = graph_start + 1
                while i < nodes_start:
                    if i < len(x_k_plus_1) and x_k[i] != x_k_plus_1[i]:
                        if x_k[i] == lparen_token_id and x_k_plus_1[i] == mask_token_id:
                            r_star[i] = 1  
                            i += 1
                            while i < nodes_start and x_k[i] != rparen_token_id:
                                r_star[i] = 1
                                i += 1
                            if i < nodes_start and x_k[i] == rparen_token_id:
                                r_star[i] = 1
                            i += 1
                        else:
                            i += 1
                    else:
                        i += 1
        
        elif operation == "CUT_DELETE_C":
            pass
        
        elif operation == "CUT_DELETE_AND_EXPAND":
            graph_token_id = self.token_to_id["GRAPH"]
            nodes_token_id = self.token_to_id["NODES"]
            
            graph_start = -1
            nodes_start = -1
            for i in range(len(x_k)):
                if x_k[i] == graph_token_id:
                    graph_start = i
                elif x_k[i] == nodes_token_id:
                    nodes_start = i
                    break
            
            if graph_start != -1 and nodes_start != -1:
                i = graph_start + 1
                while i < nodes_start:
                    if i + 5 < nodes_start and all(x_k[i+j] == mask_token_id for j in range(6)):
                        for j in range(6):
                            c_star[i+j] = 1
                        i += 6
                    else:
                        i += 1
            
            e_star[len(x_k)-1] = 1
        
        return y_star, r_star, e_star, c_star
    
    def create_vocabulary(self) -> Dict[int, str]:
        """Create token vocabulary mapping"""
        if self.vocab_cache is not None:
            return self.vocab_cache
            
        vocab = {}
        
        vocab[0] = "PROMPT"
        vocab[1] = "SRC" 
        vocab[2] = "TGT"
        vocab[3] = "GRAPH"
        vocab[4] = "NODES"
        vocab[5] = "[EOS]"
        vocab[6] = "[MASK]"
        vocab[7] = "("
        vocab[8] = ")"
        vocab[9] = "FB"
        vocab[10] = "[EOA]"
        
        token_id = 11
        
        for token in ["[INF]", "[NIL]"]:
            vocab[token_id] = token
            token_id += 1
            
        for i in range(20):
            vocab[token_id] = f"[LVL{i}]"
            token_id += 1
        
        for i in range(20):
            vocab[token_id] = str(i)
            token_id += 1
        
        self.vocab_cache = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        return vocab
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all generated samples"""
        return self.samples.copy()
    
    def save_samples(self, output_file: str, format: str = 'pkl') -> None:
        """Save samples to file"""
        print(f"💾 Save {len(self.samples)} samples to {output_file}")


# =============================================================================
# 🎯 MaxFlow Data Manager - High-Level Interface
# =============================================================================

class MaxFlowDataManager:
    """MaxFlow data manager - integrate solver, generator and loader"""
    
    def __init__(self):
        self.solver = MaxFlowSolver()
        self.generator = MaxFlowGenerator()
        self.global_sample_counter = 0
    
    def generate_from_graph(self, graph_data: Dict[str, Any], instance_id: int = 0) -> List[Dict[str, Any]]:
        """Generate APMDM training samples from single graph"""
        self.generator.reset_for_new_instance(instance_id)
        current_state = self.solver._convert_from_loader(graph_data)
        augment_count = 0
        
        while True:
            operation = self.solver._decide_next_operation(current_state)
            
            if operation is None:
                break
            
            if operation.startswith("AUG") and "FLIP" in operation:
                augment_count += 1
            
            state_before = current_state.copy()
            state_after = self.solver._execute_atomic_operation(current_state, operation)
            self.generator.generate_gdlm_sample(state_before, operation, state_after)
            current_state = state_after
        
        samples = self.generator.get_samples()
        
        for i, sample in enumerate(samples):
            sample['solver_metadata']['max_flow_value'] = current_state.augment_count
        
        return samples
    
    def generate_from_loader_data(self, loader_graph: Dict[str, Any], instance_id: int = 0) -> List[Dict[str, Any]]:
        """Generate APMDM samples from maxflow_loader graph data"""
        graph_data = {
            'edges': loader_graph['edge_list'],
            'nodes': loader_graph['node_list'],
            'source': loader_graph['source_node'],
            'target': loader_graph['target_node']
        }
        
        return self.generate_from_graph(graph_data, instance_id)
    
    def generate_random_dataset(self, num_instances: int = 100,
                              min_nodes: int = 10, max_nodes: int = 15,
                              min_edges: int = 30, max_edges: int = 50,
                              min_flow: int = 3, max_flow: int = 5,
                              seed: int = None) -> List[Dict[str, Any]]:
        """Generate random graph dataset"""
        import sys, os
        
        try:
            from maxflow_loader import generate_dataset
        except ImportError:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.append(current_dir)
            from maxflow_loader import generate_dataset
        
        print(f"🎯 Generate MaxFlow dataset: {num_instances} graphs")
        print(f"📊 Params: nodes {min_nodes}-{max_nodes}, edges {min_edges}-{max_edges}, flow {min_flow}-{max_flow}")
        
        loader_dataset = generate_dataset(
            num_samples=num_instances,
            min_nodes=min_nodes, max_nodes=max_nodes,
            min_edges=min_edges, max_edges=max_edges,
            min_flow=min_flow, max_flow=max_flow,
            seed=seed
        )
        
        all_samples = []
        success_count = 0
        
        for i, loader_graph in enumerate(loader_dataset['data']):
            try:
                if i % 20 == 0:
                    print(f"📈 Progress: {i+1}/{num_instances}")
                
                expected_flow = loader_graph.get('ground_truth_max_flow', 0)
                if expected_flow < min_flow:
                    continue
                
                graph_manager = MaxFlowDataManager()
                samples = graph_manager.generate_from_loader_data(loader_graph, instance_id=i)
                
                for j, sample in enumerate(samples):
                    sample['solver_metadata']['sample_id'] = len(all_samples) + j + 1
                
                all_samples.extend(samples)
                success_count += 1
                
            except Exception as e:
                print(f"⚠️ Graph {i} generation failed: {e}")
                continue
        
        print(f"✅ Complete: {success_count}/{num_instances} graphs, total samples: {len(all_samples)}")
        return all_samples
    
    def save_samples(self, samples: List[Dict[str, Any]], output_file: str, 
                    format: str = 'pkl', compress: bool = True) -> None:
        """Save samples to file"""
        import pickle
        import gzip
        import json
        import os
        from pathlib import Path
        
        if format.lower() == 'pkl':
            data = {
                'samples': samples, 
                'total_samples': len(samples), 
                'format_version': '1.0',
                'dataset_type': 'maxflow_apmdm'
            }
            
            if compress or output_file.endswith('.gz'):
                with gzip.open(output_file, 'wb', compresslevel=6) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            file_size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"💾 Saved {len(samples)} samples: {output_file} ({file_size_mb:.1f}MB)")
            
        elif format.lower() == 'jsonl':
            def convert_numpy(obj):
                if hasattr(obj, 'item') and hasattr(obj, 'shape') and obj.shape == ():
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
                    return int(obj)
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
    
    def generate_complete_dataset(self, output_file: str = "maxflow.pkl.gz",
                                num_instances: int = 100,
                                min_nodes: int = 10, max_nodes: int = 15,
                                min_edges: int = 30, max_edges: int = 50,
                                min_flow: int = 3, max_flow: int = 5,
                                seed: int = None) -> Dict[str, Any]:
        """Generate complete dataset and save"""
        try:
            vocab = self.generator.create_vocabulary()
            print(f"📚 Vocabulary: {len(vocab)} tokens")
            
            import pickle
            from pathlib import Path
            vocab_cache_file = "vocab_cache.pkl"
            if not Path(vocab_cache_file).exists():
                with open(vocab_cache_file, 'wb') as f:
                    pickle.dump(vocab, f)
                print(f"💾 Saved vocabulary cache: {vocab_cache_file}")
            
            all_samples = self.generate_random_dataset(
                num_instances=num_instances,
                min_nodes=min_nodes, max_nodes=max_nodes,
                min_edges=min_edges, max_edges=max_edges,
                min_flow=min_flow, max_flow=max_flow,
                seed=seed
            )
            
            self.save_samples(all_samples, output_file, format='pkl')
            
            if all_samples:
                sequence_lengths = [len(s['x_k']) for s in all_samples]
                avg_length = sum(sequence_lengths) / len(sequence_lengths)
                max_length = max(sequence_lengths)
                min_length = min(sequence_lengths)
                
                print(f"📏 Sequence length stats:")
                print(f"  - Average: {avg_length:.1f}")
                print(f"  - Maximum: {max_length}")
                print(f"  - Minimum: {min_length}")
            
            return {
                'success': True,
                'total_instances': num_instances,
                'successful_instances': len(set(s['solver_metadata']['instance_id'] for s in all_samples)),
                'total_samples': len(all_samples),
                'output_file': output_file,
                'avg_samples_per_instance': len(all_samples) / num_instances if num_instances > 0 else 0,
                'sequence_stats': {
                    'avg_length': avg_length if all_samples else 0,
                    'max_length': max_length if all_samples else 0,
                    'min_length': min_length if all_samples else 0
                } if all_samples else {}
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_from_file(self, loader_file: str, output_file: str) -> Dict[str, Any]:
        """Generate APMDM dataset from maxflow_loader JSON file"""
        import json
        from pathlib import Path
        
        try:
            if not Path(loader_file).exists():
                raise FileNotFoundError(f"File not found: {loader_file}")
            
            print(f"🚀 Generate from file: {loader_file} → {output_file}")
            
            with open(loader_file, 'r', encoding='utf-8') as f:
                loader_dataset = json.load(f)
            
            graphs = loader_dataset['data']
            total = len(graphs)
            
            vocab = self.generator.create_vocabulary()
            print(f"📚 Vocabulary: {len(vocab)} tokens")
            
            all_samples = []
            success_count = 0
            
            for i, loader_graph in enumerate(graphs):
                try:
                    if i % 20 == 0:
                        print(f"📊 {i+1}/{total}")
                    
                    expected_flow = loader_graph.get('ground_truth_max_flow', 0)
                    if expected_flow == 0:
                        continue
                    
                    graph_manager = MaxFlowDataManager()
                    samples = graph_manager.generate_from_loader_data(loader_graph, instance_id=i)
                    
                    for j, sample in enumerate(samples):
                        sample['solver_metadata']['sample_id'] = len(all_samples) + j + 1
                    
                    all_samples.extend(samples)
                    success_count += 1
                    
                except Exception as e:
                    if i % 100 == 0:
                        print(f"⚠️ Graph {i}: {str(e)}")
            
            self.save_samples(all_samples, output_file, format='pkl')
            
            if all_samples:
                sequence_lengths = [len(s['x_k']) for s in all_samples]
                avg_length = sum(sequence_lengths) / len(sequence_lengths)
                max_length = max(sequence_lengths)
                min_length = min(sequence_lengths)
                
                print(f"📏 Sequence length stats:")
                print(f"  - Average: {avg_length:.1f}")
                print(f"  - Maximum: {max_length}")
                print(f"  - Minimum: {min_length}")
            
            return {
                'success': True,
                'total_graphs': total,
                'successful_graphs': success_count,
                'total_samples': len(all_samples),
                'output_file': output_file,
                'sequence_stats': {
                    'avg_length': avg_length if all_samples else 0,
                    'max_length': max_length if all_samples else 0,
                    'min_length': min_length if all_samples else 0
                } if all_samples else {}
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =============================================================================
# 🧪 测试和演示代码  
# =============================================================================

def test_initialization_phase():
    """Test 3-step initialization phase"""
    print("🧪 Test Initialization Phase")
    print("=" * 40)
    
    # 创建简单的Test graph
    graph_data = {
        'edges': [(0, 1), (1, 2)],
        'nodes': [0, 1, 2],
        'source': 0,
        'target': 2
    }
    
    solver = MaxFlowSolver()
    
    # Initial state
    state = solver._convert_from_loader(graph_data)
    print("Initial state:")
    print(f"- Edges: {[(e.u, e.v, e.slot1, e.slot2) for e in state.edges]}")
    print(f"- Nodes: {[(n.node_id, n.level, n.parent, n.is_initialized) for n in state.nodes]}")
    
    # Step 1: INIT_SLOT1_E
    print(f"\nStep 1: INIT_SLOT1_E")
    print(f"Guard check: {solver._has_edges_or_nodes_without_any_slots(state)}")
    if solver._has_edges_or_nodes_without_any_slots(state):
        state = solver._op_init_slot1_e(state)
        print(f"After execution:")
        print(f"- Edges: {[(e.u, e.v, e.slot1, e.slot2) for e in state.edges]}")
        print(f"- Nodes: {[(n.node_id, n.level, n.parent, n.is_initialized) for n in state.nodes]}")
    
    # Step 2: INIT_SLOT2_E_AND_Y
    print(f"\nStep 2: INIT_SLOT2_E_AND_Y")
    print(f"Guard check: {solver._has_edges_or_nodes_with_one_mask(state)}")
    if solver._has_edges_or_nodes_with_one_mask(state):
        state = solver._op_init_slot2_e_and_y(state)
        print(f"After execution:")
        print(f"- Edges: {[(e.u, e.v, e.slot1, e.slot2) for e in state.edges]}")
        print(f"- Nodes: {[(n.node_id, n.level, n.parent, n.is_initialized) for n in state.nodes]}")
    
    # Step 3: INIT_COMMIT_Y
    print(f"\nStep 3: INIT_COMMIT_Y")
    print(f"Guard check: {solver._has_nodes_with_lvl_and_mask_par(state)}")
    if solver._has_nodes_with_lvl_and_mask_par(state):
        state = solver._op_init_commit_y(state)
        print(f"After execution:")
        print(f"- Edges: {[(e.u, e.v, e.slot1, e.slot2, e.is_initialized) for e in state.edges]}")
        print(f"- Nodes: {[(n.node_id, n.level, n.parent, n.is_initialized) for n in state.nodes]}")
    
    # Verify initialization complete
    print(f"\n✅ Initialization check:")
    all_edges_init = all(edge.is_initialized for edge in state.edges)
    all_nodes_init = all(node.is_initialized for node in state.nodes)
    print(f"- All edges initialized: {all_edges_init}")
    print(f"- All nodes initialized: {all_nodes_init}")
    src_node = state.get_node_by_id(state.src_node)
    print(f"- Source level: {src_node.level if src_node else 'None'}")
    print(f"- Other nodes level: {[n.level for n in state.nodes if n.node_id != state.src_node]}")
    
    return state

def test_bfs_phase():
    """Test BFS phase operations"""
    print("🧪 Test BFS Phase")
    print("=" * 40)
    
    # Start from initialized state
    state = test_initialization_phase()
    solver = MaxFlowSolver()
    
    print(f"\nBFS test start:")
    print(f"- Current max level: {solver._get_max_level(state)}")
    print(f"- Source: {state.src_node}, Target: {state.tgt_node}")
    
    # Simulate BFS rounds
    for round_num in range(3):
        print(f"\n--- BFS round {round_num + 1} ---")
        
        # Check if has relaxable edges
        current_level = solver._get_max_level(state)
        has_relaxable = solver._has_relaxable_edges_at_level(state, current_level)
        print(f"Current level: {current_level}, has relaxable: {has_relaxable}")
        
        if has_relaxable:
            # BFS_LAYER_R
            print(f"Execute BFS_LAYER_R:")
            state = solver._op_bfs_layer_r(state)
            print(f"- Node state: {[(n.node_id, n.level, n.parent) for n in state.nodes]}")
            
            # BFS_LAYER_Y
            if solver._has_nodes_with_mask_slots(state):
                print(f"Execute BFS_LAYER_Y:")
                state = solver._op_bfs_layer_y(state)
                print(f"- Node state: {[(n.node_id, n.level, n.parent) for n in state.nodes]}")
        else:
            print("No relaxable edges, BFS ends")
            break
        
        # Check if target reachable
        target_node = state.get_node_by_id(state.tgt_node)
        if target_node and target_node.level >= 0:
            print(f"🎯 Target node reachable! level={target_node.level}")
            break
    
    return state

def test_augment_phase():
    """Test augmentation phase operations"""
    print("🧪 Test Augmentation Phase")
    print("=" * 40)
    
    # Start from BFS completed state
    state = test_bfs_phase()
    solver = MaxFlowSolver()
    
    print(f"\nAugmentation test start:")
    target_node = state.get_node_by_id(state.tgt_node)
    print(f"- Target reachable: {target_node.level >= 0 if target_node else False}")
    print(f"- Target level: {target_node.level if target_node else 'None'}")
    
    # Show edge state before augmentation
    print(f"\nEdge state before aug:")
    for i, edge in enumerate(state.edges):
        print(f"  Edge {i}: ({edge.u}, {edge.v}) {edge.slot1}→{edge.slot2}")
    
    # Execute augmentation
    if solver._is_target_reachable(state):
        print(f"\nExecute AUG_FLIP_AND_RESET:")
        state = solver._op_aug_flip_and_reset(state)
        
        print(f"Edge state after aug:")
        for i, edge in enumerate(state.edges):
            print(f"  Edge {i}: ({edge.u}, {edge.v}) {edge.slot1}→{edge.slot2}")
        
        print(f"Node state after aug:")
        for node in state.nodes:
            print(f"  Node {node.node_id}: level={node.level}, parent={node.parent}")
    
    return state

def test_complete_solver():
    """Test complete solver pipeline"""
    # Create simple test graph
    graph_data = {
        'edges': [(0, 1), (1, 2)],
        'nodes': [0, 1, 2],
        'source': 0,
        'target': 2
    }
    
    solver = MaxFlowSolver()
    current_state = solver._convert_from_loader(graph_data)
    
    print(f"🧪 Test graph: source{graph_data['source']} → target{graph_data['target']}")
    print(f"Edges: {graph_data['edges']}")
    
    # Execute solve
    result = solver.solve(graph_data)
    print(f"✅ Solve complete: {result}")
    
    return current_state


# =============================================================================
# 🎯 主函数 - 类似sudoku_generator的使用方式
# =============================================================================

def generate_custom_dataset(num_instances: int = 100,
                           min_nodes: int = 10, max_nodes: int = 15,
                           min_edges: int = 30, max_edges: int = 50,
                           min_flow: int = 3, max_flow: int = 5,
                           output_file: str = "custom_maxflow.pkl.gz",
                           seed: int = None):
    """Generate dataset with custom parameters"""
    print(f"🎯 Generate dataset: {num_instances} graphs, nodes {min_nodes}-{max_nodes}, "
          f"edges {min_edges}-{max_edges}, flow {min_flow}-{max_flow} -> {output_file}")
    
    result = MaxFlowDataManager().generate_complete_dataset(
        output_file=output_file, num_instances=num_instances,
        min_nodes=min_nodes, max_nodes=max_nodes,
        min_edges=min_edges, max_edges=max_edges,
        min_flow=min_flow, max_flow=max_flow, seed=seed
    )
    
    print(f"{'✅ Success' if result['success'] else '❌ Failed'}: "
          f"{result.get('total_samples', result.get('error'))}")
    return result


def main():
    """Main function - Generate MaxFlow APMDM dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MaxFlow APMDM Dataset Generator")
    
    args_config = [
        ("--output", {"type": str, "default": "maxflow.pkl.gz", "help": "Output file path"}),
        ("--num_instances", {"type": int, "default": 1000, "help": "Number of graph instances"}),
        ("--seed", {"type": int, "default": 42, "help": "Random seed"}),
        ("--min_nodes", {"type": int, "default": 10, "help": "Min nodes"}),
        ("--max_nodes", {"type": int, "default": 10, "help": "Max nodes"}),
        ("--min_edges", {"type": int, "default": 50, "help": "Min edges"}),
        ("--max_edges", {"type": int, "default": 50, "help": "Max edges"}),
        ("--min_flow", {"type": int, "default": 1, "help": "Min guaranteed flow"}),
        ("--max_flow", {"type": int, "default": 100, "help": "Max guaranteed flow"}),
        ("--format", {"type": str, "default": "pkl", "choices": ["pkl", "jsonl"], "help": "Output format"}),
        ("--from_file", {"type": str, "default": None, "help": "Generate from JSON file"}),
        ("--demo", {"action": "store_true", "help": "Run demo mode"}),
        ("--batch_size", {"type": int, "default": 20, "help": "Batch size"}),
    ]
    
    for arg_name, kwargs in args_config:
        parser.add_argument(arg_name, **kwargs)
    
    args = parser.parse_args()
    print("🎯 MaxFlow APMDM Dataset Generator\n" + "=" * 50)
    
    manager = MaxFlowDataManager()
    
    if args.from_file:
        print(f"📁 Generate from file: {args.from_file}")
        result = manager.generate_from_file(args.from_file, args.output)
        print(f"{'✅ Success' if result['success'] else '❌ Failed'}: {result.get('total_samples', result.get('error'))}")
        return
    
    print(f"📊 Config: {args.num_instances} graphs, nodes {args.min_nodes}-{args.max_nodes}, "
          f"edges {args.min_edges}-{args.max_edges}, flow {args.min_flow}-{args.max_flow}, output {args.output}")
    
    result = manager.generate_complete_dataset(
        output_file=args.output, num_instances=args.num_instances,
        min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        min_edges=args.min_edges, max_edges=args.max_edges,
        min_flow=args.min_flow, max_flow=args.max_flow, seed=args.seed
    )
    
    if result['success']:
        stats = result
        print(f"✅ Success! Graphs: {stats['successful_instances']}/{stats['total_instances']} "
              f"Samples: {stats['total_samples']:,} Avg: {stats['avg_samples_per_instance']:.1f} Output: {stats['output_file']}")
        
        if 'sequence_stats' in stats and stats['sequence_stats']:
            seq_stats = stats['sequence_stats']
            print(f"📏 Seq length: avg {seq_stats['avg_length']:.1f} "
                  f"max {seq_stats['max_length']} min {seq_stats['min_length']}")
    else:
        print(f"❌ Failed: {result['error']}")


if __name__ == "__main__":
    main()