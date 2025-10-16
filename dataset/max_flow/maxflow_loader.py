#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import networkx as nx
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple


def sample_nodes(n: int, max_id: int = 14, seed: int = None) -> List[int]:
    """Randomly sample from node pool"""
    rng = random.Random(seed)
    return sorted(rng.sample(range(max_id + 1), n))


def ensure_flow_by_patching(G: nx.DiGraph, s: int, t: int, k: int, 
                           rng: random.Random, max_iters: int = 50) -> List[Tuple[int, int]]:
    """Use min-cut patching to increase max-flow to ≥k
    
    Return list of newly added edges
    """
    added_edges = []
    
    def current_flow():
        try:
            return nx.maximum_flow_value(G, s, t)
        except:
            return 0
    
    f = current_flow()
    iterations = 0
    
    # If already satisfied, return directly
    if f >= k:
        return added_edges
    
    while f < k and iterations < max_iters:
        iterations += 1
        
        try:
            # Calculate minimum cut
            cut_value, (S, T) = nx.minimum_cut(G, s, t)
            
            # Ensure S contains s, T contains t
            if s not in S or t not in T:
                # 🔧 Fix: If cut incorrect, randomly add edges but avoid bidirectional edges
                all_missing = [(u, v) for u in G.nodes() for v in G.nodes() 
                              if u != v and not G.has_edge(u, v) and not G.has_edge(v, u)]
                if not all_missing:
                    break
                u, v = rng.choice(all_missing[:100])  # Limit candidate count
                # 🔧 Fix: Check again before adding edges to ensure no bidirectional edges
                if not G.has_edge(v, u):
                    G.add_edge(u, v)
                    added_edges.append((u, v))
                f = current_flow()
                continue
            
            # 🔧 Fix: Find cross-cut arc candidates, avoid bidirectional edges
            candidates = [(u, v) for u in S for v in T if not G.has_edge(u, v) and not G.has_edge(v, u)]
            
            if not candidates:
                # No cross-cut arcs to add, may have reached max flow
                break
            
            # 🔧 Fix: Randomly select a cross-cut arc, ensure no bidirectional edges
            u, v = rng.choice(candidates)
            if not G.has_edge(v, u):
                G.add_edge(u, v)
                added_edges.append((u, v))
            
            # Recalculate flow
            new_f = current_flow()
            if new_f == f:
                # Flow didn't increase, possible issue, stop
                break
            f = new_f
            
        except Exception as e:
            # If error occurs, stop patching
            break
    
    return added_edges


def add_random_edges(G: nx.DiGraph, target: int, rng: random.Random):
    """Random edge supplementation - 🔧 Fix: Avoid bidirectional edges"""
    current = G.number_of_edges()
    need = max(0, target - current)
    if need == 0:
        return
    
    nodes = list(G.nodes())
    # 🔧 Fix: Only generate one-way edge candidates, avoid bidirectional edges
    candidates = []
    for u in nodes:
        for v in nodes:
            if u != v and not G.has_edge(u, v) and not G.has_edge(v, u):  # Avoid bidirectional edges
                candidates.append((u, v))
    
    if candidates and need > 0:
        add_edges = rng.sample(candidates, min(need, len(candidates)))
        G.add_edges_from(add_edges)


def generate_graph(min_nodes: int, max_nodes: int, min_edges: int, max_edges: int,
                  min_flow: int, max_flow: int, max_node_id: int, seed: int) -> Dict[str, Any]:
    """Generate single graph - Use min-cut patching algorithm to ensure minimum flow requirement"""
    rng = random.Random(seed)
    
    n = rng.randint(min_nodes, max_nodes)
    k = rng.randint(min_flow, max_flow)
    target_edges = rng.randint(min_edges, max_edges)
    
    # 🔧 New: Retry mechanism to ensure generated graph meets minimum flow requirement
    max_retries = 10
    for retry in range(max_retries):
        # 1. Create basic random graph
        node_ids = sample_nodes(n, max_node_id, seed + retry)  # Use different seed for each retry
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        
        # 2. First add20-30random edges
        initial_edges = min(rng.randint(20, 30), target_edges)
        add_random_edges(G, initial_edges, rng)
        
        # 3. Select source and target nodes
        s, t = rng.sample(node_ids, 2)
        
        # 4. Ensure basic connectivity, then min-cut patching
        # If s and t not connected, first add direct connection
        if not nx.has_path(G, s, t):
            G.add_edge(s, t)
            patched_edges = [(s, t)]
        else:
            patched_edges = []
        
        # Then use min-cut patching to target flow
        additional_edges = ensure_flow_by_patching(G, s, t, k, rng)
        patched_edges.extend(additional_edges)
        
        # 5. Supplement edges to target total
        add_random_edges(G, target_edges, rng)
        
        # 🔧 New: Remove bidirectional edges to ensure graph directionality
        edges_to_remove = []
        edges_list = list(G.edges())
        for u, v in edges_list:
            if G.has_edge(v, u) and (v, u) not in edges_to_remove:
                # If bidirectional edges exist, randomly remove one
                if rng.random() < 0.5:
                    edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((v, u))
        
        G.remove_edges_from(edges_to_remove)
        # if len(edges_to_remove) > 0:
        #     print(f"🔧 Removed {len(edges_to_remove)} bidirectional edges")
        
        # 6. Calculate final max flow - Fixed version
        try:
            # 🔧 Key fix: set capacity of each edge to 1
            for u, v in G.edges():
                G[u][v]['capacity'] = 1
            max_flow = nx.maximum_flow_value(G, s, t)
        except Exception as e:
            print(f"Warning: max flow calculation failed: {e}")
            max_flow = 0  # Safer fallback
        
        # 🔧 New: Check if minimum flow requirement is met
        if max_flow >= min_flow:
            # Requirement met, return result
            return {
                'id': 0,  # Will be set outside
                'nodes': n,
                'edges': G.number_of_edges(),
                'node_list': node_ids,
                'edge_list': list(G.edges()),
                'source_node': s,
                'target_node': t,
                'ground_truth_max_flow': max_flow,
                'min_guaranteed_flow': k
            }
        # else:
        #     # Requirement not met, retry
        #     print(f"🔄 Retry{retry+1}: flow{max_flow} < {min_flow}，regenerate")
    
    # If all retries fail, return last result (even if flow insufficient)
    print(f"⚠️ Retry{max_retries}times still cannot meet flow requirement, return last result")
    return {
        'id': 0,  # Will be set outside
        'nodes': n,
        'edges': G.number_of_edges(),
        'node_list': node_ids,
        'edge_list': list(G.edges()),
        'source_node': s,
        'target_node': t,
        'ground_truth_max_flow': max_flow,
        'min_guaranteed_flow': k
    }


def generate_dataset(num_samples: int, min_nodes: int = 10, max_nodes: int = 15,
                    min_edges: int = 30, max_edges: int = 50, min_flow: int = 3,
                    max_flow: int = 5, max_node_id: int = 14, seed: int = None) -> Dict[str, Any]:
    """Generate dataset"""
    print(f"🎯 Generating {num_samples} graph samples...")
    print(f"📊 nodes: {min_nodes}-{max_nodes}, edges: {min_edges}-{max_edges}, flow: {min_flow}-{max_flow}")
    
    start_time = time.time()
    graphs = []
    
    for i in range(num_samples):
        if i % 20 == 0 and i > 0:
            print(f"  Progress: {i}/{num_samples}")
        
        try:
            graph = generate_graph(min_nodes, max_nodes, min_edges, max_edges,
                                 min_flow, max_flow, max_node_id,
                                 random.randint(0, 1000000) if seed is None else seed + i)
            graph['id'] = i
            graphs.append(graph)
        except Exception as e:
            print(f"Skipping graph {i}: {e}")
    
    generation_time = time.time() - start_time
    
    # Simple statistics
    flows = [g['ground_truth_max_flow'] for g in graphs]
    
    dataset = {
        'metadata': {
            'samples': len(graphs),
            'generation_time': round(generation_time, 2),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': 'min_cut_patching',
            'config': {
                'min_nodes': min_nodes, 'max_nodes': max_nodes,
                'min_edges': min_edges, 'max_edges': max_edges,
                'min_flow': min_flow, 'max_flow': max_flow,
                'seed': seed
            }
        },
        'data': graphs
    }
    
    # Print statistics
    print(f"✅ Complete: {len(graphs)} samples, time {generation_time:.2f}s")
    print(f"📈 Statistics:")
    print(f"  Average max flow: {sum(flows)/len(flows):.2f}")
    print(f"  Flow range: {min(flows)}-{max(flows)}")
    print(f"  Zero-flow graphs: {sum(1 for f in flows if f == 0)}")
    
    return dataset


def save_dataset(dataset: Dict[str, Any], filename: str):
    """Save dataset"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    file_size = len(json.dumps(dataset, ensure_ascii=False)) / (1024 * 1024)
    print(f"💾 Saved to {filename} ({file_size:.1f}MB)")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Simplified graph dataset generator')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--output', type=str, default='dataset.json', help='Output file')
    parser.add_argument('--min_nodes', type=int, default=10, help='Minimum nodes')
    parser.add_argument('--max_nodes', type=int, default=15, help='Maximum nodes')
    parser.add_argument('--min_edges', type=int, default=30, help='Minimum edges')
    parser.add_argument('--max_edges', type=int, default=50, help='Maximum edges')
    parser.add_argument('--min_flow', type=int, default=3, help='Minimum guaranteed flow')
    parser.add_argument('--max_flow', type=int, default=5, help='Maximum guaranteed flow')
    parser.add_argument('--max_node_id', type=int, default=14, help='Node pool max ID')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_dataset(
        args.samples, args.min_nodes, args.max_nodes,
        args.min_edges, args.max_edges, args.min_flow, args.max_flow,
        args.max_node_id, args.seed
    )
    
    # Save
    save_dataset(dataset, args.output)


if __name__ == "__main__":
    main()
