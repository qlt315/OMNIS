import networkx as nx
import numpy as np
from typing import Dict, List

class CausalInference:
    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.effects_cache = {}
        
    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find backdoor paths between treatment and outcome"""
        paths = []
        for path in nx.all_simple_paths(self.G, treatment, outcome):
            if self._is_backdoor_path(path):
                paths.append(path)
        return paths
    
    def _is_backdoor_path(self, path: List[str]) -> bool:
        """Check if a path is a backdoor path"""
        return len(path) > 2 and self.G.has_edge(path[1], path[0])
    
    def estimate_effect(self, action: Dict[str, float], outcome: str) -> float:
        """Estimate causal effect of action on outcome"""
        total_effect = 0.0
        
        for node, value in action.items():
            # Direct effect
            if self.G.has_edge(node, outcome):
                total_effect += 0.3 * value  # Direct effect weight
                
            # Indirect effects through paths
            paths = list(nx.all_simple_paths(self.G, node, outcome))
            path_effect = sum(0.1 * value / len(path) for path in paths)  # Decay with path length
            total_effect += path_effect
            
        return total_effect
    
    def get_adjustment_set(self, treatment: str, outcome: str) -> List[str]:
        """Get minimal adjustment set for causal identification"""
        ancestors_treatment = nx.ancestors(self.G, treatment)
        ancestors_outcome = nx.ancestors(self.G, outcome)
        return list(ancestors_treatment & ancestors_outcome)