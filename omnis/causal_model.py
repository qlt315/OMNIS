# omnis/causal_model.py
import numpy as np
import networkx as nx
from scipy.stats import beta
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class CausalModel:
    def __init__(self):
        self.observations = []
        
        # Dynamic weights based on context
        self.base_weights = {
            'accuracy': 2.5,
            'delay': 0.2,
            'energy': 0.2
        }
        
        # Performance tracking with SNR binning
        self.model_stats = {}
        self.snr_bins = np.arange(0, 25, 5)  # 0-20dB in 5dB steps
        
        self.obs_array = None  # Cache for observations
        self.last_update = 0
        
    def _update_obs_array(self):
        """Convert observations to numpy array for faster computation"""
        self.obs_array = np.array([
            [obs["SNR"], obs["Action"], obs["Accuracy"], 
             obs["Delay"], obs["Energy"]] 
            for obs in self.observations
        ])
        self.last_update = len(self.observations)
    
    def update_weights(self, snr):
        """Dynamically adjust weights based on SNR regime"""
        if snr < 10:  # Low SNR regime
            self.accuracy_weight = self.base_weights['accuracy'] * 1.2
            self.delay_weight = self.base_weights['delay'] * 0.8
        else:  # High SNR regime
            self.accuracy_weight = self.base_weights['accuracy']
            self.delay_weight = self.base_weights['delay']
            
    def estimate_effect(self, snr):
        """Vectorized effect estimation"""
        if not self.observations:
            return None
            
        # Update cached array if needed
        if len(self.observations) != self.last_update:
            self._update_obs_array()
            
        self.update_weights(snr)
        
        # Vectorized SNR filtering
        mask = np.abs(self.obs_array[:, 0] - snr) < 2.0
        if not np.any(mask):
            return None
            
        filtered_obs = self.obs_array[mask]
        effects = np.zeros(6)
        counts = np.zeros(6)
        
        # Vectorized reward computation
        for action in range(6):
            action_mask = filtered_obs[:, 1] == action
            if np.any(action_mask):
                action_obs = filtered_obs[action_mask]
                effects[action] = np.mean(
                    self.accuracy_weight * action_obs[:, 2] -
                    self.delay_weight * action_obs[:, 3] -
                    self.energy_weight * action_obs[:, 4]
                )
                counts[action] = np.sum(action_mask)
        
        counts[counts == 0] = 1
        return effects / counts

    def add_observation(self, observation):
        """Optimized observation addition"""
        if len(self.observations) % 100 == 0:  # Pre-allocate in chunks
            self.observations.extend([None] * 100)
        
        idx = len(self.observations)
        self.observations[idx] = {
            "SNR": observation["SNR"],
            "Action": observation["Action"],
            "Accuracy": observation["Accuracy"],
            "Delay": observation["Delay"],
            "Energy": observation["Energy"]
        }
        
    def analyze_causal_impact(self):
        """Analyze causal impact of different factors"""
        impacts = {
            'SNR': [],
            'Model': [],
            'CodingRate': []
        }
        
        for obs in self.observations:
            snr_range = obs['SNR'] // 5 * 5  # Group by 5dB ranges
            model = obs['Action']
            
            if snr_range not in self.model_stats:
                self.model_stats[snr_range] = {
                    'models': {},
                    'total_obs': 0
                }
                
            if model not in self.model_stats[snr_range]['models']:
                self.model_stats[snr_range]['models'][model] = {
                    'accuracy': [],
                    'delay': [],
                    'energy': []
                }
                
            stats = self.model_stats[snr_range]['models'][model]
            stats['accuracy'].append(obs['Accuracy'])
            stats['delay'].append(obs['Delay']) 
            stats['energy'].append(obs['Energy'])
            
        return self.model_stats

    def analyze_performance(self):
        """Parallel performance analysis"""
        def process_snr_range(snr_range):
            snr_min, snr_max = snr_range
            mask = (self.obs_array[:, 0] >= snr_min) & (self.obs_array[:, 0] < snr_max)
            regime_obs = self.obs_array[mask]
            
            if len(regime_obs) > 0:
                return (f"{snr_min}-{snr_max}dB", {
                    "accuracy_mean": np.mean(regime_obs[:, 2]),
                    "delay_mean": np.mean(regime_obs[:, 3]),
                    "energy_mean": np.mean(regime_obs[:, 4])
                })
            return None

        # Update cached array
        if len(self.observations) != self.last_update:
            self._update_obs_array()
        
        # Generate SNR ranges
        snr_ranges = [(snr_min, snr_min + 5) for snr_min in self.snr_bins[:-1]]
        
        # Parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_snr_range, snr_ranges)
        
        return {k: v for r in results if r for k, v in [r]}
