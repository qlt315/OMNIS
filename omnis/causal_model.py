# omnis/causal_model.py
import numpy as np

class CausalModel:
    def __init__(self):
        self.observations = []
        self.learning_rate = 0.1
        
        # Adjusted weights to scale rewards appropriately
        self.accuracy_weight = 2.5   # Increased for reward scaling
        self.delay_weight = 0.2      # Reduced penalty weight
        self.energy_weight = 0.2     # Reduced penalty weight
        
        # Keep tracking but don't apply strict thresholds
        self.best_models = {}      
        self.model_stats = {}      
        
    def add_observation(self, observation):
        """Add new observation with original values."""
        self.observations.append({
            "SNR": observation["SNR"],
            "Action": observation["Action"],
            "Accuracy": observation["Accuracy"],
            "Delay": observation["Delay"],
            "Energy": observation["Energy"]
        })
        
    def estimate_effect(self, snr):
        """Estimate causal effect with scaled reward calculation."""
        if not self.observations:
            return None
            
        similar_obs = [obs for obs in self.observations 
                      if abs(obs["SNR"] - snr) < 2.0]
        
        if not similar_obs:
            return None
            
        effects = np.zeros(6)
        counts = np.zeros(6)
        
        for obs in similar_obs:
            action = obs["Action"]
            # Modified reward calculation with penalties
            weighted_reward = (
                self.accuracy_weight * obs["Accuracy"] 
                - self.delay_weight * obs["Delay"]    # Convert to penalty
                - self.energy_weight * obs["Energy"]  # Convert to penalty
            )
            
            effects[action] += weighted_reward
            counts[action] += 1
            
            # Track good models without strict thresholds
            snr_key = f"{int(snr)}"
            if snr_key not in self.best_models:
                self.best_models[snr_key] = []
            self.best_models[snr_key].append(action)
        
        # Simple average without additional scaling
        counts[counts == 0] = 1
        return effects / counts
