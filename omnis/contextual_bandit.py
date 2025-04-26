import numpy as np

class ContextualBandit:
    def __init__(self, action_space, causal_model=None):
        self.action_space = action_space
        self.causal_model = causal_model
        self.exploration_rate = 0.1  # Initial exploration rate
        self.learning_rate = 0.1
        self.action_values = {}  # Store Q-values for state-action pairs
        
    def select_action(self, context):
        """Select action using epsilon-greedy strategy with causal influence."""
        try:
            # Exploration
            if np.random.random() < self.exploration_rate:
                return np.random.randint(0, len(self.action_space["quantization_levels"]))
            
            # Exploitation with causal influence
            state_key = self._get_state_key(context)
            if state_key not in self.action_values:
                self.action_values[state_key] = np.zeros(len(self.action_space["quantization_levels"]))
            
            # Get base Q-values
            q_values = self.action_values[state_key]
            
            # Incorporate causal knowledge if available
            if self.causal_model is not None:
                causal_effects = self.causal_model.estimate_effect(context["snr"])
                if causal_effects is not None:
                    q_values = q_values + 0.2 * causal_effects  # Weighted combination
            
            return np.argmax(q_values)
        
        except Exception as e:
            print(f"Error in select_action: {e}")
            return np.random.randint(0, len(self.action_space["quantization_levels"]))
    
    def update(self, context, action, reward):
        """Update Q-values based on received reward."""
        try:
            state_key = self._get_state_key(context)
            if state_key not in self.action_values:
                self.action_values[state_key] = np.zeros(len(self.action_space["quantization_levels"]))
            
            # Simple Q-learning update
            current_q = self.action_values[state_key][action]
            reward_value = reward["accuracy"] - 0.3 * reward["delay"] - 0.3 * reward["energy"]
            self.action_values[state_key][action] += self.learning_rate * (reward_value - current_q)
            
            # Update exploration rate (decay)
            self.exploration_rate *= 0.995
            
            # Update causal model if available
            if self.causal_model is not None:
                self.causal_model.add_observation({
                    "SNR": context["snr"],
                    "Action": action,
                    "Reward": reward_value
                })
                
        except Exception as e:
            print(f"Error in update: {e}")
    
    def _get_state_key(self, context):
        """Convert continuous context to discrete state key."""
        snr_bin = int(context["snr"] * 10) // 10
        delay_bin = int(context["delay_constraint"] * 100) // 10
        energy_bin = int(context["energy_constraint"] * 100) // 10
        return f"{snr_bin}_{delay_bin}_{energy_bin}"