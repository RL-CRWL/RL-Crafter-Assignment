# dueling_dqn.py
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CrafterCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Crafter with dueling architecture support
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # Stacked frames
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))

class DuelingDQN(DQN):
    """
    Dueling DQN implementation for Crafter
    Based on: "Dueling Network Architectures for Deep Reinforcement Learning"
    """
    
    def __init__(self, *args, **kwargs):
        # Use custom policy kwargs for dueling architecture
        dueling_kwargs = dict(
            features_extractor_class=CrafterCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, 128],  # Smaller network since we have feature extractor
        )
        
        # Merge with any existing policy kwargs
        policy_kwargs = kwargs.get('policy_kwargs', {})
        policy_kwargs.update(dueling_kwargs)
        kwargs['policy_kwargs'] = policy_kwargs
        
        super().__init__(*args, **kwargs)