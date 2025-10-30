from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RLResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + x)

class LSTMCNN(BaseFeaturesExtractor):
    
    #CNN + LSTM feature extractor.
    #takes a sequence of frames and extracts spatial + temporal features.
    
    def __init__(self, observation_space, features_dim=512, lstm_hidden_size=256, num_lstm_layers=1):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # CNN Encoder - spatial feature extractor per frame
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            RLResBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            RLResBlock(128),
            nn.Flatten(),
        )

        # Determine CNN output dimension dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        # LSTM - temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=n_flatten,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # projection
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim)
        )

    def forward(self, observations):
        """
        observations: [batch_size, channels, height, width] 
                      OR [batch_size, seq_len, channels, height, width]
        """
        # if there is no sequence dimension, assume seq_len = 1
        if observations.dim() == 4:
            observations = observations.unsqueeze(1)  # -> [batch, seq, C, H, W]

        batch_size, seq_len, C, H, W = observations.shape

        # merge batch and seq dims for CNN processing
        cnn_input = observations.view(batch_size * seq_len, C, H, W)
        cnn_features = self.cnn(cnn_input)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # [B, T, F]

        # pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # [B, T, hidden]
        last_output = lstm_out[:, -1, :]  # use last timestep feature

        return self.linear(last_output)