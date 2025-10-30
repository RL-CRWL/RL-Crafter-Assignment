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

class ImprovedCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

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

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))


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
    
class CURLCNN(BaseFeaturesExtractor):
    
    #Contrastive Unsupervised Representation Learning (CURL)
    #CNN feature extractor for visual DQN.
    #Works as a drop-in replacement for ImprovedCNN.
    def __init__(self, observation_space, features_dim=512, projection_dim=128, temperature=0.07):
        super().__init__(observation_space, features_dim)
        self.temperature = temperature

        n_input_channels = observation_space.shape[0]
        self.encoder = nn.Sequential(
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

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.encoder(sample).shape[1]

        # Projector head for contrastive space
        self.projector = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

        # Main linear layer for DQN features
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim)
        )

    def augment(self, x):
        #Simple image augmentation: random crop + small noise
        # crop
        h, w = x.shape[-2:]
        crop_h, crop_w = int(h * 0.9), int(w * 0.9)
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        x = x[..., start_h:start_h + crop_h, start_w:start_w + crop_w]
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # add small Gaussian noise
        x = x + 0.01 * torch.randn_like(x)
        return x.clamp(0, 1)

    def contrastive_loss(self, z_a, z_pos):
        #Compute InfoNCE loss for contrastive learning
        z_a = F.normalize(z_a, dim=1)
        z_pos = F.normalize(z_pos, dim=1)
        logits = torch.matmul(z_a, z_pos.T) / self.temperature
        labels = torch.arange(z_a.size(0)).long().to(z_a.device)
        return F.cross_entropy(logits, labels)

    def forward(self, obs):

        #obs: [batch, channels, height, width]
        #Returns DQN feature vector; adds contrastive loss for self-supervision.

        # standard encoder
        x = self.encoder(obs)
        features = self.linear(x)

        # contrastive augmentation + loss (only during training)
        if self.training:
            obs_aug = self.augment(obs)
            x_aug = self.encoder(obs_aug)
            z_a = self.projector(x)
            z_pos = self.projector(x_aug)
            self.curl_loss = self.contrastive_loss(z_a, z_pos)
        else:
            self.curl_loss = torch.tensor(0.0, device=obs.device)

        return features