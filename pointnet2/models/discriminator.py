import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudDiscriminator(nn.Module):
    def __init__(self):
        super(PointCloudDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.max(dim=2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Logits
        return x