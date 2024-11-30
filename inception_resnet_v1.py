import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionResnetV1(nn.Module):
    def __init__(self, embedding_size=128):
        super(InceptionResnetV1, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.residual_block_1 = self._make_residual_block(128, 128)
        self.residual_block_2 = self._make_residual_block(128, 128)

        self.fc = nn.Linear(128, embedding_size)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)

        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        return F.normalize(x, p=2, dim=1)  # L2 Normalization
