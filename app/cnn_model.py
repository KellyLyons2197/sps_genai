import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv2D: 16 filters, 3x3, stride 1, padding 1   -> 3x64x64 becomes 16x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # MaxPool 2x2, stride 2 (halves height and width)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv2D: 32 filters, 3x3, stride 1, padding 1   -> 16x32x32 becomes 32x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Fully connected: 32 channels * 16 * 16 = 8192 inputs -> 100 units
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        # Output layer: 10 classes
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x64x64  -> 16x64x64 -> 16x32x32
        x = self.pool(F.relu(self.conv2(x)))  # 16x32x32 -> 32x32x32 -> 32x16x16
        x = x.view(-1, 32 * 16 * 16)          # Flatten  -> 8192
        x = F.relu(self.fc1(x))               # 8192 -> 100
        x = self.fc2(x)                       # 100 -> 10
        return x


if __name__ == "__main__":
    # Quick check: a fake 64x64 image should give 10 outputs
    model = SimpleCNN()
    print(model)
    print(model(torch.randn(1, 3, 64, 64)).shape)  # torch.Size([1, 10])