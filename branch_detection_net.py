# Insert code for branch detection task
import torch
import torch.nn as nn


# Very simple block to start with
class ResNetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return x + self.block(x)

# Very simple block to start with
class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


# ResNet structure to start with
class BranchDetector(nn.Module):
    def __init__(self, num_blocks: int, spatial_res: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.fc = nn.Linear(in_features=3 * spatial_res ** 2, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        y = self.fc(y.flatten(start_dim=1))
        return y

# Fucnction to initilise parameters of neural network
def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
