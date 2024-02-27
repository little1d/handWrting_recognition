import torch
import torchvision
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data',train=True,download=True
                               ,transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.137,)(0.3081))
        ])),
    batch_size=