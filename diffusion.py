import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from itertools import pairwise
from torch.utils.data import Dataset, DataLoader


class DatasaurusDozen(Dataset):
    def __init__(self, csv_file: str, dataset: str):
        df = pd.read_csv(csv_file, delimiter="\t")
        points = df[df["dataset"] == dataset][["x", "y"]]
        points = (points.to_numpy() - 50.) / 50.
        self.points = torch.from_numpy(points)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.points[i % len(self.points)]

    def __len__(self) -> int:
        return len(self.points) * 50


class NoiseScheduler:
    def __init__(self,
                 sigma_min: float = 0.02,
                 sigma_max: float = 10.0,
                 T: int = 200):
        self.sigmas = torch.logspace(np.log10(sigma_min), np.log10(sigma_max), T)

    def __getitem__(self, i) -> torch.Tensor:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        return self[torch.randint(len(self), (batch_size,))]


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [4, 16, 128, 128, 16]
        layers = []
        for input_dim, output_dim in pairwise(dims):
            layers.extend([nn.Linear(input_dim, output_dim), nn.GELU()])
        layers.append(nn.Linear(dims[-1], 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, sigma):
        """
        Args:
          x: (torch.tensor) input tensor, sized [N,2].
          sigma: (torch.tensor) input noise, sized [N,].
        """
        sigma_embs = get_sigma_embs(sigma)  # [N,2]
        inputs = torch.cat([x, sigma_embs], dim=1)  # [N,4]
        outputs = self.layers(inputs)
        return outputs


def get_sigma_embs(sigma):
    """Embed sigmas sinusoidally.

    Args:
      sigma: (torch.Tensor) noise level, sized [N,].

    Returns:
      (torch.Tensor) sigma embeds as [sin(log(σ)/2), cos(log(σ)/2)], sized [N,2].
    """
    return torch.stack([(0.5 * sigma.log()).sin(), (0.5 * sigma.log()).cos()], dim=-1)


def generate_sample(x0, noise_scheduler):
    """Generate a batch of data.

    Args:
      x0: (torch.tensor) data tensor, sized [N,D].
      noise_scheduler: (NoiseScheduler) noise scheduler.

    Returns:
      (torch.tensor) x0: input data, sized [N,D].
      (torch.tensor) sigma: sampled noise level, sized [N,].
      (torch.tensor) eps: sampled gaussian noise, sized [N,D].
    """
    sigma = noise_scheduler.sample_batch(len(x0))
    eps = torch.randn_like(x0)
    return x0, sigma, eps


# Prepare data
dataset = DatasaurusDozen(csv_file="./data/DatasaurusDozen.tsv", dataset="dino")
dataloader = DataLoader(dataset, batch_size=143)

# Noise scheduler
noise_scheduler = NoiseScheduler(sigma_min=0.02, sigma_max=10.0, T=200)

# Define model
model = SimpleMLP()

# Train
num_epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
model.train()
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for x0 in dataloader:
        optimizer.zero_grad()
        x0, sigma, eps = generate_sample(x0.float(), noise_scheduler)
        y = model(x0 + sigma[:, None] * eps, sigma)
        loss = criterion(y, eps)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.2f}")
    scheduler.step()
