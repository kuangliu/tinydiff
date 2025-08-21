import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import pairwise
from torch.utils.data import Dataset, DataLoader


class DatasaurusDozen(Dataset):
    def __init__(self, csv_file: str, dataset: str):
        df = pd.read_csv(csv_file, delimiter="\t")
        points = df[df["dataset"] == dataset][["x", "y"]]
        points = (points.to_numpy() - 50.) / 50.
        self.points = torch.from_numpy(points).float()

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.points[i % len(self.points)]

    def __len__(self) -> int:
        return len(self.points) * 15


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

    def sample_sigmas(self, steps: int) -> torch.Tensor:
        indices = list((len(self) * (1 - np.arange(0, steps)/steps))
                       .round().astype(np.int64) - 1)
        return self[indices + [0]]


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [4, 16, 128, 128, 128, 128, 16]
        layers = []
        for input_dim, output_dim in pairwise(dims):
            layers.extend([nn.Linear(input_dim, output_dim), nn.GELU()])
        layers.append(nn.Linear(dims[-1], 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (torch.tensor) input tensor, sized [N,2].
          sigma: (torch.tensor) input noise level, sized [N,].
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


@torch.no_grad()
def samples(model: nn.Module,
            sigmas: torch.Tensor,
            gam: float = 1.,
            mu: float = 0.,
            batch_size: int = 1):
    """Generalizes most commonly-used samplers:
      DDPM       : gam=1, mu=0.5
      DDIM       : gam=1, mu=0
      Accelerated: gam=2, mu=0
    """
    model.eval()
    xt = torch.randn(batch_size, 2) * sigmas[0]
    eps = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps_prev, eps = eps, model(xt, sig.repeat(len(xt)))
        eps_av = eps * gam + eps_prev * (1-gam) if i > 0 else eps
        sig_p = (sig_prev/sig**mu)**(1/(1-mu))
        eta = (sig_prev**2 - sig_p**2).sqrt()
        xt = xt - (sig - sig_p) * eps_av + eta * torch.randn(len(xt), 2)
        yield xt


@torch.no_grad()
def ddim_samples(model: nn.Module,
                 sigmas: torch.Tensor,
                 gam: float = 1.,
                 mu: float = 0.,
                 batch_size: int = 1):
    model.eval()
    xt = torch.randn(batch_size, 2) * sigmas[0]
    for (sig, sig_prev) in pairwise(sigmas):
        eps = model(xt, sig.repeat(len(xt)))
        xt = xt - (sig - sig_prev) * eps
        yield xt


def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:, 0], batch[:, 1], marker=".")


# Prepare data
dataset = DatasaurusDozen(csv_file="./data/DatasaurusDozen.tsv", dataset="dino")
dataloader = DataLoader(dataset, batch_size=len(dataset))

# Noise scheduler
noise_scheduler = NoiseScheduler(sigma_min=0.01, sigma_max=10.0, T=200)

# Define model
model = SimpleMLP()

# Train
# num_epochs = 15000
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# model.train()
# for epoch in range(num_epochs):
#     print(f"Epoch: {epoch}")
#     for x0 in dataloader:
#         optimizer.zero_grad()
#         x0, sigma, eps = generate_sample(x0, noise_scheduler)
#         y = model(x0 + sigma[:, None] * eps, sigma)
#         loss = criterion(y, eps)
#         loss.backward()
#         optimizer.step()
#         print(f"Loss: {loss.item():.2f}")
#     scheduler.step()
# torch.save(model.state_dict(), "./checkpoint/model.pth")

# Sampling
model.load_state_dict(torch.load("./checkpoint/model.pth"))
*xts, x0 = ddim_samples(model, noise_scheduler.sample_sigmas(20), batch_size=1500, gam=2, mu=0)
print(x0.shape)
x0 = x0.detach().numpy()
plt.scatter(x0[:, 0], x0[:, 1], marker='.')
plt.savefig("z.png")
