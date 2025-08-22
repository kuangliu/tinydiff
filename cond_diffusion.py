import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import pairwise
from smalldiffusion import TreeDataset
from torch.utils.data import Dataset, DataLoader

from diffusion import plot_batch, get_sigma_embs, generate_sample, NoiseScheduler


class CondLabelEmbedding(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes+1, hidden_size)
        self.no_cond = num_classes
        self.dropout = dropout

    def forward(self, labels):
        if self.training:
            dropped = torch.rand(len(labels)) < self.dropout
            labels[dropped] = self.no_cond
        return self.emb(labels)


class ConditionalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [8, 16, 128, 128, 128, 128, 16]
        layers = []
        for input_dim, output_dim in pairwise(dims):
            layers.extend([nn.Linear(input_dim, output_dim), nn.GELU()])
        layers.append(nn.Linear(dims[-1], 2))
        self.layers = nn.Sequential(*layers)
        self.cond_emb = CondLabelEmbedding(hidden_size=4, num_classes=27)

    def predict_eps(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        uncond = torch.full_like(cond, self.cond_emb.no_cond)
        eps_cond, eps_uncond = self.forward(torch.cat([x, x]), sigma, torch.cat([cond, uncond])).chunk(2)
        return eps_cond + 1.0 * (eps_cond - eps_uncond)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (torch.tensor) input tensor, sized [N,2].
          sigma: (torch.tensor) input noise level, sized [N,].
          condition: (torch.tensor) input condition labels, sized [N,].
        """
        sigma_embs = get_sigma_embs(sigma)  # [N,2]
        cond_embs = self.cond_emb(condition)  # [N,4]
        inputs = torch.cat([x, sigma_embs, cond_embs], dim=1)  # [N,8]
        outputs = self.layers(inputs)
        return outputs


@torch.no_grad()
def ddim_cond_samples(model: ConditionalMLP,
                      sigmas: torch.Tensor,
                      cond: torch.Tensor,
                      batch_size: int = 1):
    model.eval()
    xt = torch.randn(batch_size, 2) * sigmas[0]
    for (sig, sig_prev) in pairwise(sigmas):
        eps = model.predict_eps(xt, sig.repeat(2*len(xt)), cond)
        xt = xt - (sig - sig_prev) * eps
        yield xt


# Prepare data
dataset = TreeDataset(branching_factor=3, depth=3, num_samples_per_path=50)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

model = ConditionalMLP()
noise_scheduler = NoiseScheduler(sigma_min=0.01, sigma_max=10.0, T=200)

# num_epochs = 30000
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# model.train()
# for epoch in range(num_epochs):
#     print(f"Epoch: {epoch}")
#     for x0, cond in dataloader:
#         optimizer.zero_grad()
#         x0, sigma, eps = generate_sample(x0, noise_scheduler)
#         y = model(x0 + sigma[:, None] * eps, sigma, cond)
#         loss = criterion(y, eps)
#         loss.backward()
#         optimizer.step()
#         print(f"Loss: {loss.item():.2f}")
#     scheduler.step()
# torch.save(model.state_dict(), "./checkpoint/model.pth")

# Sampling
model.load_state_dict(torch.load("./checkpoint/model.pth"))
num_classes = 27
batch_size = 100
for c in range(num_classes):
    *_, x0 = ddim_cond_samples(model, noise_scheduler.sample_sigmas(20),
                               cond=torch.tensor([c] * batch_size), batch_size=batch_size)
    plot_batch(x0)
plt.savefig("z.png")
