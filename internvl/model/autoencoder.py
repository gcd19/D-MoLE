import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def compute_reconstruction_loss(self, embeddings):
        self.eval()
        with torch.no_grad():
            reconstruction = self(embeddings)
            losses = torch.mean((reconstruction - embeddings) ** 2, dim=1)
        return losses
