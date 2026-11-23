
import torch
import torch.nn as nn


class BM_AI_1M(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.position = nn.Embedding(256, 128)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=512,
                batch_first=True
            ),
            num_layers=4
        )

        self.output = nn.Linear(128, vocab_size)

    def forward(self, x):
        positions = torch.arange(
            x.size(1),
            device=x.device
        )

        x = self.embedding(x) + self.position(positions)

        mask = torch.triu(
            torch.ones(
                x.size(1),
                x.size(1),
                device=x.device
            ),
            diagonal=1
        ).bool()

        x = self.transformer(x, mask=mask)

        return self.output(x)
