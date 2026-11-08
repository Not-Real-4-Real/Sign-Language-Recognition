import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size=150,
        hidden_size=128,
        num_layers=2,
        num_classes=15,
        dropout=0.2,
        bidirectional=False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Recurrent feature extractor
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # Classification head
        directions = 2 if bidirectional else 1

        self.classifier = nn.Linear(
            hidden_size * directions,
            num_classes
        )

    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, input_size)
            lengths:  (B,)

        Returns:
            logits: (B, num_classes)
        """

        # Pack sequences to exclude padded frames
        packed = pack_padded_sequence(
            features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Run the GRU
        _, h_n = self.gru(packed)

        # Extract the final hidden state
        if self.bidirectional:
            # Top layer's forward and backward hidden states
            final_hidden = torch.cat(
                (h_n[-2], h_n[-1]),
                dim=1
            )
        else:
            # Top layer's final hidden state
            final_hidden = h_n[-1]

        # Map sequence representation to class logits
        logits = self.classifier(final_hidden)

        return logits

# Create a sample batch
B = 3
T = 5
F = 150

features = torch.randn(B, T, F)

lengths = torch.tensor([3, 5, 2])

# Create model
model = GRUClassifier(
    input_size=150,
    hidden_size=128,
    num_layers=2,
    num_classes=15,
    dropout=0.2
)

# Forward pass
logits = model(features, lengths)

print("Input shape:", features.shape)
print("Logits shape:", logits.shape)

# Parameter count
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)