import torch
import torch.nn as nn
import torch.nn.functional as F



# 1. Device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)



# 2. Settings
batch_size = 4
block_size = 16
embedding_size = 32



# 3. Single Attention Head
class Head(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()

        # Convert input embeddings into Queries
        self.query = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        # Convert input embeddings into Keys
        self.key = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        # Convert input embeddings into Values
        self.value = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False
        )

        # Causal mask: tokens cannot look into the future
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

    def forward(self, x):

        # x shape:
        # [batch_size, block_size, embedding_size]

        B, T, C = x.shape

        # Generate Queries, Keys and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        attention_scores = Q @ K.transpose(-2, -1)

        # Scale scores
        attention_scores = attention_scores / (C ** 0.5)

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        # Weighted combination of values
        output = attention_weights @ V

        return output



# 4. Test the Head
x = torch.randn(
    batch_size,
    block_size,
    embedding_size,
    device=device
)

head = Head(embedding_size).to(device)

output = head(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)