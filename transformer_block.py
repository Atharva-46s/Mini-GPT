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
number_of_heads = 4


 
# 3. Single Attention Head
 
class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.query = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_size,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = Q @ K.transpose(-2, -1)

        attention_scores = attention_scores / (
            K.shape[-1] ** 0.5
        )

        attention_scores = attention_scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        output = attention_weights @ V

        return output


 
# 4. Multi-Head Attention
 
class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_size, number_of_heads):

        super().__init__()

        head_size = embedding_size // number_of_heads

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(number_of_heads)
        ])

        self.projection = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        head_outputs = [
            head(x)
            for head in self.heads
        ]

        combined_output = torch.cat(
            head_outputs,
            dim=-1
        )

        output = self.projection(combined_output)

        return output


 
# 5. Feed-Forward Network
 
class FeedForward(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_size,
                4 * embedding_size
            ),

            nn.ReLU(),

            nn.Linear(
                4 * embedding_size,
                embedding_size
            )
        )

    def forward(self, x):

        return self.network(x)


 
# 6. Complete Transformer Block
 
class TransformerBlock(nn.Module):

    def __init__(self, embedding_size, number_of_heads):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_size,
            number_of_heads
        )

        self.feed_forward = FeedForward(
            embedding_size
        )

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        # Normalize before attention
        normalized_x = self.layer_norm_1(x)

        # Attention + residual connection
        x = x + self.attention(normalized_x)

        # Normalize before feed-forward network
        normalized_x = self.layer_norm_2(x)

        # Feed-forward + residual connection
        x = x + self.feed_forward(normalized_x)

        return x


 
# 7. Test Transformer Block
 
x = torch.randn(
    batch_size,
    block_size,
    embedding_size,
    device=device
)

transformer_block = TransformerBlock(
    embedding_size,
    number_of_heads
).to(device)

output = transformer_block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)