import torch
import torch.nn as nn


 
# 1. Device
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


 
# 2. Settings
 
batch_size = 4
block_size = 16
embedding_size = 32


 
# 3. Feed-Forward Network
 
class FeedForward(nn.Module):

    def __init__(self, embedding_size):

        super().__init__()

        self.network = nn.Sequential(

            # Expand the representation
            nn.Linear(
                embedding_size,
                4 * embedding_size
            ),

            # Non-linear activation
            nn.ReLU(),

            # Compress back to original size
            nn.Linear(
                4 * embedding_size,
                embedding_size
            )
        )

    def forward(self, x):

        return self.network(x)


 
# 4. Test
 
x = torch.randn(
    batch_size,
    block_size,
    embedding_size,
    device=device
)

feed_forward = FeedForward(
    embedding_size
).to(device)

output = feed_forward(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)