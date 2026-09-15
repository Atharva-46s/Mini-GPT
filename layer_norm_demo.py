import torch
import torch.nn as nn

# 1. Device
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


 
# 2. Settings
 
batch_size = 4
block_size = 16
embedding_size = 32


 
# 3. Example input
 
x = torch.randn(
    batch_size,
    block_size,
    embedding_size,
    device=device
)


 
# 4. Create LayerNorm
 

layer_norm = nn.LayerNorm(
    embedding_size
).to(device)


 
# 5. Apply normalization
 

normalized_x = layer_norm(x)


 
# 6. Display shapes
 

print("Input shape:", x.shape)
print("Normalized shape:", normalized_x.shape)


 
# 7. Check statistics for one token
 

sample_token = normalized_x[0, 0]

print("\nMean of one normalized token:",
      sample_token.mean().item())

print("Standard deviation of one normalized token:",
      sample_token.std(unbiased=False).item())