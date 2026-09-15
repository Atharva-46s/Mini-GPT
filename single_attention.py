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


 
# 3. Example embeddings

# Simulates token + positional embeddings
x = torch.randn(
    batch_size,
    block_size,
    embedding_size,
    device=device
)

print("Input shape:", x.shape)


 
# 4. Create Query, Key and Value layers
 
query_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False
).to(device)

key_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False
).to(device)

value_layer = nn.Linear(
    embedding_size,
    embedding_size,
    bias=False
).to(device)


 
# 5. Generate Q, K and V
 
Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)


 
# 6. Calculate attention scores
 
attention_scores = Q @ K.transpose(-2, -1)

print("Attention scores shape:", attention_scores.shape)


 
# 7. Scale attention scores
 
attention_scores = attention_scores / (embedding_size ** 0.5)


 
# 8. Apply causal mask
 
mask = torch.tril(
    torch.ones(block_size, block_size, device=device)
)

attention_scores = attention_scores.masked_fill(
    mask == 0,
    float("-inf")
)


 
# 9. Convert scores into probabilities
 
attention_weights = F.softmax(
    attention_scores,
    dim=-1
)

print("Attention weights shape:", attention_weights.shape)


 
# 10. Combine information from other tokens
 
output = attention_weights @ V

print("Attention output shape:", output.shape)