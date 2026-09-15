import torch
import torch.nn as nn

 
# 1. Device
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


 
# 2. Basic settings
 
vocab_size = 33
block_size = 16

# Size of the vector representing each token
embedding_size = 32


 
# 3. Token embedding
 
token_embedding_table = nn.Embedding(
    vocab_size,
    embedding_size
).to(device)


 
# 4. Positional embedding
 
position_embedding_table = nn.Embedding(
    block_size,
    embedding_size
).to(device)


 
# 5. Example input
# Four examples, each containing 16 token IDs
index = torch.randint(
    0,
    vocab_size,
    (4, block_size),
    device=device
)

print("\nInput shape:")
print(index.shape)


 
# 6. Convert tokens into vectors

token_embeddings = token_embedding_table(index)

print("\nToken embedding shape:")
print(token_embeddings.shape)


 
# 7. Create position IDs
 
positions = torch.arange(
    block_size,
    device=device
)

print("\nPosition IDs:")
print(positions)


 
# 8. Convert positions into vectors
 
position_embeddings = position_embedding_table(positions)

print("\nPosition embedding shape:")
print(position_embeddings.shape)


 
# 9. Combine token and position information
 
combined_embeddings = (
    token_embeddings + position_embeddings
)

print("\nCombined embedding shape:")
print(combined_embeddings.shape)