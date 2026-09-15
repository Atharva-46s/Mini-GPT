import torch
import torch.nn as nn
 
# 1. Device
 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


 
# 2. Example input
 
x = torch.randn(
    4,
    16,
    32,
    device=device
)


 
# 3. Example transformation
 
transformation = nn.Linear(
    32,
    32
).to(device)

transformed_x = transformation(x)


 
# 4. Residual connection
 
output = x + transformed_x


 
# 5. Display shapes
 
print("Original input shape:", x.shape)
print("Transformed shape:", transformed_x.shape)
print("Residual output shape:", output.shape)
