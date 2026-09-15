import torch

# Select GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)