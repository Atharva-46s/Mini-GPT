import torch
import time

# Select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Selected device:", device)

# Create large matrices directly on the selected device
a = torch.randn(3000, 3000, device=device)
b = torch.randn(3000, 3000, device=device)

# Synchronize before timing GPU operations
if device == "cuda":
    torch.cuda.synchronize()

start_time = time.time()

# Matrix multiplication
c = torch.matmul(a, b)

# Wait until GPU finishes
if device == "cuda":
    torch.cuda.synchronize()

end_time = time.time()

print("Matrix A device:", a.device)
print("Matrix B device:", b.device)
print("Result device:", c.device)
print("Time taken:", round(end_time - start_time, 4), "seconds")

if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))
    print(
        "Allocated GPU memory:",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
        "MB"
    )