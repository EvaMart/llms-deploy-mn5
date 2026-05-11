import torch
print(f"MPS backend is available: {torch.backends.mps.is_available()}")
print(f"MPS backend is built: {torch.backends.mps.is_built()}")

# CPU basics
print("Num threads:", torch.get_num_threads())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)