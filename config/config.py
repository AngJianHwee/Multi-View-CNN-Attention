import torch

# Configuration settings
batch_size = 64
num_epochs = 50
learning_rate = 1e-4
data_set_root = "./datasets"

# Device configuration
gpu_indx = 0
device = torch.device(gpu_indx if torch.cuda.is_available() else 'cpu')

# print out everything
print(f"Batch size: {batch_size}")
print(f"Number of epochs: {num_epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Dataset root: {data_set_root}")
print(f"Device: {device}")
