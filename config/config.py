import torch

# Configuration settings
batch_size = 64
num_epochs = 50
learning_rate = 1e-4
data_set_root = "../../datasets"

# Device configuration
gpu_indx = 0
device = torch.device(gpu_indx if torch.cuda.is_available() else 'cpu')