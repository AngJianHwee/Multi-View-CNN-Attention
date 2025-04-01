import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from config.config import batch_size, num_epochs, learning_rate, data_set_root, device, reshape_size

def get_data_loaders(batch_size=64, validation_split=0.1, dataset_root="./datasets"):
    # Define transform for both train and test (simplified for consistency)
    transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.Resize((reshape_size, reshape_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = torchvision.datasets.CIFAR10(root=dataset_root, train=True, 
                                            download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=dataset_root, train=False, 
                                           download=True, transform=transform)

    # Check total dataset size
    total_samples = len(train_data)
    if total_samples == 0:
        raise ValueError("Training dataset is empty. Check dataset_root or download process.")

    # Calculate number of validation samples
    num_valid_samples = int(total_samples * validation_split)
    num_train_samples = total_samples - num_valid_samples
    print(f"Total training samples: {total_samples}, Validation samples: {num_valid_samples}")
    # Split dataset into training and validation
    train_data, valid_data = data.random_split(train_data, [num_train_samples, num_valid_samples])
    print(f"Training samples: {len(train_data)}, Validation samples: {len(valid_data)}")
    # Check if split was successful
    if len(train_data) + len(valid_data) != total_samples:
        raise ValueError("Dataset split error. Check the split sizes.")
    # Check if any dataset is empty
    if len(train_data) == 0 or len(valid_data) == 0:
        raise ValueError("One of the datasets is empty after splitting. Check split sizes.")
    # Check if batch size is valid
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if batch_size > len(train_data):
        raise ValueError("Batch size exceeds the number of training samples.")
    # Check if validation split is valid
    if validation_split <= 0 or validation_split >= 1:
        raise ValueError("Validation split must be between 0 and 1.")

    # Create data loaders
    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    # Print diagnostics
    print(f"Number of training examples: {len(train_data)}")
    print(f"Number of validation examples: {len(valid_data)}")
    print(f"Number of test examples: {len(test_data)}")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in valid loader: {len(valid_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    # Test the function standalone
    train_loader, valid_loader, test_loader = get_data_loaders()