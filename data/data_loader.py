import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

def get_data_loaders(batch_size=64, validation_split=0.1, dataset_root="./datasets"):
    # Define transform for both train and test (simplified for consistency)
    transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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