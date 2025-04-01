import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

def get_data_loaders(batch_size=64, validation_split=0.1, dataset_root="./datasets"):
    transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)

    n_train_examples = int(len(train_data) * (1 - validation_split))
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples], generator=torch.Generator().manual_seed(42))

    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader