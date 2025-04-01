import torch
from config.config import batch_size, num_epochs, learning_rate, data_set_root, device
from data.data_loader import get_data_loaders  # Assuming this still exists
from models.model import ThreeViewCNN  # Updated import
from training.train import train  # Will need to update this too
from training.evaluate import evaluate  # Will need to update this too
from utils.visualization import plot_training_metrics, visualize_predictions
import os

def main():
    # check if data_set_root exists
    if not os.path.exists(data_set_root):
        os.makedirs(data_set_root)
    
    train_loader, valid_loader, test_loader = get_data_loaders(batch_size=batch_size, dataset_root=data_set_root)
    
    # Initialize ThreeViewCNN with 3 channels for each view (CIFAR-10 has 3 channels)
    model = ThreeViewCNN(channels_ins=[3, 3, 3], output_dim=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()
    
    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []

    for epoch in range(num_epochs):
        # Train with three identical inputs
        model, optimizer, training_loss_logger = train(model, optimizer, train_loader, 
                                                      device, loss_fun, training_loss_logger)
        train_acc = evaluate(model, device, train_loader)
        valid_acc = evaluate(model, device, valid_loader)
        
        validation_acc_logger.append(valid_acc)
        training_acc_logger.append(train_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

    # Plot metrics
    plot_training_metrics(training_loss_logger, training_acc_logger, validation_acc_logger, num_epochs)
    
    # Visualize predictions (will need to adjust visualize_predictions too)
    visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()