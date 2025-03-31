import torch
from config.config import batch_size, num_epochs, learning_rate, data_set_root, device
from data.data_loader import get_data_loaders
from models.model import CNN
from training.train import train
from training.evaluate import evaluate
from utils.visualization import plot_training_metrics, visualize_predictions

def main():
    train_loader, valid_loader, test_loader = get_data_loaders(data_set_root, batch_size)
    
    model = CNN(channels_in=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []

    for epoch in range(num_epochs):
        model, optimizer, training_loss_logger = train(model, optimizer, train_loader, device, training_loss_logger)
        train_acc = evaluate(model, device, train_loader)
        valid_acc = evaluate(model, device, valid_loader)
        
        validation_acc_logger.append(valid_acc)
        training_acc_logger.append(train_acc)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}')

    plot_training_metrics(training_loss_logger, training_acc_logger, validation_acc_logger)
    visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()