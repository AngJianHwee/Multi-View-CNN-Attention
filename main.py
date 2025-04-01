from utils.model_summary import print_model_summary
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

    train_loader, valid_loader, test_loader = get_data_loaders(
        batch_size=batch_size, dataset_root=data_set_root
    )

    # Initialize ThreeViewCNN with 3 channels for each view (CIFAR-10 has 3 channels)

    model = ThreeViewCNN(
        channels_ins=[3, 3, 3],
        output_dims_individual=[10, 10, 10],
        output_dim=10
    ).to(device)
    
    # Print model summary before training
    print_model_summary(model, input_size=[(3, 32, 32)] * 3, device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()

    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []

    best_3_acc = [
        {"epoch": -1, "acc": 0.0, "model_dict": model.state_dict().copy().cpu()},
        {"epoch": -2, "acc": 0.01, "model_dict": model.state_dict().copy().cpu()},
        {"epoch": -3, "acc": 0.02, "model_dict": model.state_dict().copy().cpu()},
    ]
    for epoch in range(num_epochs):
        # Train with three identical inputs
        model, optimizer, training_loss_logger = train(
            model, optimizer, train_loader, device, loss_fun, training_loss_logger
        )
        train_acc = evaluate(model, device, train_loader)
        valid_acc = evaluate(model, device, valid_loader)

        validation_acc_logger.append(valid_acc)
        training_acc_logger.append(train_acc)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}"
        )
        
        # Evaluate on test set
        test_acc = evaluate(model, device, test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Save the best model based on validation accuracy, 
        # so it will compare with all 3, select the best 3
        best_3_acc.append({"epoch": epoch, "acc": valid_acc, "model_dict": model.state_dict().copy().cpu()})
        
        # Sort the best 3 accuracies
        best_3_acc.sort(key=lambda x: x["acc"], reverse=True)
        
        # Keep only the top 3
        best_3_acc = best_3_acc[:3]
        print(f"Best 3 accuracies so far: {best_3_acc}")
        
    # Save the model for 3 best epochs
    for i, best_model in enumerate(best_3_acc):
        model_path = os.path.join(data_set_root, f"best_model_epoch_{best_model['epoch']}.pth")
        torch.save(best_model["model_dict"], model_path)
        print(f"Saved model for epoch {best_model['epoch']} with accuracy {best_model['acc']} to {model_path}")
    # Save the final model
    final_model_path = os.path.join(data_set_root, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
        
        

    # Plot metrics
    plot_training_metrics(training_loss_logger, training_acc_logger, validation_acc_logger, num_epochs)

    # Visualize predictions (will need to adjust visualize_predictions too)
    visualize_predictions(model, test_loader, device)


if __name__ == "__main__":
    main()
