import matplotlib.pyplot as plt
import torchvision
import torch

def plot_training_metrics(training_loss_logger, training_acc_logger, validation_acc_logger, num_epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate points per epoch for proper x-axis alignment
    points_per_epoch_loss = len(training_loss_logger) // num_epochs
    points_per_epoch_acc = len(training_acc_logger) // num_epochs  # Should be 1 per epoch, but for flexibility
    
    # Generate x-axis points
    x_loss = np.linspace(1, num_epochs, len(training_loss_logger))
    x_acc = np.arange(1, num_epochs + 1)  # One point per epoch for accuracies

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot training loss
    ax1.plot(x_loss, training_loss_logger, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    # Plot training and validation accuracy
    ax2.plot(x_acc, training_acc_logger, label='Training Accuracy', color='blue')
    ax2.plot(x_acc, validation_acc_logger, label='Validation Accuracy', color='orange')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('training_metrics.png')  # Save the figure

# Rest of your visualization.py (like visualize_predictions) would go here
def visualize_predictions(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        test_images, test_labels = next(iter(test_loader))
        test_images = test_images.to(device)
        fx, _ = model(test_images, test_images, test_images)  # Get predictions
        pred = fx.argmax(-1)
        
        plt.figure(figsize=(20, 10))
        out = torchvision.utils.make_grid(test_images[:8].cpu(), 8, normalize=True)
        plt.imshow(out.numpy().transpose((1, 2, 0)))
        plt.show()
        plt.savefig('predictions.png')  # Save the figure
        
        print("Predicted Values\n", list(pred[:8].cpu().numpy()))
        print("True Values\n", list(test_labels[:8].numpy()))


def visualize_attention(image, attention_map, x_dim, y_dim):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Plot the original image
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[0].scatter(x_dim, y_dim, color='red', marker='x')

    # Plot the attention map
    axes[1].imshow(attention_map.reshape(32, 32).cpu().numpy(), cmap='viridis')
    axes[1].set_title("Attention Map")
    axes[1].axis('off')

    plt.show()
    plt.savefig('attention_map.png')  # Save the figure