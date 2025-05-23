import matplotlib.pyplot as plt
import torchvision
import torch

# Set the style for matplotlib
plt.style.use('ggplot')

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
    plt.savefig('training_metrics.png')  # Save the figure
    print("Training metrics saved as training_metrics.png")
    # plt.show()


def visualize_attention(images, attention_maps, x_dim, y_dim):
    """
    Visualize 5 images and their mean attention maps from three views in a 5x4 grid.
    
    Args:
        images: Tensor of shape [5, C, H, W]
        attention_maps: List of 5 tuples, each with 3 mean attention maps [32, 32]
        x_dim, y_dim: Coordinates to mark on the original images
    """
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))  # 5 rows, 4 columns (20 total subplots)

    for i in range(5):
        # Normalize image for display
        img_display = images[i].permute(1, 2, 0).cpu().numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        # Plot the original image
        axes[i, 0].imshow(img_display)
        if i == 0:
            axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        axes[i, 0].scatter(x_dim, y_dim, color='red', marker='x')

        # Plot mean attention maps for each view
        titles = ["View 1 Mean Attention", "View 2 Mean Attention", "View 3 Mean Attention"]
        for j, (att_map, title) in enumerate(zip(attention_maps[i], titles)):
            # Overlay attention map on the image
            axes[i, j + 1].imshow(img_display, alpha=0.5)
            axes[i, j + 1].imshow(att_map, cmap='hot', alpha=0.5)
            
            if i == 0:
                axes[i, j + 1].set_title(title)
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig('mean_attention_maps_five_images.png')
    print("Mean attention maps for 5 images saved as mean_attention_maps_five_images.png")
    # plt.show()  # Uncomment if you want to display it too
    
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
        plt.savefig('predictions.png')  # Save the figure
        # plt.show()
        
        
        print("Predicted Values\n", list(pred[:8].cpu().numpy()))
        print("True Values\n", list(test_labels[:8].numpy()))


