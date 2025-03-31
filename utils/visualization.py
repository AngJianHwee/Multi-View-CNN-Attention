def plot_training_metrics(training_loss, validation_accuracy, num_epochs):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, num_epochs + 1), training_loss, label='Training Loss', color='blue')
    plt.plot(np.arange(1, num_epochs + 1), validation_accuracy, label='Validation Accuracy', color='orange')
    plt.title('Training Loss and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

def visualize_predictions(images, true_labels, predictions, num_images=8):
    import matplotlib.pyplot as plt
    import torchvision

    plt.figure(figsize=(20, 10))
    out = torchvision.utils.make_grid(images[:num_images], nrow=8, normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.title("Predictions vs True Labels")
    plt.axis('off')

    for i in range(num_images):
        plt.text(i * 25, 0, f'True: {true_labels[i]}\nPred: {predictions[i]}', color='white', fontsize=12)

    plt.show()

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