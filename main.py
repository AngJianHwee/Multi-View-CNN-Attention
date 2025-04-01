from utils.model_summary import print_model_summary
import torch
from config.config import batch_size, num_epochs, learning_rate, data_set_root, device
from data.data_loader import get_data_loaders  # Assuming this still exists
from models.model import ThreeViewCNN  # Updated import
from training.train import train  # Will need to update this too
from training.evaluate import evaluate  # Will need to update this too
from utils.visualization import plot_training_metrics, visualize_attention

import os


def main(batch_size, num_epochs, learning_rate, data_set_root, device):
    # check if data_set_root exists
    if not os.path.exists(data_set_root):
        os.makedirs(data_set_root)

    train_loader, valid_loader, test_loader = get_data_loaders(
        batch_size=batch_size, dataset_root=data_set_root
    )

    # Initialize ThreeViewCNN with 3 channels for each view (CIFAR-10 has 3 channels)

    model = ThreeViewCNN(
        channels_ins=[3, 3, 3], output_dims_individual=[10, 10, 10], output_dim=10
    ).to(device)

    # Print model summary before training
    print_model_summary(model, input_size=[(3, 32, 32)] * 3, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = torch.nn.CrossEntropyLoss()

    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []

    best_3_acc = [
        {"epoch": -1, "acc": 0.0, "model_dict": model.state_dict().copy()},
        {"epoch": -2, "acc": 0.01, "model_dict": model.state_dict().copy()},
        {"epoch": -3, "acc": 0.02, "model_dict": model.state_dict().copy()},
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
        best_3_acc.append(
            {"epoch": epoch, "acc": valid_acc, "model_dict": model.state_dict().copy()}
        )

        # Sort the best 3 accuracies
        best_3_acc.sort(key=lambda x: x["acc"], reverse=True)

        # Keep only the top 3
        best_3_acc = best_3_acc[:3]
        print(
            f"Best 3 accuracies so far: {[(b['epoch'], b['acc']) for b in best_3_acc]}"
        )

    # Save the model for 3 best epochs
    for i, best_model in enumerate(best_3_acc):
        model_path = os.path.join(
            data_set_root, f"best_model_epoch_{best_model['epoch']}.pth"
        )
        torch.save(best_model["model_dict"], model_path)
        print(
            f"Saved model for epoch {best_model['epoch']} with accuracy {best_model['acc']} to {model_path}"
        )
    # Save the final model
    final_model_path = os.path.join(data_set_root, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Plot metrics
    plot_training_metrics(
        training_loss_logger, training_acc_logger, validation_acc_logger, num_epochs
    )

    # # Visualize predictions (will need to adjust visualize_predictions too)
    # visualize_predictions(model, test_loader, device)

    # save everything about logs into a json
    logs = {
        "training_loss": training_loss_logger,
        "validation_accuracy": validation_acc_logger,
        "training_accuracy": training_acc_logger,
        "best_3_acc_no_state_dict": [
            {"epoch": b["epoch"], "acc": b["acc"]} for b in best_3_acc
        ],
    }
    logs_path = os.path.join(data_set_root, "logs.json")
    with open(logs_path, "w") as f:
        import json

        json.dump(logs, f)
    print(f"Saved logs to {logs_path}")

    # Add attention visualization for all three views with batch inference
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images.to(device)

        # Take first 5 images for visualization
        sample_images = test_images[:5]

        # Batch inference for all images in the batch to get attention maps
        batch_size = test_images.size(0)

        # Process through each view for the entire batch
        x1 = model.view1.conv1(test_images)  # [batch_size, 64, 32, 32]
        _, att_map1 = model.view1.use_attention(x1)  # [batch_size, 1024, 1024]
        x2 = model.view2.conv1(test_images)
        _, att_map2 = model.view2.use_attention(x2)
        x3 = model.view3.conv1(test_images)
        _, att_map3 = model.view3.use_attention(x3)

        # keep only first 5
        att_map1 = att_map1[:5]
        att_map2 = att_map1[:5]
        att_map3 = att_map1[:5]

        # print shape
        print(f"Attention map 1 shape: {att_map1.shape}")
        print(f"Attention map 2 shape: {att_map2.shape}")
        print(f"Attention map 3 shape: {att_map3.shape}")
        print(f"Sample images shape: {sample_images.shape}")
        
        # Prepare attention maps for visualization (same mean map for all 5 images)
        attention_maps = []
        for i in range(5):
            att_map1_mean = att_map1[i].mean(dim=0)
            att_map2_mean = att_map2[i].mean(dim=0)
            att_map3_mean = att_map3[i].mean(dim=0)
            attention_maps.append((att_map1_mean, att_map2_mean, att_map3_mean))

        # Visualize 5 images with their corresponding mean attention maps
        visualize_attention(sample_images, attention_maps, x_dim=16, y_dim=16)


if __name__ == "__main__":
    main(
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        data_set_root=data_set_root,
        device=device,
    )
