import torch
from torchsummary import summary
import os
import tempfile

def print_model_summary(model, input_size, device):
    """
    Prints a summary of the model architecture and saves/analyzes its state dict size.
    
    Args:
        model: PyTorch model instance
        input_size: Tuple or list of tuples representing input size(s) (channels, H, W)
        device: Torch device to run the summary on
    """
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Print model summary using torchsummary
    print("Model Summary:")
    if isinstance(input_size, tuple):
        # Single input case
        summary(model, input_size=input_size)
    elif isinstance(input_size, list) and len(input_size) == 3:
        # For ThreeViewCNN with three inputs
        # torchsummary doesn't directly support multiple inputs, so we'll simulate it
        print("Note: ThreeViewCNN takes three inputs. Showing summary for a single view.")
        summary(model.view1, input_size[0])  # Print summary for one view as representative
        print("\nFull model has three such views with fusion layer.")
    else:
        raise ValueError("input_size must be a tuple or list of 3 tuples")

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,} "
          f"(~{total_params / 1e6:.2f}M)")

    # Save model state dict to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        temp_path = tmp_file.name
        torch.save(model.state_dict(), temp_path)
        
        # Get file size
        file_size = os.path.getsize(temp_path) / (1024 * 1024)  # Size in MB
        print(f"Model state dict saved to: {temp_path}")
        print(f"Size of saved model state dict: {file_size:.2f} MB")
    
    # Clean up temporary file
    os.remove(temp_path)
    print(f"Temporary file {temp_path} removed.")

if __name__ == "__main__":
    # Example usage with ThreeViewCNN
    from models.model import ThreeViewCNN
    
    # Assuming CIFAR-10 input size (3, 32, 32) for each view
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeViewCNN(channels_ins=[3, 3, 3], output_dim=10)
    input_size = [(3, 32, 32)] * 3  # Three identical inputs
    
    print_model_summary(model, input_size, device)