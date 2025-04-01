import torch
from tqdm import tqdm  # Assuming tqdm is used based on your imports

def train(model, optimizer, loader, device, loss_fun, loss_logger):
    model.train()
    for i, (x, y) in enumerate(tqdm(loader, desc=f"Training")):
        x = x.to(device)
        y = y.to(device)
        # Pass the same input to all three views
        fx, _ = model(x, x, x)  # Ignore features during training
        loss = loss_fun(fx, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_logger.append(loss.item())
    return model, optimizer, loss_logger


def evaluate(model, device, loader):
    epoch_acc = 0
    model.eval()
    total_samples = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Evaluating")):
            x = x.to(device)
            y = y.to(device)
            fx, _ = model(x, x, x)  # Ignore features during evaluation
            epoch_acc += (fx.argmax(1) == y).sum().item()
            total_samples += y.size(0)
    return epoch_acc / total_samples  # Normalize by total samples instead of dataset length