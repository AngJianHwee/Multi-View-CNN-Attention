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
        # break  # TODO: Remove this break to train on the entire dataset
        
        # if i > len(loader) // 10: # TODO: Remove this break to train on the entire dataset
        #     # Only break after 10% of the dataset
        #     break
    return model, optimizer, loss_logger


