def train(model, optimizer, loader, device, loss_fun, loss_logger):
    model.train()
    for i, (x, y) in enumerate(loader):
        fx = model(x.to(device))
        loss = loss_fun(fx, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_logger.append(loss.item())
        
    return model, optimizer, loss_logger

def evaluate(model, device, loader):
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            fx = model(x.to(device))
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()
            
    return epoch_acc / len(loader.dataset)