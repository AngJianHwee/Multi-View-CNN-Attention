# source: https://github.com/LukeDitria/pytorch_tutorials/blob/main/section13_attention/solutions/Pytorch3_CNN_Self_Attention.ipynb

# !pip install torch torchvision 
# !pip install torchsummary
# !pip install matplotlib tqdm ipython

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.dataloader as dataloader

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import trange, tqdm

batch_size = 64
num_epochs = 50
learning_rate = 1e-4
data_set_root = "../../datasets"

gpu_indx = 0
device = torch.device(gpu_indx if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]) 

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])]) 

train_data = datasets.CIFAR10(data_set_root, train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(data_set_root, train=False, download=True, transform=test_transform)

validation_split = 0.9

n_train_examples = int(len(train_data) * validation_split)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples],
                                                       generator=torch.Generator().manual_seed(42))



print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


train_loader = dataloader.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = dataloader.DataLoader(valid_data, batch_size=batch_size)
test_loader  = dataloader.DataLoader(test_data, batch_size=batch_size)


# <h2> Create the CNN with Self Attention</h2>

class CNN(nn.Module):
    def __init__(self, channels_in):
        # Call the __init__ function of the parent nn.module class
        super(CNN, self).__init__()
        # Define Convolution Layers
        self.conv1 = nn.Conv2d(channels_in, 64, 3, 1, 1, padding_mode='reflect')
        
        # Define Layer Normalization and Multi-head Attention layers
        self.norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(64, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

        # Define additional Convolution Layers
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Define Dropout and Fully Connected Layers
        self.do = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128*4*4, 10)
        
    def use_attention(self, x):
        # Reshape input for multi-head attention
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        
        # Apply Layer Normalization
        x_att = self.norm(x_att)
        # Apply Multi-head Attention
        att_out, att_map  = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        
        # Apply self-attention mechanism and add to the input
        x = self.scale * self.use_attention(x)[0] + x
        
        # Apply batch normalization and ReLU activation
        x = F.relu(x)
        
        # Additional convolutional layers
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        
        # Flatten the output and apply dropout
        x = self.do(x.reshape(x.shape[0], -1))

        # Fully connected layer for final output
        return self.fc_out(x)

# Create a dataloader itterable object
dataiter = next(iter(test_loader))
# Sample from the itterable object
test_images, test_labels = dataiter

# Lets visualise an entire batch of images!
plt.figure(figsize = (20,10))
out = torchvision.utils.make_grid(test_images, 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))

# Create an instance of our network
# Set channels_in to the number of channels of the dataset images (1 channel for MNIST)
model = CNN(channels_in = test_images.shape[1]).to(device)

# View the network
# Note that the layer order is simply the order in which we defined them, NOT the order of the forward pass
print(model)

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in model.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Pass image through network
out = model(test_images.to(device))
# Check output
out.shape

# <h3> Set up the optimizer </h3>

# Pass our network parameters to the optimiser set our lr as the learning_rate
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Define a Cross Entropy Loss
loss_fun = nn.CrossEntropyLoss()

# # Define the training process

# This function should perform a single training epoch using our training data
def train(model, optimizer, loader, device, loss_fun, loss_logger):
    
    # Set Network in train mode
    model.train()
    for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="Training")):
        # Forward pass of image through network and get output
        fx = model(x.to(device))
        
        # Calculate loss using loss function
        loss = loss_fun(fx, y.to(device))

        # Zero Gradents
        optimizer.zero_grad()
        # Backpropagate Gradents
        loss.backward()
        # Do a single optimization step
        optimizer.step()
        
        # Log the loss for plotting
        loss_logger.append(loss.item())
        
    # Return the avaerage loss and acc from the epoch as well as the logger array       
    return model, optimizer, loss_logger

# # Define the testing process

# This function should perform a single evaluation epoch, it WILL NOT be used to train our model
def evaluate(model, device, loader):
    
    # Initialise counter
    epoch_acc = 0
    
    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values (More on these layer types soon!)
    model.eval()
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, leave=False, desc="Evaluating")):
            # Forward pass of image through network
            fx = model(x.to(device))
            
            # Log the cumulative sum of the acc
            epoch_acc += (fx.argmax(1) == y.to(device)).sum().item()
            
    # Return the accuracy from the epoch     
    return epoch_acc / len(loader.dataset)

# # The training process

training_loss_logger = []
validation_acc_logger = []
training_acc_logger = []

valid_acc = 0
train_acc = 0

# This cell implements our training loop
pbar = trange(0, num_epochs, leave=False, desc="Epoch")    
for epoch in pbar:
    
    # Call the training function and pass training dataloader etc
    model, optimizer, training_loss_logger = train(model=model, 
                                                   optimizer=optimizer, 
                                                   loader=train_loader, 
                                                   device=device, 
                                                   loss_fun=loss_fun, 
                                                   loss_logger=training_loss_logger)
    
    # Call the evaluate function and pass the dataloader for both validation and training
    train_acc = evaluate(model=model, device=device, loader=train_loader)
    valid_acc = evaluate(model=model, device=device, loader=valid_loader)
    
    # Log the train and validation accuracies
    validation_acc_logger.append(valid_acc)
    training_acc_logger.append(train_acc)

print("Training Complete")

# ## Plot Metrics

plt.figure(figsize = (10,5))
train_x = np.linspace(0, num_epochs, len(training_loss_logger))
plt.plot(train_x, training_loss_logger)
_ = plt.title("LeNet Training Loss")

plt.figure(figsize = (10,5))
train_x = np.linspace(0, num_epochs, len(training_acc_logger))
plt.plot(train_x, training_acc_logger, c = "y")
valid_x = np.linspace(0, num_epochs, len(validation_acc_logger))
plt.plot(valid_x, validation_acc_logger, c = "k")

plt.title("LeNet")
_ = plt.legend(["Training accuracy", "Validation accuracy"])

# # Evaluate

# Call the evaluate function and pass the evaluation/test dataloader etc
test_acc = evaluate(model=model, device=device, loader=test_loader)

# Lets visualise the prediction for a few test images!

with torch.no_grad():
    fx = model(test_images[:8].to(device))
    pred = fx.argmax(-1)
    
plt.figure(figsize = (20,10))
out = torchvision.utils.make_grid(test_images[:8], 8, normalize=True)
plt.imshow(out.numpy().transpose((1, 2, 0)))

print("Predicted Values\n", list(pred.cpu().numpy()))
print("True Values\n", list(test_labels[:8].numpy()))

# Assuming model and test_images are already defined and loaded
with torch.no_grad():
    x = model.conv1(test_images[:8].to(device))
    _, att_map = model.use_attention(x)
    
# Index of the image you want to visualize
img_idx = 6

# Specify the dimensions for the attention map visualization
x_dim = 5
y_dim = 25

assert x_dim < test_images.shape[3], "x_dim must be less than " + str(test_images.shape[3] - 1)
assert y_dim < test_images.shape[2], "y_dim must be less than " + str(test_images.shape[2] - 1)

# Plot the image and its corresponding attention map
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# Plot the original image
img_out = test_images[img_idx]
img_out = (img_out - img_out.min())/(img_out.max() - img_out.min())
axes[0].imshow(img_out.permute(1, 2, 0).cpu().numpy())
axes[0].set_title("Original Image")
axes[0].axis('off')
axes[0].scatter(x_dim, y_dim, color='red', marker='x')

# Plot the attention map
axes[1].imshow(att_map[img_idx, x_dim * y_dim].reshape(32, 32).cpu().numpy(), cmap='viridis')
axes[1].set_title("Attention Map")
axes[1].axis('off')

plt.show()


