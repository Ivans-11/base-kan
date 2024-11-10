import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath('..'))
from kans import GaussianKAN
from kans.utils import create_dataset

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Generate dataset
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
print('train_input size:', dataset['train_input'].shape)
print('train_label',dataset['train_label'].shape)
print('test_input size:', dataset['test_input'].shape)
print('test_label',dataset['test_label'].shape)

# Create data loader
train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create model
layer_sizes = [2,5,3,1]  # Specify the number of nodes per layer
grid_range = [-1,1]  # Range of the grid
grid_count = 5 # Number of grid points
model = GaussianKAN(layer_sizes, grid_range, grid_count)
model.to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training process
num_epochs = 50
epoch_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        running_loss += loss.item()
        
        # Print information every certain steps
        # if (i + 1) % 10 == 0:
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Print average loss of this epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {epoch_loss:.4f}")
    epoch_losses.append(epoch_loss)

# Save the model
torch.save(model.state_dict(), 'model/gaussian_kan_model.pth')

# Plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss of g_kan model')
plt.grid(True)
plt.show()

# Load the model
# model = GaussianKAN(layer_sizes, grid_range, grid_count)
# model.load_state_dict(torch.load('model/gaussian_kan_model.pth'))

# Test the model
model.eval()
test_input = dataset['test_input']
test_label = dataset['test_label']
with torch.no_grad():
	test_output = model(test_input)
test_loss = criterion(test_output, test_label).item()
print(f"Test Loss: {test_loss:.4f}")