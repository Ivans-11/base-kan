# train HybridKANbyLayer on Iris dataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath('..'))
from kans import HybridKANbyLayer
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if os.path.exists('model') is False:
	os.makedirs('model')


def train(model, train_loader, num_epochs=50, save=False, model_name=''):
	print('-'*50)
	print(f'Training {model_name} model...')
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss() # Use CrossEntropyLoss for classification
	epoch_losses = []
	bar = tqdm(total=num_epochs)
	start_t = time.time()
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
		epoch_loss = running_loss / len(train_loader)
		epoch_losses.append(epoch_loss)
		bar.set_description(f'{model_name} Epoch {epoch+1}')
		bar.set_postfix(loss=f'{epoch_loss:.4f}')
		bar.update(1)
	bar.close()
	end_t = time.time()
	epoch_time = (end_t - start_t) / num_epochs
	print(f'Average Epoch Training Time :{epoch_time}s')
	if save:
		torch.save(model.state_dict(), f'model/{model_name}.pth')
		print(f'Model saved as model/{model_name}.pth')
	return epoch_losses, epoch_time

def test(model, test_input, test_label, model_name):
	model.eval()
	with torch.no_grad():
		test_output = model(test_input)
		_, predicted = torch.max(test_output, 1)
		accuracy = (predicted == test_label).sum().item() / test_label.size(0) * 100
	print(f'Test Accuracy of {model_name} model: {accuracy:.2f}%')
	return accuracy

# Generate dataset
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X) # Standardize the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float64, device=device)
print('train_input size:', X_train.shape)
X_test = torch.tensor(X_test, dtype=torch.float64, device=device)
print('test_input size:', X_test.shape)
y_train = torch.tensor(y_train, dtype=torch.long, device=device)
print('train_label size:', y_train.shape)
y_test = torch.tensor(y_test, dtype=torch.long, device=device)
print('test_label size:', y_test.shape)
dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}

# Create data loader
train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Setting of the model: The number of learnable parameters per layer is: input_size * output_size * (p + 1)
layer_sizes = [4,3]  # Specify the number of nodes per layer

# b_kan: p = b_grid_count
b_order = 3  # Order of B-spline
b_grid_range = [-1,1]  # Grid range of B-spline
b_grid_count = 6  # Grid count of B-spline

# f_kan: p = frequency_count * 2 + 1
frequency_count = 3  # Frequency count of Fourier series

# g_kan: p = g_grid_count
g_grid_range = [-1,1]  # Grid range of Gaussian radial basis function
g_grid_count = 6 # Grid count of Gaussian radial basis function

# j_kan: p = j_order + 1
j_order = 5  # Order of Jacobi polynomial
alpha = 0.5  # Alpha of Jacobi polynomial
beta = 0.5  # Beta of Jacobi polynomial

# r_kan: p = mole_order + deno_order + 1
mole_order = 3  # Order of numerator
deno_order = 2  # Order of denominator

# t_kan: p = t_order + 1
t_order = 5  # Order of Taylor polynomial

# w_kan: p = wave_num * 3
wave_num = 2  # Number of wavelets
wave_type = 'morlet'  # Type of wavelet

# be_kan: p = be_order + 1
be_order = 5  # Order of Bernstein polynomial
inter_range = [0, 1]  # Interpolation range

# Train and test the models
num_epochs = 50
# hybrid_kan
model = HybridKANbyLayer(layer_sizes,
				  ['bspline', 'fourier', 'gaussian', 'jacobi', 'rational', 'taylor', 'wavelet', 'bernstein'],
				  b_order=b_order, b_grid_range=b_grid_range, b_grid_count=b_grid_count,
				  frequency_count=frequency_count,
				  g_grid_range=g_grid_range, g_grid_count=g_grid_count,
				  j_order=j_order, alpha=alpha, beta=beta,
				  mole_order=mole_order, deno_order=deno_order,
				  t_order=t_order,
				  wave_num=wave_num, wave_type=wave_type,
				  be_order=be_order, inter_range=inter_range
				  ).to(device)
hybrid_epoch_losses, hybrid_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='HybridKAN_iris')
hybrid_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'HybridKAN_iris')
model.show_contributions()
del model
torch.cuda.empty_cache()

# b_kan
model = HybridKANbyLayer(layer_sizes, ['bspline']*8, b_order=b_order, b_grid_range=b_grid_range, b_grid_count=b_grid_count).to(device)
b_epoch_losses, b_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='BSplineKAN_iris')
b_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'BSplineKAN_iris')
del model
torch.cuda.empty_cache()

# f_kan
model = HybridKANbyLayer(layer_sizes, ['fourier']*8, frequency_count=frequency_count).to(device)
f_epoch_losses, f_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='FourierKAN_iris')
f_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'FourierKAN_iris')
del model
torch.cuda.empty_cache()

# g_kan
model = HybridKANbyLayer(layer_sizes, ['gaussian']*8, g_grid_range=g_grid_range, g_grid_count=g_grid_count).to(device)
g_epoch_losses, g_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='GaussianKAN_iris')
g_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'GaussianKAN_iris')
del model
torch.cuda.empty_cache()

# j_kan
model = HybridKANbyLayer(layer_sizes, ['jacobi']*8, j_order=j_order, alpha=alpha, beta=beta).to(device)
j_epoch_losses, j_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='JacobiKAN_iris')
j_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'JacobiKAN_iris')
del model
torch.cuda.empty_cache()

# r_kan
model = HybridKANbyLayer(layer_sizes, ['rational']*8, mole_order=mole_order, deno_order=deno_order).to(device)
r_epoch_losses, r_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='RationalKAN_iris')
r_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'RationalKAN_iris')
del model
torch.cuda.empty_cache()

# t_kan
model = HybridKANbyLayer(layer_sizes, ['taylor']*8, t_order=t_order).to(device)
t_epoch_losses, t_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='TaylorKAN_iris')
t_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'TaylorKAN_iris')
del model
torch.cuda.empty_cache()

# w_kan
model = HybridKANbyLayer(layer_sizes, ['wavelet']*8, wave_num=wave_num, wave_type=wave_type).to(device)
w_epoch_losses, w_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='WaveletKAN_iris')
w_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'WaveletKAN_iris')
del model
torch.cuda.empty_cache()

# be_kan
model = HybridKANbyLayer(layer_sizes, ['bernstein']*8, be_order=be_order, inter_range=inter_range).to(device)
be_epoch_losses, be_epoch_time = train(model, train_loader, num_epochs=num_epochs, save=True, model_name='BernsteinKAN_iris')
be_test_accuracy = test(model, dataset['test_input'], dataset['test_label'], 'BernsteinKAN_iris')
del model
torch.cuda.empty_cache()

# Plot the loss curve
plt.figure(figsize=(8,6))
plt.plot(hybrid_epoch_losses, label=f'HybridKANbyLayer,{hybrid_epoch_time:.4f}s/epoch,Test Accuracy:{hybrid_test_accuracy:.2f}%', color='brown')
plt.plot(b_epoch_losses, label=f'BSplineKAN,{b_epoch_time:.4f}s/epoch,Test Accuracy:{b_test_accuracy:.2f}%', color='r', linestyle='dashed')
plt.plot(f_epoch_losses, label=f'FourierKAN,{f_epoch_time:.4f}s/epoch,Test Accuracy:{f_test_accuracy:.2f}%', color='g', linestyle='dashed')
plt.plot(g_epoch_losses, label=f'GaussianKAN,{g_epoch_time:.4f}s/epoch,Test Accuracy:{g_test_accuracy:.2f}%', color='b', linestyle='dashed')
plt.plot(j_epoch_losses, label=f'JacobiKAN,{j_epoch_time:.4f}s/epoch,Test Accuracy:{j_test_accuracy:.2f}%', color='c', linestyle='dashed')
plt.plot(r_epoch_losses, label=f'RationalKAN,{r_epoch_time:.4f}s/epoch,Test Accuracy:{r_test_accuracy:.2f}%', color='m', linestyle='dashed')
plt.plot(t_epoch_losses, label=f'TaylorKAN,{t_epoch_time:.4f}s/epoch,Test Accuracy:{t_test_accuracy:.2f}%', color='y', linestyle='dashed')
plt.plot(w_epoch_losses, label=f'WaveletKAN,{w_epoch_time:.4f}s/epoch,Test Accuracy:{w_test_accuracy:.2f}%', color='k', linestyle='dashed')
plt.plot(be_epoch_losses, label=f'BernsteinKAN,{be_epoch_time:.4f}s/epoch,Test Accuracy:{be_test_accuracy:.2f}%', color='orange', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss of KAN models on Iris dataset')
plt.grid(True)
plt.legend()
plt.show()

# Plot the loss curve(log)
plt.figure(figsize=(8,6))
plt.plot(hybrid_epoch_losses, label=f'HybridKANbyLayer', color='brown')
plt.plot(b_epoch_losses, label=f'BSplineKAN', color='r', linestyle='dashed')
plt.plot(f_epoch_losses, label=f'FourierKAN', color='g', linestyle='dashed')
plt.plot(g_epoch_losses, label=f'GaussianKAN', color='b', linestyle='dashed')
plt.plot(j_epoch_losses, label=f'JacobiKAN', color='c', linestyle='dashed')
plt.plot(r_epoch_losses, label=f'RationalKAN', color='m', linestyle='dashed')
plt.plot(t_epoch_losses, label=f'TaylorKAN', color='y', linestyle='dashed')
plt.plot(w_epoch_losses, label=f'WaveletKAN', color='k', linestyle='dashed')
plt.plot(be_epoch_losses, label=f'BernsteinKAN', color='orange', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss($log_{10}$)')
plt.yscale('log')
plt.title('Training Loss($log_{10}$) of KAN models on Iris dataset')
plt.grid(True)
plt.legend()
plt.show()