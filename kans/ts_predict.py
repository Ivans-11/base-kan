import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

def create_series_dataset(series, history_size):
    X, Y = [], []
    for i in range(len(series) - history_size):
        X.append(series[i:i+history_size].flatten())
        Y.append(series[i+history_size])
    return torch.stack(X), torch.stack(Y)

def test(model, test_input, test_data, criterion):
	model.eval()
	predict_data = torch.empty(test_data.size())
	with torch.no_grad():
		for i in range(len(test_data)):
			output = model(test_input)
			predict_data[i] = output
			test_input = torch.cat((test_input[:,test_data.size(1):], output), dim=1)
		loss = criterion(predict_data, test_data)
	return loss.item(), predict_data

def plot_fitting_and_predict(model, model_name, series, X_train, test_input, test_data, criterion, series_mean, series_std):
	train_predict = model(X_train).detach()
	test_loss, test_predict = test(model, test_input, test_data, criterion)
	print(f'Test Loss of {model_name}: {test_loss:.4f}')
	train_predict = train_predict * series_std + series_mean
	test_predict = test_predict * series_std + series_mean
	series = series * series_std + series_mean
	channel_size = series.size(1)
	history_size = X_train.size(1) // channel_size
	for i in range(channel_size):
		plt.figure(figsize=(8,6))
		plt.plot(series[:,i], label='True Data', color='b', linestyle='dashed', alpha=0.5)
		plt.plot(range(history_size, len(train_predict)+history_size), train_predict[:,i], label='Fitting Data', color='r')
		plt.plot(range(len(train_predict)+history_size, len(series)), test_predict[:,i], label='Predict Data', color='g')
		plt.title(f'{model_name} Fitting and Predict (channel{i+1})')
		plt.xlabel('Time Index')
		plt.ylabel('Value')
		plt.grid(True)
		plt.legend()
		plt.show()
	predicts = torch.cat((series[:history_size], train_predict, test_predict), dim=0)
	return predicts

def train_ts_model(model, series, model_name='model', test_ratio=0.2, history_size=2, batch_size = 16, num_epochs=50, save=False, criterion=nn.MSELoss()):
	"""
		Train a series prediction model
		Args:
			model (torch.nn.Module): Model to be trained (input: [batch, history_size*channel], output: [batch, channel])
			series (torch.Tensor): Time series data (shape: [time, channel])
			model_name (str): Name of the model (default: 'model')
			test_ratio (float): Ratio of test data (default: 0.2)
			history_size (int): Number of historical steps to consider in the prediction of the next step (default: 2)
			batch_size (int): Batch size of training (default: 16)
			num_epochs (int): Number of epochs to train (default: 50)
			save (bool): Save the model or not (default: False)
			criterion (torch.nn.Module): Loss function (default: nn.MSELoss()))
		Returns:
			epoch_losses (list): List of train losses of each epoch (len: num_epochs)
			test_losses (list): List of test losses of each epoch (len: num_epochs)
			epoch_time (float): Average time of each epoch
			predicts (torch.Tensor): Predicted time series data, including fitting history and predicting future (shape: [time, channel])
	"""
	print('-'*50)
	
	# Normalize data
	series_mean = series.mean(dim=0)
	series_std = series.std(dim=0)
	series = (series - series_mean) / series_std
	
	# Split data
	test_size = int(len(series) * test_ratio)
	print(f'Target: predicting {test_size} future steps from {len(series)-test_size} historical steps with {series.size(1)} channels')
	train_data = series[:-test_size]
	test_data = series[-test_size:]
	train_X, train_Y = create_series_dataset(train_data, history_size)
	test_input = train_data[-history_size:].flatten().unsqueeze(0)
	
	# Create data loader
	train_dataset = TensorDataset(train_X, train_Y)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	
	# Training
	print(f'Training {model_name} ...')
	optimizer = optim.Adam(model.parameters(), lr=0.01)
	epoch_losses = []
	test_losses = []
	bar = tqdm(total=num_epochs)
	start_t = time.time()
	for epoch in range(num_epochs):
		running_loss = 0.0
		model.train()
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
		test_loss, _ = test(model, test_input, test_data, criterion)
		test_losses.append(test_loss)
		bar.set_description(f'{model_name} Epoch {epoch+1}')
		bar.set_postfix(loss=f'{epoch_loss:.4f}', test_loss=f'{test_loss:.4f}')
		bar.update(1)
	bar.close()
	end_t = time.time()
	epoch_time = (end_t - start_t) / num_epochs
	print(f'Average Epoch Training Time :{epoch_time}s')
	predicts = plot_fitting_and_predict(model, model_name, series, train_X, test_input, test_data, criterion, series_mean, series_std)
	if save:
		if os.path.exists('model') == False:
			os.makedirs('model')
		torch.save(model.state_dict(), f'model/{model_name}.pth')
		print(f'Model saved as model/{model_name}.pth')
	return epoch_losses, test_losses, epoch_time, predicts