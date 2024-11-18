import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

class TSModel():
	"""
        Time series prediction model
        Args:
            model (torch.nn.Module): Model to be trained £¨input: [batch, history_size*channel], output: [batch, channel])
            model_name (str): Name of the model (default: 'model')
            history_size (int): Number of historical steps to predict future steps (default: 0)
            channel (int): Number of channels of the time series data (default: 0)
    """
	def __init__(self, model, model_name='model', history_size=0, channel=0):
		self.model = model
		self.model_name = model_name
		self.history_size = history_size
		self.channel = channel
		self.mean = 0.0
		self.std = 1.0
		self.first_train = True

	def setmeanstd(self, mean, std):
		self.mean = mean
		self.std = std
		self.first_train = False

	def create_series_dataset(self, series):
		X, Y = [], []
		for i in range(len(series) - self.history_size):
			X.append(series[i:i+self.history_size].flatten())
			Y.append(series[i+self.history_size])
		return torch.stack(X), torch.stack(Y)

	def plot(self, series, X_train, test_input, test_data, criterion):
		train_predict = self.model(X_train).detach()
		test_loss, test_predict = self.test(test_input, test_data, criterion, out=False)
		print(f'Test Loss of {self.model_name}: {test_loss:.4f}')
		train_predict = train_predict * self.std + self.mean
		test_predict = test_predict * self.std + self.mean
		series = series * self.std + self.mean
		for i in range(self.channel):
			plt.figure(figsize=(8,6))
			plt.plot(series[:,i], label='True Data', color='b', linestyle='dashed', alpha=0.5)
			plt.plot(range(self.history_size, len(train_predict)+self.history_size), train_predict[:,i], label='Fitting Data', color='r')
			plt.plot(range(len(train_predict)+self.history_size, len(series)), test_predict[:,i], label='Predict Data', color='g')
			plt.title(f'{self.model_name} Fitting and Predict (channel{i+1})')
			plt.xlabel('Time Index')
			plt.ylabel('Value')
			plt.grid(True)
			plt.legend()
			plt.show()
		predicts = torch.cat((series[:self.history_size], train_predict, test_predict), dim=0)
		return predicts
	
	def test(self, test_input, test_data, criterion=nn.MSELoss(), out=True):
		"""
			Test a time series prediction model
			Args:
				test_input (torch.Tensor): Input data of the test data (shape: [1, history_size*channel])
				test_data (torch.Tensor): Test data (shape: [time, channel])
				criterion (torch.nn.Module): Loss function (default: nn.MSELoss())
			Returns:
				loss (float): Loss of the test data
				predict_data (torch.Tensor): Predicted time series data (shape: [time, channel])
		"""
		if out:
			test_input = (test_input - self.mean) / self.std
		self.model.eval()
		predict_data = torch.empty(test_data.size())
		with torch.no_grad():
			for i in range(len(test_data)):
				output = self.model(test_input)
				predict_data[i] = output
				test_input = torch.cat((test_input[:,test_data.size(1):], output), dim=1)
			loss = criterion(predict_data, test_data)
		if out:
			print(f'Test Loss: {loss.item()}')
			predict_data = predict_data * self.std + self.mean
		return loss.item(), predict_data

	def train(self, series, test_ratio=0.2, batch_size = 16, num_epochs=50, save=False, criterion=nn.MSELoss(), history_size=0):
		"""
			Train a series prediction model
			Args:
				series (torch.Tensor): Time series data (shape: [time, channel])
				test_ratio (float): Ratio of test data (default: 0.2)
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
		
		if self.history_size == 0:
			if history_size == 0:
				raise ValueError('history_size should be specified')
			self.history_size = history_size
		
		if self.channel == 0:
			self.channel = series.size(1)
		else:
			if self.channel != series.size(1):
				raise ValueError('channel size should be consistent with the input data')

		# Normalize data
		if self.first_train:
			series_mean = series.mean(dim=0)
			series_std = series.std(dim=0)
			series = (series - series_mean) / series_std
			self.setmeanstd(series_mean, series_std)
		else:
			series = (series - self.mean) / self.std

		# Split data
		test_size = int(len(series) * test_ratio)
		print(f'Target: predicting {test_size} future steps from {len(series)-test_size} historical steps with {series.size(1)} channels')
		train_data = series[:-test_size]
		test_data = series[-test_size:]
		train_X, train_Y = self.create_series_dataset(train_data)
		test_input = train_data[-history_size:].flatten().unsqueeze(0)
	
		# Create data loader
		train_dataset = TensorDataset(train_X, train_Y)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	
		# Training
		print(f'Training {self.model_name} ...')
		optimizer = optim.Adam(self.model.parameters(), lr=0.01)
		epoch_losses = []
		test_losses = []
		bar = tqdm(total=num_epochs)
		start_t = time.time()
		for epoch in range(num_epochs):
			running_loss = 0.0
			self.model.train()
			for i, (inputs, labels) in enumerate(train_loader):
				# Forward pass
				outputs = self.model(inputs)
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
			test_loss, _ = self.test(test_input, test_data, criterion, out=False)
			test_losses.append(test_loss)
			bar.set_description(f'{self.model_name} Epoch {epoch+1}')
			bar.set_postfix(loss=f'{epoch_loss:.4f}', test_loss=f'{test_loss:.4f}')
			bar.update(1)
		bar.close()
		end_t = time.time()
		epoch_time = (end_t - start_t) / num_epochs
		print(f'Average Epoch Training Time :{epoch_time}s')
		predicts = self.plot(series, train_X, test_input, test_data, criterion)
		if save:
			if os.path.exists('model') == False:
				os.makedirs('model')
			torch.save(self.model.state_dict(), f'model/{self.model_name}.pth')
			print(f'Model saved as model/{self.model_name}.pth')
		return epoch_losses, test_losses, epoch_time, predicts

class ClassifyModel():
	"""
        Classification model
        Args:
            model (torch.nn.Module): Model to be trained (input: [batch, input_size], output: [batch, num_classes])
            model_name (str): Name of the model (default: 'model')
    """
	def __init__(self, model, model_name='model'):
		self.model = model
		self.model_name = model_name
		self.mean = 0.0
		self.std = 1.0
		self.first_train = True

	def setmeanstd(self, mean, std):
		self.mean = mean
		self.std = std
		self.first_train = False

	def test(self, test_input, test_label, out=True):
		"""
			Test a classification model
			Args:
				test_input (torch.Tensor): Input data of the test data (shape: [data_size, input_size])
				test_label (torch.Tensor): Label data of the test data (shape: [data_size])
			Returns:
				accuracy (float): Accuracy of the test data
		"""
		if out:
			test_input = (test_input - self.mean) / self.std
		self.model.eval()
		with torch.no_grad():
			test_output = self.model(test_input)
			_, predicted = torch.max(test_output, 1)
			accuracy = (predicted == test_label).sum().item() / test_label.size(0) * 100
		if out:
			print(f'Test Accuracy: {accuracy:.2f}%')
		return accuracy

	def train(self, input_datas, label_datas, test_ratio=0.2, batch_size=32, num_epochs=50, save=False, criterion=nn.CrossEntropyLoss()):
		"""
			Train a classification model
			Args:
				input_datas (torch.Tensor): Input data (shape: [data_size, input_size])
				label_datas (torch.Tensor): Label data (shape: [data_size])
				test_ratio (float): Ratio of test data (default: 0.2)
				batch_size (int): Batch size of training (default: 32)
				num_epochs (int): Number of epochs to train (default: 50)
				save (bool): Save the model or not (default: False)
				criterion (torch.nn.Module): Loss function (default: nn.CrossEntropyLoss()))
			Returns:
				epoch_losses (list): List of train losses of each epoch (len: num_epochs)
				epoch_accuracies (list): List of test accuracies of each epoch (len: num_epochs)
				epoch_time (float): Average time of each epoch
		"""
		print('-'*50)
        
		# Normalize data
		if self.first_train:
			input_mean = input_datas.mean(dim=0)
			input_std = input_datas.std(dim=0)
			input_datas = (input_datas - input_mean) / input_std
			self.setmeanstd(input_mean, input_std)
		else:
			input_datas = (input_datas - self.mean) / self.std
        
		# Split data
		test_size = int(len(input_datas) * test_ratio)
		train_input = input_datas[:-test_size]
		train_label = label_datas[:-test_size]
		test_input = input_datas[-test_size:]
		test_label = label_datas[-test_size:]
		print(f'Train_input: {train_input.shape}, Train_label: {train_label.shape}, Test_input: {test_input.shape}, Test_label: {test_label.shape}')
    
		# Create data loader
		train_dataset = TensorDataset(train_input, train_label)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
		# Training
		print(f'Training {self.model_name} model...')
		optimizer = optim.Adam(self.model.parameters(), lr=0.01)
		epoch_losses = []
		epoch_accuracies = []
		bar = tqdm(total=num_epochs)
		start_t = time.time()
		for epoch in range(num_epochs):
			running_loss = 0.0
			self.model.train()
			for i, (inputs, labels) in enumerate(train_loader):
				# Forward pass
				outputs = self.model(inputs)
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
			epoch_accuracy = self.test(test_input, test_label, out=False)
			epoch_accuracies.append(epoch_accuracy)
			bar.set_description(f'{self.model_name} Epoch {epoch+1}')
			bar.set_postfix(loss=f'{epoch_loss:.4f}', accuracy=f'{epoch_accuracy:.2f}%')
			bar.update(1)
		bar.close()
		end_t = time.time()
		epoch_time = (end_t - start_t) / num_epochs
		print(f'Average Epoch Training Time :{epoch_time}s')
		if save:
			if os.path.exists('model') == False:
				os.makedirs('model')
			torch.save(self.model.state_dict(), f'model/{self.model_name}.pth')
			print(f'Model saved as model/{self.model_name}.pth')
		return epoch_losses, epoch_accuracies, epoch_time

class RegressModel():
	"""
        Regression model
        Args:
            model (torch.nn.Module): Model to be trained (input: [batch, input_size], output: [batch, output_size])
            model_name (str): Name of the model (default: 'model')
    """
	def __init__(self, model, model_name='model'):
		self.model = model
		self.model_name = model_name
		self.mean = 0.0
		self.std = 1.0
		self.first_train = True

	def setmeanstd(self, mean, std):
		self.mean = mean
		self.std = std
		self.first_train = False

	def test(self, test_input, test_label, criterion=nn.MSELoss(), out=True):
		"""
			Test a regression model
			Args:
				test_input (torch.Tensor): Input data of the test data (shape: [data_size, input_size])
				test_label (torch.Tensor): Label data of the test data (shape: [data_size, output_size])
				criterion (torch.nn.Module): Loss function (default: nn.MSELoss())
			Returns:
				loss (float): Loss of the test data
        """
		if out:
			test_input = (test_input - self.mean) / self.std
		self.model.eval()
		with torch.no_grad():
			test_output = self.model(test_input)
			loss = criterion(test_output, test_label)
		if out:
			print(f'Test Loss: {loss.item()}')
		return loss.item()

	def train(self, input_datas, label_datas, test_ratio=0.2, batch_size=32, num_epochs=50, save=False, criterion=nn.MSELoss()):
		"""
            Train a regression model
            Args:
                input_datas (torch.Tensor): Input data (shape: [data_size, input_size])
                label_datas (torch.Tensor): Label data (shape: [data_size, output_size])
                test_ratio (float): Ratio of test data (default: 0.2)
                batch_size (int): Batch size of training (default: 32)
                num_epochs (int): Number of epochs to train (default: 50)
                save (bool): Save the model or not (default: False)
                criterion (torch.nn.Module): Loss function (default: nn.MSELoss()))
            Returns:
				epoch_losses (list): List of train losses of each epoch (len: num_epochs)
				test_losses (list): List of test losses of each epoch (len: num_epochs)
				epoch_time (float): Average time of each epoch
		"""
		print('-'*50)
        
		# Normalize data
		if self.first_train:
			input_mean = input_datas.mean(dim=0)
			input_std = input_datas.std(dim=0)
			input_datas = (input_datas - input_mean) / input_std
			self.setmeanstd(input_mean, input_std)
		else:
			input_datas = (input_datas - self.mean) / self.std
        
		# Split data
		test_size = int(len(input_datas) * test_ratio)
		train_input = input_datas[:-test_size]
		train_label = label_datas[:-test_size]
		test_input = input_datas[-test_size:]
		test_label = label_datas[-test_size:]
		print(f'Train_input: {train_input.shape}, Train_label: {train_label.shape}, Test_input: {test_input.shape}, Test_label: {test_label.shape}')
    
		# Create data loader
		train_dataset = TensorDataset(train_input, train_label)
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
		# Training
		print(f'Training {self.model_name} model...')
		optimizer = optim.Adam(self.model.parameters(), lr=0.01)
		epoch_losses = []
		test_losses = []
		bar = tqdm(total=num_epochs)
		start_t = time.time()
		for epoch in range(num_epochs):
			running_loss = 0.0
			self.model.train()
			for i, (inputs, labels) in enumerate(train_loader):
				# Forward pass
				outputs = self.model(inputs)
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
			test_loss = self.test(test_input, test_label, criterion, out=False)
			test_losses.append(test_loss)
			bar.set_description(f'{self.model_name} Epoch {epoch+1}')
			bar.set_postfix(loss=f'{epoch_loss:.4f}', test_loss=f'{test_loss:.4f}')
			bar.update(1)
		bar.close()
		end_t = time.time()
		epoch_time = (end_t - start_t) / num_epochs
		print(f'Average Epoch Training Time :{epoch_time}s')
		if save:
			if os.path.exists('model') == False:
				os.makedirs('model')
			torch.save(self.model.state_dict(), f'model/{self.model_name}.pth')
			print(f'Model saved as model/{self.model_name}.pth')
		return epoch_losses, test_losses, epoch_time