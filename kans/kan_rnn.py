# build a simple RNN model with KAN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from .b_kan import CustomBSplineLayer
from .f_kan import CustomFourierLayer
from .g_kan import CustomGaussianLayer
from .j_kan import CustomJacobiLayer
from .r_kan import CustomRationalLayer
from .t_kan import CustomTaylorLayer
from .w_kan import CustomWaveletLayer
from .be_kan import CustomBernsteinLayer

class KAN_RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, basis_i2h, basis_h2o, name='model', active=False, **kwargs):
		super(KAN_RNN, self).__init__()
		self.hidden_size = hidden_size
		self.basis_i2h = basis_i2h
		self.basis_h2o = basis_h2o
		self.i2h = self.generate_layer(input_size+hidden_size, hidden_size, basis_i2h, **kwargs)
		self.h2o = self.generate_layer(hidden_size, output_size, basis_h2o, **kwargs)
		self.name = name
		self.active = active
    
	def generate_layer(self, input_size, output_size, basis, **kwargs):
		bf = basis.lower()
		if bf == 'bspline' or bf == 'b-spline' or bf == 'bsplinekan' or bf == 'b_kan':
			return CustomBSplineLayer(input_size, output_size, kwargs.get('b_order', 3), kwargs.get('b_grid_range', [-1, 1]), kwargs.get('b_grid_count', 6))
		elif bf == 'fourier' or bf == 'fourierkan' or bf == 'f_kan':
			return CustomFourierLayer(input_size, output_size, kwargs.get('frequency_count', 3))
		elif bf == 'gaussian' or bf == 'gaussiankan' or bf == 'g_kan':
			return CustomGaussianLayer(input_size, output_size, kwargs.get('g_grid_range', [-1, 1]), kwargs.get('g_grid_count', 6))
		elif bf == 'jacobi' or bf == 'jacobikan' or bf == 'j_kan':
			return CustomJacobiLayer(input_size, output_size, kwargs.get('j_order', 5), kwargs.get('alpha', 0.5), kwargs.get('beta', 0.5))
		elif bf == 'rational' or bf == 'rationalkan' or bf == 'r_kan':
			return CustomRationalLayer(input_size, output_size, kwargs.get('mole_order', 3), kwargs.get('deno_order', 2))
		elif bf == 'taylor' or bf == 'taylorkan' or bf == 't_kan':
			return CustomTaylorLayer(input_size, output_size, kwargs.get('t_order', 5))
		elif bf == 'wavelet' or bf == 'waveletkan' or bf == 'w_kan':
			return CustomWaveletLayer(input_size, output_size, kwargs.get('wave_num', 2), kwargs.get('wave_type', 'morlet'))
		elif bf == 'bernstein' or bf == 'bernsteinkan' or bf == 'be_kan':
			return CustomBernsteinLayer(input_size, output_size, kwargs.get('be_order', 5), kwargs.get('inter_range', [0, 1]))
		elif bf == 'mlp' or bf == 'linear':
			return nn.Linear(input_size, output_size)
		else:
			raise ValueError('Unknown basis function: {}'.format(bf))

	def forward(self, x):
		# input to hidden
		i2h_in = torch.cat((x, self.hidden), 1)
		if self.active:
			self.hidden = torch.tanh(self.i2h(i2h_in))
		else:
			self.hidden = self.i2h(i2h_in)
		
		# hidden to output
		out = self.h2o(self.hidden)
		return out
	
	def clear_hidden(self):
		self.hidden = torch.zeros(self.hidden_size).unsqueeze(0)
	
	def train_series(self, series, test_ratio=0.2, num_epochs=100, criterion=nn.MSELoss(), save=True):
		print('-'*10, self.name, '-'*10)
		
		# split the data
		test_size = int(len(series) * test_ratio)
		print(f'Target: predicting {test_size} future steps from {len(series)-test_size} historical steps with {series.size(1)} channels')
		train_input = series[:-test_size-1]
		train_label = series[1:-test_size]
		test_input = series[-test_size-1:-test_size]
		test_label = series[-test_size:]
        
        # train the model
		print("Training the model...")
		optimizer = optim.Adam(self.parameters(), lr=0.01)
		epoch_losses = []
		test_losses = []
		bar = tqdm(total=num_epochs)
		start_t = time.time()
		for epoch in range(num_epochs):
			self.clear_hidden()
			running_loss = 0.0
			self.train()
			optimizer.zero_grad()
			for i in range(len(train_input)):
				output = self(train_input[i:i+1])
				loss = criterion(output, train_label[i:i+1])
				running_loss += loss
			running_loss.backward()
			optimizer.step()
			epoch_loss = running_loss.item() / len(train_input)
			epoch_losses.append(epoch_loss)
			test_loss = self.test_series(test_input, test_label, criterion)
			test_losses.append(test_loss)
			bar.set_description(f'{self.name} Epoch {epoch+1}')
			bar.set_postfix(loss=f'{epoch_loss:.4f}', test_loss=f'{test_loss:.4f}')
			bar.update(1)
		bar.close()
		end_t = time.time()
		epoch_time = (end_t - start_t) / num_epochs
		print(f'Average Epoch Training Time :{epoch_time}s')
		predicts = self.plot(series, train_input, train_label, test_input, test_label, criterion)
		if save:
			if os.path.exists('model') == False:
				os.makedirs('model')
			torch.save(self.state_dict(), f'model/{self.name}.pth')
			print(f'Model saved as model/{self.name}.pth')
		return epoch_losses, test_losses, epoch_time, predicts
	
	def test_series(self, test_input, test_label, criterion=nn.MSELoss()):
		self.eval()
		predicts = torch.empty(test_label.size())
		with torch.no_grad():
			for i in range(len(test_label)):
				output = self(test_input)
				predicts[i] = output
				test_input = output
			loss = criterion(predicts, test_label)
		return loss
	
	def plot(self, series, train_input, train_label, test_input, test_label, criterion=nn.MSELoss()):
		self.eval()
		self.clear_hidden()
		fits = torch.empty(train_label.size())
		predicts = torch.empty(test_label.size())
		with torch.no_grad():
			for i in range(len(train_input)):
				output = self(train_input[i:i+1])
				fits[i] = output
			for i in range(len(test_label)):
				output = self(test_input)
				predicts[i] = output
				test_input = output
		for i in range(series.size(1)):
			plt.figure(figsize=(8, 6))
			plt.plot(series[:,i], label='True Data', color='b', linestyle='dashed', alpha=0.5)
			plt.plot(np.arange(1, len(fits)+1), fits[:,i], label='Fitting', color='r')
			plt.plot(np.arange(len(fits)+1, len(fits)+len(predicts)+1), predicts[:,i], label='Prediction', color='g')
			plt.legend()
			plt.xlabel('Time Steps')
			plt.ylabel('Value')
			plt.title(f'{self.name} Channel {i} Prediction')
			plt.grid(True)
			plt.show()
		return torch.cat((series[:1], fits, predicts), 0)