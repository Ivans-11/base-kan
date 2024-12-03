# Multi-Layer Perceptrons compared to the KANS model
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
	def __init__(self, input_size, output_size, p_num):
		super(LinearLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.p_num = p_num
		self.tanh_range = nn.Parameter(torch.tensor(1.0))
		# Initialize the weights of the propagation matrix
		# self.weights = nn.Parameter(torch.randn(output_size, input_size))
		# Define separate Linear basis functions for each pair of inputs and outputs
		self.coef = nn.Parameter(torch.randn(output_size, input_size, p_num) * 0.1)
	
	def forward(self, x):
		values = (torch.tanh(x * self.tanh_range)).unsqueeze(2)
		transformed_x = torch.stack([
            torch.sum(self.coef[j] * values, dim=2)
            for j in range(self.output_size)
        ], dim=1)
		output = torch.sum(transformed_x, dim=2)
		return output

class MLP(nn.Module):
	"""
		Multi-Layer Perceptron model with linear basis functions
		Args:
			layer_sizes (list): List of integers representing the number of neurons in each layer
			p_num (optional, int): Number of learnable parameters in the linear basis function
	"""
	def __init__ (self, layer_sizes, p_num=1):
		super(MLP, self).__init__()
		self.layer_sizes = layer_sizes
		self.layers = nn.ModuleList()
		for i in range(1, len(layer_sizes)):
			self.layers.append(LinearLayer(layer_sizes[i-1], layer_sizes[i], p_num))
	
	def forward(self, x):
		for i, layer in enumerate(self.layers):
			if i == len(self.layers) - 1:
				x = layer(x)
			else:
				x = torch.relu(layer(x))
		return x