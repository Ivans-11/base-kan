# Multi-Layer Perceptrons compared to the KANS model
import torch
import torch.nn as nn

class LinearBasisFunction(nn.Module):
	def __init__(self, p_num):
		super(LinearBasisFunction, self).__init__()
		self.p_num = p_num # Number of learnable parameters
		
        # Initialize the coefficients
		self.coefficients = nn.Parameter(torch.randn(p_num) * 0.1)
    
	def forward(self, x):
		coef_sum = torch.sum(self.coefficients)
		value = x * coef_sum
		return value
        

class LinearLayer(nn.Module):
	def __init__(self, input_size, output_size, p_num):
		super(LinearLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.p_num = p_num
		# Initialize the weights of the propagation matrix
		self.weights = nn.Parameter(torch.randn(output_size, input_size))
		# Define separate Linear basis functions for each pair of inputs and outputs
		self.linear_bases = nn.ModuleList([
            nn.ModuleList([LinearBasisFunction(p_num) for _ in range(input_size)])
			for _ in range(output_size)
        ])
	
	def forward(self, x):
		batch_size = x.size(0)
		output = torch.zeros(batch_size, self.output_size, device=x.device)
		transformed_x = torch.stack([
            torch.stack([self.linear_bases[j][i](x[:,i]) for i in range(self.input_size)], dim=1)
            for j in range(self.output_size)
        ], dim=1)
		output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
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