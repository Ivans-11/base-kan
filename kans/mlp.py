# Multi-Layer Perceptrons compared to the KANS model
import torch
import torch.nn as nn

class MLP(nn.Module):
	def __init__ (self, layer_sizes):
		super(MLP, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(1, len(layer_sizes)):
			self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
	
	def forward(self, x):
		for layer in self.layers:
			x = torch.relu(layer(x))
		return x