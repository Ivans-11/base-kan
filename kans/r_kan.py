# KAN model using rational function as basis function
import torch
import torch.nn as nn

class CustomRationalLayer(nn.Module):
    def __init__(self, input_size, output_size, mole_order, deno_order, device='cpu'):
        super(CustomRationalLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mole_order = mole_order
        self.deno_order = deno_order
        # Initialize the weights of the propagation matrix
        # self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Initialize the tanh range parameter
        self.tanh_range = nn.Parameter(torch.tensor(1.0))
        self.mole_coef = nn.Parameter(torch.randn(output_size, input_size, mole_order + 1)*0.1)
        self.deno_coef = nn.Parameter(torch.randn(output_size, input_size, deno_order)*0.1)
        
        self.to(device)

    def forward(self, x):
        mole_values, deno_values = self.precompute_poly(x)
        transformed_x = torch.stack([
            torch.sum(self.mole_coef[j]*mole_values,dim=2) / (torch.abs(torch.sum(self.deno_coef[j]*deno_values,dim=2) * x) + 1)
            for j in range(self.output_size)
        ], dim=1)
        output = torch.sum(transformed_x, dim=2)
        return output

    def to(self, device):
        super(CustomRationalLayer, self).to(device)
        self.device = device
        return self
    
    def precompute_poly(self, x):
        x = torch.tanh(x * self.tanh_range)
        num = max(self.mole_order, self.deno_order)
        values = torch.zeros(x.size(0), self.input_size, num + 1, device=x.device)
        values[:,:,0] = torch.ones(x.size(0), self.input_size, device=x.device)
        values[:,:,1] = x
        for i in range(2, num + 1):
            values[:,:,i] = x * values[:,:,i-1].clone()
        
        mole_values = values[:,:,:self.mole_order + 1]
        deno_values = values[:,:,1:self.deno_order + 1]
        return mole_values, deno_values


class RationalKAN(nn.Module):
    """
        KAN model using rational function as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            mole_order(optional, int): Order of the molecular polynomials
            deno_order(optional, int): Order of the denominator polynomials
    """
    def __init__(self, layer_sizes, mole_order=3, deno_order=2):
        super(RationalKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomRationalLayer(layer_sizes[i-1], layer_sizes[i], mole_order, deno_order))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x
