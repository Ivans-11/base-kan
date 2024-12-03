# KAN model using Taylor series (polynomials) as basis function
import torch
import torch.nn as nn

class CustomTaylorLayer(nn.Module):
    def __init__(self, input_size, output_size, order, device='cpu'):
        super(CustomTaylorLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.order = order
        # Initialize the weights of the propagation matrix
        # self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Initialize the tanh range parameter
        self.tanh_range = nn.Parameter(torch.tensor(1.0))
        self.coef = nn.Parameter(torch.randn(output_size, input_size, order + 1)*0.1)

        self.to(device)

    def forward(self, x):
        poly_values = self.precompute_poly(x)
        transformed_x = torch.stack([
            torch.sum(self.coef[j]*poly_values,dim=2)
            for j in range(self.output_size)
        ], dim=1)
        output = torch.sum(transformed_x, dim=2)
        return output
    
    def to(self, device):
        super(CustomTaylorLayer, self).to(device)
        self.device = device
        return self
    
    def precompute_poly(self, x):
        x = torch.tanh(x * self.tanh_range)
        poly_values = torch.zeros(x.size(0), self.input_size, self.order + 1, device=x.device)
        poly_values[:,:,0] = torch.ones(x.size(0), self.input_size, device=x.device)
        poly_values[:,:,1] = x
        for i in range(2, self.order + 1):
            poly_values[:,:,i] = x * poly_values[:,:,i-1].clone()
        return poly_values


class TaylorKAN(nn.Module):
    """
        KAN model using Taylor series (polynomials) as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            order(optional, int): Order of the Taylor series (polyonomial)
    """
    def __init__(self, layer_sizes, order=5):
        super(TaylorKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomTaylorLayer(layer_sizes[i-1], layer_sizes[i], order))
    
    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x
