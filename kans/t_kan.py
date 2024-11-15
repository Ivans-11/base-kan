# KAN model using Taylor series (polynomials) as basis function
import torch
import torch.nn as nn

class TaylorBasisFunction(nn.Module):
    def __init__(self, order):
        super(TaylorBasisFunction, self).__init__()
        self.order = order
        # Initialize the coefficients of the Taylor basis function
        self.coefficients = nn.Parameter(torch.randn(order + 1) * 0.1)
    
    def forward(self, x):
        # Use Horner's method to compute values of polynomials
        value = self.coefficients[-1]
        for i in range(self.order - 1, -1, -1):
            value = value * x + self.coefficients[i]
        return value
    

class CustomTaylorLayer(nn.Module):
    def __init__(self, input_size, output_size, order):
        super(CustomTaylorLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.order = order
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Initialize the tanh range parameter
        self.tanh_range = nn.Parameter(torch.tensor(1.0))
        # Define separate Taylor basis functions for each pair of inputs and outputs
        self.taylor_bases = nn.ModuleList([
            nn.ModuleList([TaylorBasisFunction(order) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.tanh(x) * self.tanh_range
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.taylor_bases[j][i](x[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output


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