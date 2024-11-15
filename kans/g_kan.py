# KAN model using Gaussian radial basis function as basis function
import torch
import torch.nn as nn

class GaussianBasisFunction(nn.Module):
    def __init__(self, grid_range, grid_count):
        super(GaussianBasisFunction, self).__init__()
        # self.grid_range = grid_range
        self.grid_count = grid_count
        # self.centers = torch.linspace(grid_range[0], grid_range[-1], grid_count)
        # self.width = (grid_range[-1] - grid_range[0]) / ((grid_count - 1) * 2)
        
        # Initialize the coefficients of the Gaussian radial basis function
        self.coefficients = nn.Parameter(torch.randn(grid_count) * 0.1)
    
    def forward(self, exp_values):
        # Calculate the value of the radial basis function using the incoming exp value
        terms = []
        for i in range(self.grid_count):
            terms.append(self.coefficients[i] * exp_values[:,i])
        return sum(terms)


class CustomGaussianLayer(nn.Module):
    def __init__(self, input_size, output_size, grid_range, grid_count):
        super(CustomGaussianLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.grid_range = grid_range
        self.grid_count = grid_count
        
        self.centers = torch.linspace(grid_range[0], grid_range[-1], grid_count)
        self.zoom = (grid_range[-1] - grid_range[0]) / 2
        self.pan = (grid_range[-1] + grid_range[0]) / 2
        self.width = self.zoom / (grid_count - 1)
        
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Define separate Gaussian basis functions for each pair of inputs and outputs
        self.gaussian_bases = nn.ModuleList([
            nn.ModuleList([GaussianBasisFunction(grid_range, grid_count) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        exp_values = self.precompute_exp(x)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.gaussian_bases[j][i](exp_values[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output

    def precompute_exp(self, x):
        x = torch.tanh(x) * self.zoom + self.pan # Ensure inputs are within grid_range
        exp_values = torch.stack([torch.exp(-0.5 * ((x - center) / self.width) ** 2) for center in self.centers], dim=2)
        return exp_values

class GaussianKAN(nn.Module):
    """
        KAN model using Gaussian radial basis function as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            grid_range(optional, list): List of two floats specifying the range of the grid
            grid_count(optional, int): Number of grid points
    """
    def __init__(self, layer_sizes, grid_range=[-1,1], grid_count=6):
        super(GaussianKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomGaussianLayer(layer_sizes[i-1], layer_sizes[i], grid_range, grid_count))
    
    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x