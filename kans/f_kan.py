# KAN model using Fourier series as basis function
import torch
import torch.nn as nn

class CustomFourierLayer(nn.Module):
    def __init__(self, input_size, output_size, frequency_count, device='cpu'):
        super(CustomFourierLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.frequency_count = frequency_count
        # Initialize the weights of the propagation matrix
        # self.weights = nn.Parameter(torch.randn(output_size, input_size))
        self.coef = nn.Parameter(torch.randn(output_size, input_size, 2 * frequency_count + 1)*0.1)

        self.to(device)

    def forward(self, x):
        values = self.precompute_sin_cos(x)
        transformed_x = torch.stack([
            torch.sum(self.coef[j]*values,dim=2)
            for j in range(self.output_size)
        ], dim=1)
        output = torch.sum(transformed_x, dim=2)
        return output
    
    def to(self, device):
        super(CustomFourierLayer, self).to(device)
        self.device = device
        return self

    def precompute_sin_cos(self, x):
        sin_values = torch.stack([torch.sin((k + 1) * x) for k in range(self.frequency_count)], dim=2)
        cos_values = torch.stack([torch.cos((k + 1) * x) for k in range(self.frequency_count)], dim=2)
        return torch.cat([sin_values, cos_values,torch.ones(sin_values.size(0), self.input_size, 1, device=x.device)], dim=2)

    def increase_frequency(self, new_frequency_count):
        # Dynamically increase the frequency of each Fourier basis function, initialized to 0
        new_coef = torch.zeros(self.output_size, self.input_size, 2 * new_frequency_count + 1, device=self.coef.device)
        new_coef[:,:,:2*self.frequency_count+1] = self.coef
        self.coef = nn.Parameter(new_coef)
        self.frequency_count = new_frequency_count


class FourierKAN(nn.Module):
    """
        KAN model using Fourier series as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            frequency_count(optional, int): Number of frequencies in the Fourier basis function
    """
    def __init__(self, layer_sizes, frequency_count=3):
        super(FourierKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomFourierLayer(layer_sizes[i-1], layer_sizes[i], frequency_count))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x

    def increase_frequency(self, new_frequency_count):
        """
            Method to increase the frequency of the Fourier basis function for each layer
            Args:
                new_frequency_count(int): New number of frequencies in the Fourier basis function
        """
        # Dynamically increase the Fourier basis function frequency for each layer
        for layer in self.layers:
            layer.increase_frequency(new_frequency_count)
