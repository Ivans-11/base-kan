# KAN model using Fourier series as basis function
import torch
import torch.nn as nn

class FourierBasisFunction(nn.Module):
    def __init__(self, frequency_count):
        super(FourierBasisFunction, self).__init__()
        self.frequency_count = frequency_count
        
        # Initialize the coefficients of the Fourier basis function
        self.coefficients = nn.Parameter(torch.randn(2 * frequency_count + 1) * 0.1)
        # self.coefficients = nn.Parameter(torch.zeros(2 * frequency_count + 1))
        # self.coefficients = nn.Parameter(torch.randn(2 * frequency_count + 1) / torch.arange(1, 2 * frequency_count + 2))

    def forward(self, sin_values, cos_values):
        # Calculate the value of the Fourier basis function using the incoming sin and cos value
        terms = [self.coefficients[0]]  # a_0
        for k in range(self.frequency_count):
            terms.append(self.coefficients[2 * k + 1] * sin_values[:,k])  # sin((k+1)x)
            terms.append(self.coefficients[2 * k + 2] * cos_values[:,k])  # cos((k+1)x)
        return sum(terms)


class CustomFourierLayer(nn.Module):
    def __init__(self, input_size, output_size, frequency_count):
        super(CustomFourierLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.frequency_count = frequency_count
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Define separate Fourier basis functions for each pair of inputs and outputs
        self.fourier_bases = nn.ModuleList([
            nn.ModuleList([FourierBasisFunction(frequency_count) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        sin_values, cos_values = self.precompute_sin_cos(x)
        output = torch.zeros(batch_size, self.output_size, device=x.device)

        transformed_x = torch.stack([
            torch.stack([self.fourier_bases[j][i](sin_values[:,i], cos_values[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output

    def precompute_sin_cos(self, x):
        sin_values = torch.stack([torch.sin((k + 1) * x) for k in range(self.frequency_count)], dim=2)
        cos_values = torch.stack([torch.cos((k + 1) * x) for k in range(self.frequency_count)], dim=2)
        return sin_values, cos_values

    def increase_frequency(self, new_frequency_count):
        # Dynamically increase the frequency of each Fourier basis function, initialized to 0
        for i in range(self.output_size):
            for j in range(self.input_size):
                old_coeffs = self.fourier_bases[i][j].coefficients.data
                new_coeffs = torch.cat([old_coeffs, torch.zeros(2 * (new_frequency_count - self.fourier_bases[i][j].frequency_count), device=old_coeffs.device)])
                self.fourier_bases[i][j].coefficients = nn.Parameter(new_coeffs)
                self.fourier_bases[i][j].frequency_count = new_frequency_count
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