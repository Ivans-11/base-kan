# KAN model using wavelet function as basis function
import torch
import torch.nn as nn

def mexican_hat(x):
    return (1 - x ** 2) * torch.exp(-x ** 2 / 2)

def morlet(x):
    return  torch.cos(5 * x) * torch.exp(-x ** 2 / 2)

def dog(x):
    return -x * torch.exp(-x ** 2 / 2)

class WaveletBasisFunction(nn.Module):
    def __init__(self, wave_num, wave_type):
        super(WaveletBasisFunction, self).__init__()
        self.num = wave_num
        self.select_func(wave_type)
        # Initialize the coefficients of the wavelet basis function
        self.coefficients = nn.Parameter(torch.randn(self.num) * 0.1)
        self.zooms = nn.Parameter(torch.ones(self.num))
        self.pans = nn.Parameter(torch.randn(self.num) * 0.5)
    
    def select_func(self, wave_type):
        if wave_type == 'mexican_hat':
            self.func = mexican_hat
        elif wave_type == 'morlet':
            self.func = morlet
        elif wave_type == 'dog':
            self.func = dog
        else:
            raise ValueError('Unknown wavelet type: {}'.format(wave_type))
    
    def forward(self, x):
        # Calculate the value of the wavelet basis function using the incoming x-value
        terms = []
        for i in range(self.num):
            terms.append(self.coefficients[i] * self.func((x - self.pans[i]) / self.zooms[i]))
        return sum(terms)

class CustomWaveletLayer(nn.Module):
    def __init__(self, input_size, output_size, wave_num, wave_type):
        super(CustomWaveletLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.wave_num = wave_num
        self.wave_type = wave_type
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Define separate wavelet basis functions for each pair of inputs and outputs
        self.wavelet_bases = nn.ModuleList([
            nn.ModuleList([WaveletBasisFunction(wave_num, wave_type) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.wavelet_bases[j][i](x[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output
    

class WaveletKAN(nn.Module):
    def __init__(self, layer_sizes, wave_num, wave_type):
        super(WaveletKAN, self).__init__()
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomWaveletLayer(layer_sizes[i-1], layer_sizes[i], wave_num, wave_type))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x