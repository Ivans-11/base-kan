# KAN model using wavelet function as basis function
import torch
import torch.nn as nn

def mexican_hat(x):
    return (1 - x ** 2) * torch.exp(-x ** 2 / 2)

def morlet(x):
    return  torch.cos(5 * x) * torch.exp(-x ** 2 / 2)

def dog(x):
    return -x * torch.exp(-x ** 2 / 2)

class CustomWaveletLayer(nn.Module):
    def __init__(self, input_size, output_size, wave_num, wave_type, device='cpu'):
        super(CustomWaveletLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.wave_num = wave_num
        self.wave_type = wave_type
        self.select_func(wave_type)
        self.tanh_range = nn.Parameter(torch.tensor(1.0))
        # Initialize the weights of the propagation matrix
        # self.weights = nn.Parameter(torch.randn(output_size, input_size))
        self.coef = nn.Parameter(torch.randn(output_size, input_size, wave_num)*0.1)
        self.zoom = nn.Parameter(torch.ones(output_size, input_size, wave_num))
        self.pan = nn.Parameter(torch.randn(output_size, input_size, wave_num)*0.5)

        self.to(device)

    def forward(self, x):
        values = (torch.tanh(x * self.tanh_range)).unsqueeze(2)
        transformed_x = torch.stack([
            torch.sum(self.coef[j] * self.func(values * self.zoom[j] - self.pan[j]),dim=2)
            for j in range(self.output_size)
        ], dim=1)
        output = torch.sum(transformed_x, dim=2)
        return output
    
    def to(self, device):
        super(CustomWaveletLayer, self).to(device)
        self.device = device
        return self
    
    def select_func(self, wave_type):
        if wave_type == 'mexican_hat':
            self.func = mexican_hat
        elif wave_type == 'morlet':
            self.func = morlet
        elif wave_type == 'dog':
            self.func = dog
        else:
            raise ValueError('Unknown wavelet type: {}'.format(wave_type))
    

class WaveletKAN(nn.Module):
    """
        KAN model using wavelet function as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            wave_num(optional, int): Number of wavelet basis functions
            wave_type(optional, str): Type of wavelet basis functions
    """
    def __init__(self, layer_sizes, wave_num=2, wave_type='morlet'):
        super(WaveletKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomWaveletLayer(layer_sizes[i-1], layer_sizes[i], wave_num, wave_type))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x
