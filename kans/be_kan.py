# KAN model using Bernsteins polynomials as basis function
import torch
import torch.nn as nn

class BernsteinBasisFunction(nn.Module):
    def __init__(self, order, inter_range):
        super(BernsteinBasisFunction, self).__init__()
        self.order = order
        # self.range = inter_range
        # Initialize the coefficients of the Bernstein basis function
        self.coefficients = nn.Parameter(torch.randn(order + 1) * 0.1)
    
    def forward(self, ber_values):
        # Calculate the value of the Bernstein basis function using the incoming Bernstein value
        terms = []
        for i in range(self.order + 1):
            terms.append(self.coefficients[i] * ber_values[:,i])
        return sum(terms)
    

class CustomBernsteinLayer(nn.Module):
    def __init__(self, input_size, output_size, order, inter_range):
        super(CustomBernsteinLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.order = order
        self.range = inter_range
        
        self.zoom = (inter_range[-1] - inter_range[0]) / 2
        self.pan = (inter_range[-1] + inter_range[0]) / 2
        
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Define separate Bernstein basis functions for each pair of inputs and outputs
        self.bernstein_bases = nn.ModuleList([
            nn.ModuleList([BernsteinBasisFunction(order, inter_range) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        ber_values = self.precompute_bernstein(x)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.bernstein_bases[j][i](ber_values[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output

    def precompute_bernstein(self, x):
        x = torch.tanh(x) * self.zoom + self.pan # Ensure inputs are within inter_range
        x_a = x - self.range[0]
        x_b = self.range[-1] - x
        x_a_n = torch.ones(x.size(0), x.size(1), self.order + 1, device=x.device)
        x_b_n = torch.ones(x.size(0), x.size(1), self.order + 1, device=x.device)
        for i in range(1, self.order + 1):
            x_a_n[:,:,i] = x_a_n[:,:,i-1].clone() * x_a
            x_b_n[:,:,i] = x_b_n[:,:,i-1].clone() * x_b
        ber_values = torch.stack([x_a_n[:,:,i] * x_b_n[:,:,self.order - i] for i in range(self.order + 1)], dim=2)
        return ber_values


class BernsteinKAN(nn.Module):
    """
        KAN model using Bernstein polynomials as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            order(optional, int): Order of the Bernstein polynomials
            inter_range(optional, list): List of two floats specifying the interpolation range of the Bernstein polynomials
    """
    def __init__(self, layer_sizes, order=5, inter_range=[0,1]):
        super(BernsteinKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomBernsteinLayer(layer_sizes[i-1], layer_sizes[i], order, inter_range))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x