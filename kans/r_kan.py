# KAN model using rational function as basis function
import torch
import torch.nn as nn

class RationalBasisFunction(nn.Module):
    def __init__(self, mole_order, deno_order):
        super(RationalBasisFunction, self).__init__()
        self.mole_order = mole_order
        self.deno_order = deno_order
        
        self.mole_coefficients = nn.Parameter(torch.randn(mole_order + 1) * 0.1) # Coefficients of molecular polynomials
        self.deno_coefficients = nn.Parameter(torch.randn(deno_order) * 0.1) # Coefficients of denominator polynomials

    def forward(self, x):
        # Use Horner's method to compute values of molecular polynomials
        mole_value = self.mole_coefficients[-1]
        for i in range(self.mole_order - 1, -1, -1):
            mole_value = mole_value * x + self.mole_coefficients[i]
		
		# Use Horner's method to compute values of denominator polynomials
        deno_value = self.deno_coefficients[-1]
        for i in range(self.deno_order - 2, -1, -1):
            deno_value = deno_value * x + self.deno_coefficients[i]
        deno_value = torch.abs(deno_value * x) + 1
        
        return mole_value / deno_value

class CustomRationalLayer(nn.Module):
    def __init__(self, input_size, output_size, mole_order, deno_order):
        super(CustomRationalLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mole_order = mole_order
        self.deno_order = deno_order
        # Initialize the weights of the propagation matrix
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Initialize the tanh range parameter
        self.tanh_range = nn.Parameter(torch.tensor(1.0))
        # Define separate rational basis functions for each pair of inputs and outputs
        self.rational_bases = nn.ModuleList([
            nn.ModuleList([RationalBasisFunction(mole_order, deno_order) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.tanh(x) * self.tanh_range
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
			torch.stack([self.rational_bases[j][i](x[:,i]) for i in range(self.input_size)],dim=1)
			for j in range(self.output_size)
		], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output


class RationalKAN(nn.Module):
    def __init__(self, layer_sizes, mole_order, deno_order):
        super(RationalKAN, self).__init__()
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomRationalLayer(layer_sizes[i-1], layer_sizes[i], mole_order, deno_order))

    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x