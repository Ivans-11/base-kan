# KAN model using Jacobi polynomials as basis function
import torch
import torch.nn as nn

class CustomJacobiLayer(nn.Module):
    def __init__(self, input_size, output_size, order, alpha, beta, device='cpu'):
        super(CustomJacobiLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.order = order
        self.a = alpha
        self.b = beta
        
        self.k1 = torch.zeros(order + 1, dtype=torch.float32)
        self.k2 = torch.zeros(order + 1, dtype=torch.float32)
        self.k3 = torch.zeros(order + 1, dtype=torch.float32)
        self.set_k()
        
        # Initialize the weights of the propagation matrix
        # self.weights = nn.Parameter(torch.randn(output_size, input_size))
        self.coef = nn.Parameter(torch.randn(output_size, input_size, order + 1)*0.1)
        
        self.to(device)
    
    def set_k(self):
        for i in range(2, self.order + 1):
            self.k1[i] = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            self.k2[i] = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            self.k3[i] = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))

    def forward(self, x):
        jacobi_values = self.precompute_jacobi(x)
        transformed_x = torch.stack([
            torch.sum(self.coef[j]*jacobi_values,dim=2)
            for j in range(self.output_size)
        ], dim=1)
        output = torch.sum(transformed_x, dim=2)
        return output

    def to(self, device):
        super(CustomJacobiLayer, self).to(device)
        self.device = device
        return self
    
    def precompute_jacobi(self, x):
        x = torch.tanh(x)
        jacobi_values = torch.zeros(x.size(0), self.input_size, self.order + 1, device=x.device)
        jacobi_values[:,:,0] = torch.ones(x.size(0), self.input_size, device=x.device)
        jacobi_values[:,:,1] = 0.5 * (self.a + self.b + 2) * x - 0.5 * (self.a - self.b)
        for i in range(2, self.order + 1):
            jacobi_values[:,:,i] = (self.k1[i] * x + self.k2[i]) * jacobi_values[:,:,i-1].clone() - self.k3[i] * jacobi_values[:,:,i-2].clone()
        return jacobi_values
    

class JacobiKAN(nn.Module):
    """
        KAN model using Jacobi polynomials as basis function
        Args:
            layer_sizes(list): List of integers specifying the number of neurons in each layer
            order(optional, int): Order of the Jacobi polynomials
            alpha(optional, float): Alpha parameter of the Jacobi polynomials
            beta(optional, float): Beta parameter of the Jacobi polynomials
    """
    def __init__(self, layer_sizes, order=5, alpha=0.5, beta=0.5):
        super(JacobiKAN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomJacobiLayer(layer_sizes[i-1], layer_sizes[i], order, alpha, beta))
            
    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x
