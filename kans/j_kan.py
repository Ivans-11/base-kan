# KAN model using Jacobi polynomials as basis function
import torch
import torch.nn as nn

class JacobiBasisFunction(nn.Module):
    def __init__(self, order, alpha, beta):
        super(JacobiBasisFunction, self).__init__()
        self.order = order
        self.alpha = alpha
        self.beta = beta
        # 初始化Jacobi基函数的系数
        self.coefficients = nn.Parameter(torch.randn(order + 1) * 0.1)
    
    def forward(self, jacobi_values):
        # 使用传入的Jacobi值计算Jacobi多项式的值
        terms = []
        for i in range(self.order + 1):
            terms.append(self.coefficients[i] * jacobi_values[:,i])
        return sum(terms)


class CustomJacobiLayer(nn.Module):
    def __init__(self, input_size, output_size, order, alpha, beta):
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
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # Define separate Jacobi basis functions for each pair of inputs and outputs
        self.jacobi_bases = nn.ModuleList([
            nn.ModuleList([JacobiBasisFunction(order, alpha, beta) for _ in range(input_size)])
            for _ in range(output_size)
        ])
    
    def set_k(self):
        for i in range(2, self.order + 1):
            self.k1[i] = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            self.k2[i] = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            self.k3[i] = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))

    def forward(self, x):
        batch_size = x.size(0)
        jacobi_values = self.precompute_jacobi(x)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.jacobi_bases[j][i](jacobi_values[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output
    
    def precompute_jacobi(self, x):
        x = torch.tanh(x)
        jacobi_values = torch.zeros(x.size(0), self.input_size, self.order + 1, device=x.device)
        jacobi_values[:,:,0] = torch.ones(x.size(0), self.input_size, device=x.device)
        jacobi_values[:,:,1] = 0.5 * (self.a + self.b + 2) * x - 0.5 * (self.a - self.b)
        for i in range(2, self.order + 1):
            jacobi_values[:,:,i] = (self.k1[i] * x + self.k2[i]) * jacobi_values[:,:,i-1].clone() - self.k3[i] * jacobi_values[:,:,i-2].clone()
        return jacobi_values
    

class JacobiKAN(nn.Module):
    def __init__(self, layer_sizes, order, alpha, beta):
        super(JacobiKAN, self).__init__()
        self.layers = nn.ModuleList()
        # Build all layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomJacobiLayer(layer_sizes[i-1], layer_sizes[i], order, alpha, beta))
            
    def forward(self, x):
        # Calculated output layer-by-layer
        for layer in self.layers:
            x = layer(x)
        return x