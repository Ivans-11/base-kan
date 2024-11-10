# 使用泰勒级数（多项式系数）作为基函数的KAN模型
import torch
import torch.nn as nn

class TaylorBasisFunction(nn.Module):
    def __init__(self, order):
        super(TaylorBasisFunction, self).__init__()
        self.order = order
        # 初始化泰勒基函数的系数
        self.coefficients = nn.Parameter(torch.randn(order + 1) * 0.1)
    
    def forward(self, x):
        # 使用霍纳方法计算多项式的值
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
        # 初始化传播矩阵的权重
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # 为每对输入输出定义独立的泰勒基函数
        self.taylor_bases = nn.ModuleList([
            nn.ModuleList([TaylorBasisFunction(order) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.taylor_bases[j][i](x[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output


class TaylorKAN(nn.Module):
    def __init__(self, layer_sizes, order):
        super(TaylorKAN, self).__init__()
        self.layers = nn.ModuleList()
        # 构建所有层
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomTaylorLayer(layer_sizes[i-1], layer_sizes[i], order))
    
    def forward(self, x):
        # 逐层计算输出
        for layer in self.layers:
            x = layer(x)
        return x