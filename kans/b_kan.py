# 使用B-spline函数作为基函数的KAN模型
import torch
import torch.nn as nn

class BSplineBasisFunction(nn.Module):
    def __init__(self, order, grid_range, grid_count):
        super(BSplineBasisFunction, self).__init__()
        # self.order = order
        # self.grid_range = grid_range
        self.grid_count = grid_count
        
        # 初始化B-spline基函数的系数
        self.coefficients = nn.Parameter(torch.randn(grid_count) * 0.1)
    
    def forward(self, spline_value):
        # 使用传入的spline值计算B-spline函数的值
        terms = []
        for i in range(self.grid_count):
            terms.append(self.coefficients[i] * spline_value[:,i])
        return sum(terms)
    

class CustomBSplineLayer(nn.Module):
    def __init__(self, input_size, output_size, order, grid_range, grid_count):
        super(CustomBSplineLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.order = order
        self.grid_range = grid_range
        self.grid_count = grid_count
        
        self.center_count = grid_count + order + 1
        end_center = grid_range[-1] + (grid_range[-1] - grid_range[0]) * (order + 1) / (grid_count - 1)
        self.centers = torch.linspace(grid_range[0], end_center, self.center_count)
        
        # 初始化传播矩阵的权重
        self.weights = nn.Parameter(torch.randn(output_size, input_size))
        # 为每对输入输出定义独立的B-spline基函数
        self.bspline_bases = nn.ModuleList([
            nn.ModuleList([BSplineBasisFunction(order, grid_range, grid_count) for _ in range(input_size)])
            for _ in range(output_size)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        spline_values = self.precompute_spline(x)
        output = torch.zeros(batch_size, self.output_size, device=x.device)
        transformed_x = torch.stack([
            torch.stack([self.bspline_bases[j][i](spline_values[:,i]) for i in range(self.input_size)],dim=1)
            for j in range(self.output_size)
        ], dim=1)
        output += torch.einsum('boi,oi->bo', transformed_x, self.weights)
        return output

    def precompute_spline(self, x):
        x = torch.clamp(x, self.grid_range[0], self.grid_range[-1]) # 确保输入在grid_range内
        last_b = torch.stack([(x >= self.centers[i]) * (x < self.centers[i+1]) for i in range(self.center_count - 1)], dim=2)
        for k in range(1, self.order + 1):
            new_b = torch.stack([(x - self.centers[i]) / (self.centers[i+k] - self.centers[i]) * last_b[:,:,i] + (self.centers[i+k+1] - x) / (self.centers[i+k+1] - self.centers[i+1]) * last_b[:,:,i+1] for i in range(self.center_count - k - 1)], dim=2)
            last_b = new_b
        spline_values = last_b
        return spline_values
    

class BSplineKAN(nn.Module):
    def __init__(self, layer_sizes, order, grid_range, grid_count):
        super(BSplineKAN, self).__init__()
        self.layers = nn.ModuleList()
        # 构建所有层
        for i in range(1, len(layer_sizes)):
            self.layers.append(CustomBSplineLayer(layer_sizes[i-1], layer_sizes[i], order, grid_range, grid_count))
    
    def forward(self, x):
        # 逐层计算输出
        for layer in self.layers:
            x = layer(x)
        return x