# KAN models combining multiple basis functions
from ctypes.wintypes import tagPOINT
import torch
import torch.nn as nn
from .b_kan import CustomBSplineLayer, BSplineKAN
from .f_kan import CustomFourierLayer, FourierKAN
from .g_kan import CustomGaussianLayer, GaussianKAN
from .j_kan import CustomJacobiLayer, JacobiKAN
from .r_kan import CustomRationalLayer, RationalKAN
from .t_kan import CustomTaylorLayer, TaylorKAN
from .w_kan import CustomWaveletLayer, WaveletKAN
from .be_kan import CustomBernsteinLayer, BernsteinKAN
from .mlp import MLP

class CustomHybridLayer(nn.Module):
	def __init__(self, input_size, output_size, basis_functions=[], **kwargs):
		super(CustomHybridLayer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.basis_functions = basis_functions
		self.bf_num = len(basis_functions)
		if basis_functions:
			self.contributions = nn.Parameter(torch.full((self.bf_num,), 1.0 / self.bf_num))
		else:
			self.contributions = nn.Parameter(torch.empty(0))
		self.layers = nn.ModuleList()
		self.set_layer(basis_functions, **kwargs)
		
	def set_layer(self, basis_functions, **kwargs):
		for bf in basis_functions:
			bf = bf.lower()
			if bf == 'bspline' or bf == 'b-spline' or bf == 'bsplinekan' or bf == 'b_kan':
				self.layers.append(CustomBSplineLayer(self.input_size, self.output_size, kwargs.get('b_order', 3), kwargs.get('b_grid_range', [-1, 1]), kwargs.get('b_grid_count', 6)))
			elif bf == 'fourier' or bf == 'fourierkan' or bf == 'f_kan':
				self.layers.append(CustomFourierLayer(self.input_size, self.output_size, kwargs.get('frequency_count', 3)))
			elif bf == 'gaussian' or bf == 'gaussiankan' or bf == 'g_kan':
				self.layers.append(CustomGaussianLayer(self.input_size, self.output_size, kwargs.get('g_grid_range', [-1, 1]), kwargs.get('g_grid_count', 6)))
			elif bf == 'jacobi' or bf == 'jacobikan' or bf == 'j_kan':
				self.layers.append(CustomJacobiLayer(self.input_size, self.output_size, kwargs.get('j_order', 5), kwargs.get('alpha', 0.5), kwargs.get('beta', 0.5)))
			elif bf == 'rational' or bf == 'rationalkan' or bf == 'r_kan':
				self.layers.append(CustomRationalLayer(self.input_size, self.output_size, kwargs.get('mole_order', 3), kwargs.get('deno_order', 2)))
			elif bf == 'taylor' or bf == 'taylorkan' or bf == 't_kan':
				self.layers.append(CustomTaylorLayer(self.input_size, self.output_size, kwargs.get('t_order', 5)))
			elif bf == 'wavelet' or bf == 'waveletkan' or bf == 'w_kan':
				self.layers.append(CustomWaveletLayer(self.input_size, self.output_size, kwargs.get('wave_num', 2), kwargs.get('wave_type', 'morlet')))
			elif bf == 'bernstein' or bf == 'bernsteinkan' or bf == 'be_kan':
				self.layers.append(CustomBernsteinLayer(self.input_size, self.output_size, kwargs.get('be_order', 5), kwargs.get('inter_range', [0, 1])))
			elif bf == 'mlp' or bf == 'linear':
				self.layers.append(nn.Linear(self.input_size, self.output_size))
			else:
				raise ValueError('Unknown basis function: {}'.format(bf))
	
	def forward(self, x):
		batch_size = x.size(0)
		output = torch.zeros(batch_size, self.output_size, device=x.device)
		for i, layer in enumerate(self.layers):
			output += self.contributions[i] * layer(x)
		return output
	
	def add_layer(self, basis_functions, **kwargs):
		self.bf_num += len(basis_functions)
		self.contributions = nn.Parameter(torch.cat((self.contributions, torch.randn(len(basis_functions))*0.1), dim=0))
		self.set_layer(basis_functions, **kwargs)


class HybridKANbyLayer(nn.Module):
	"""
		Combines multiple KAN models using different basis functions in each layer
		Args:
			layer_sizes (list): List of integers specifying the number of neurons in each layer
			basis_functions (optional, list): List of strings specifying the basis functions to be used in each layer
			**kwargs: Additional arguments for each basis function
		Basis functions:
			- B-spline (bspline, b-spline, bsplinekan, b_kan)
			- Fourier (fourier, fourierkan, f_kan)
			- Gaussian (gaussian, gaussiankan, g_kan)
			- Jacobi (jacobi, jacobikan, j_kan)
			- Rational (rational, rationalkan, r_kan)
			- Taylor (taylor, taylorkan, t_kan)
			- Wavelet (wavelet, waveletkan, w_kan)
			- Bernstein (bernstein, bernsteinkan, be_kan)
			- MLP (mlp, linear)
		Kwargs:
			b_order (int): Order of B-spline basis function
			b_grid_range (list): Grid range of B-spline basis function
			b_grid_count (int): Grid count of B-spline basis function
			frequency_count (int): Frequency count of Fourier series
			g_grid_range (list): Grid range of Gaussian radial basis function
			g_grid_count (int): Grid count of Gaussian radial basis function
			j_order (int): Order of Jacobi polynomial
			alpha (float): Alpha of Jacobi polynomial
			beta (float): Beta of Jacobi polynomial
			mole_order (int): Order of numerator in Rational function
			deno_order (int): Order of denominator in Rational function
			t_order (int): Order of Taylor polynomial
			wave_num (int): Number of wavelets
			wave_type (str): Type of wavelet
			be_order (int): Order of Bernstein polynomial
			inter_range (list): Interpolation range of Bernstein polynomial
	"""
	def __init__(self, layer_sizes, basis_functions, **kwargs):
		super(HybridKANbyLayer, self).__init__()
		self.layer_sizes = layer_sizes
		self.layers = nn.ModuleList()
		for i in range(1, len(layer_sizes)):
			self.layers.append(CustomHybridLayer(layer_sizes[i-1], layer_sizes[i], basis_functions, **kwargs))
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	
	def show_contributions(self):
		"""
            Method to show the contributions of each basis function in each layer
        """
		print('**Contributions of each basis function in each layer:**')
		for i, layer in enumerate(self.layers):
			print('Layer {}:'.format(i+1))
			contributions_sum = layer.contributions.sum()
			for j, contribution in enumerate(layer.contributions):
				print('Contribution of {}: {:.2f}'.format(layer.basis_functions[j], contribution.item()/contributions_sum))
			print()
	
	def add_base_functions(self, basis_functions, **kwargs):
		"""
			Method to add basis functions to the network
			Args:
                basis_functions (list): List of strings specifying the basis functions to be added
                **kwargs: Additional arguments for each basis function
		"""
		for layer in self.layers:
			layer.add_layer(basis_functions, **kwargs)


class HybridKANbyNet(nn.Module):
	"""
        Combines multiple KAN models using different basis functions in each network
        Args:
            layer_sizes (list): List of integers specifying the number of neurons in each layer
            basis_functions (optional, list): List of strings specifying the basis functions to be used in each network
            **kwargs: Additional arguments for each basis function
        Basis functions:
            - B-spline (bspline, b-spline, bsplinekan, b_kan)
            - Fourier (fourier, fourierkan, f_kan)
            - Gaussian (gaussian, gaussiankan, g_kan)
            - Jacobi (jacobi, jacobikan, j_kan)
            - Rational (rational, rationalkan, r_kan)
            - Taylor (taylor, taylorkan, t_kan)
            - Wavelet (wavelet, waveletkan, w_kan)
            - Bernstein (bernstein, bernsteinkan, be_kan)
			- MLP (mlp)
        Kwargs:
            b_order (int): Order of B-spline basis function
            b_grid_range (list): Grid range of B-spline basis function
            b_grid_count (int): Grid count of B-spline basis function
            frequency_count (int): Frequency count of Fourier series
            g_grid_range (list): Grid range of Gaussian radial basis function
            g_grid_count (int): Grid count of Gaussian radial basis function
            j_order (int): Order of Jacobi polynomial
            alpha (float): Alpha of Jacobi polynomial
            beta (float): Beta of Jacobi polynomial
            mole_order (int): Order of numerator in Rational function
            deno_order (int): Order of denominator in Rational function
            t_order (int): Order of Taylor polynomial
            wave_num (int): Number of wavelets
            wave_type (str): Type of wavelet
            be_order (int): Order of Bernstein polynomial
            inter_range (list): Interpolation range of Bernstein polynomial
			p_num (int): Number of learnable parameters per input-output pair in MLP
    """
	def __init__(self, layer_sizes, basis_functions=[], **kwargs):
		super(HybridKANbyNet, self).__init__()
		self.layer_sizes = layer_sizes
		self.basis_functions = basis_functions
		self.bf_num = len(basis_functions)
		if basis_functions:
			self.contributions = nn.Parameter(torch.full((self.bf_num,), 1.0 / self.bf_num))
		else:
			self.contributions = nn.Parameter(torch.empty(0))
		self.nets = nn.ModuleList()
		self.set_net(layer_sizes, basis_functions, **kwargs)
        
	def set_net(self, layer_sizes, basis_functions, **kwargs):
		for bf in basis_functions:
			bf = bf.lower()
			if bf == 'bspline' or bf == 'b-spline' or bf == 'bsplinekan' or bf == 'b_kan':
				self.nets.append(BSplineKAN(layer_sizes, kwargs.get('b_order', 3), kwargs.get('b_grid_range', [-1, 1]), kwargs.get('b_grid_count',6)))
			elif bf == 'fourier' or bf == 'fourierkan' or bf == 'f_kan':
				self.nets.append(FourierKAN(layer_sizes, kwargs.get('frequency_count', 3)))
			elif bf == 'gaussian' or bf == 'gaussiankan' or bf == 'g_kan':
				self.nets.append(GaussianKAN(layer_sizes, kwargs.get('g_grid_range', [-1, 1]), kwargs.get('g_grid_count', 6)))
			elif bf == 'jacobi' or bf == 'jacobikan' or bf == 'j_kan':
				self.nets.append(JacobiKAN(layer_sizes, kwargs.get('j_order', 5), kwargs.get('alpha', 0.5), kwargs.get('beta', 0.5)))
			elif bf == 'rational' or bf == 'rationalkan' or bf == 'r_kan':
				self.nets.append(RationalKAN(layer_sizes, kwargs.get('mole_order', 3), kwargs.get('deno_order', 2)))
			elif bf == 'taylor' or bf == 'taylorkan' or bf == 't_kan':
				self.nets.append(TaylorKAN(layer_sizes, kwargs.get('t_order', 5)))
			elif bf == 'wavelet' or bf == 'waveletkan' or bf == 'w_kan':
				self.nets.append(WaveletKAN(layer_sizes, kwargs.get('wave_num', 2), kwargs.get('wave_type', 'morlet')))
			elif bf == 'bernstein' or bf == 'bernsteinkan' or bf == 'be_kan':
				self.nets.append(BernsteinKAN(layer_sizes, kwargs.get('be_order', 5), kwargs.get('inter_range', [0, 1])))
			elif bf == 'mlp':
				self.nets.append(MLP(layer_sizes, kwargs.get('p_num', 1)))
			else:
				raise ValueError('Unknown basis function: {}'.format(bf))
	
	def forward(self, x):
		for i, net in enumerate(self.nets):
			if i == 0:
				output = self.contributions[i] * net(x)
			else:
				output += self.contributions[i] * net(x)
		return output
	
	def show_contributions(self):
		"""
			Method to show the contributions of each basis function
		"""	
		print('**Contributions of each basis function:**')
		contribution_sum = self.contributions.sum()
		for i, contribution in enumerate(self.contributions):
			print('Contribution of {}: {:.2f}'.format(self.basis_functions[i], contribution.item()/contribution_sum))
		print()
		
	def add_base_functions(self, basis_functions, **kwargs):
		"""
			Method to add basis functions to the network
			Args:
                basis_functions (list): List of strings specifying the basis functions to be added
                **kwargs: Additional arguments for each basis function
		"""
		self.bf_num += len(basis_functions)
		self.contributions = nn.Parameter(torch.cat((self.contributions, torch.randn(len(basis_functions))*0.1), dim=0))
		self.set_net(self.layer_sizes, basis_functions, **kwargs)
		
	def add_models(self, models):
		"""
            Method to add KAN models to the network
            Args:
                models (list): List of KAN models to be added
        """
		if self.bf_num == 0:
			self.bf_num = len(models)
			self.contributions = nn.Parameter(torch.full((self.bf_num,), 1.0 / self.bf_num))
			for model in models:
				if model.layer_sizes != self.layer_sizes:
					raise ValueError('Layer sizes of the added model do not match.')
				self.nets.append(model)
		else:
			for model in models:
				if model.layer_sizes != self.layer_sizes:
					raise ValueError('Layer sizes of the added model do not match.')
				self.bf_num += 1
				self.contributions = nn.Parameter(torch.cat((self.contributions, torch.randn(1)*0.1), dim=0))
				self.nets.append(model)