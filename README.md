# base-kan

Implementation, Testing and Comparison of Various Basis Functions of the KAN Model

## Introduction

The `base-kan` project focuses on the implementation, testing, and comparison of various basis functions of the KAN model. The KAN model is a framework used in various computational and data analysis applications. This repository aims to provide a comprehensive set of tools and examples to demonstrate the effectiveness of different basis functions within the KAN model.

## Features

- Implementation of various basis functions for the KAN model
- Comprehensive testing framework
- Comparison tools for different basis functions

## Installation

To set up the project and install the necessary dependencies, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/Ivans-11/base-kan.git
   cd base-kan
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Jupyter Notebooks

- `train.ipynb`: This notebook demonstrates how to train and compare various KAN models using PyTorch, including setting up the training and testing functions, and running and comparing the models on synthetic data.
- `b_kan_train.ipynb`, `f_kan_train.ipynb`, `g_kan_train.ipynb`, `j_kan_train.ipynb`, `r_kan_train.ipynb`, `t_kan_train.ipynb`, `w_kan_train.ipynb`: These notebooks are specialized for training the BSplineKAN, FourierKAN, GaussianKAN, JacobiKAN, RationalKAN, TaylorKAN and WaveletKAN models respectively, showcasing their specific configurations and training procedures.

## KAN

Kolmogorov-Arnold layer:

$${\\bf \\Phi}= \\begin{pmatrix} w_{1,1}\\phi_{1,1}(\\cdot) & \\cdots & w_{1,n_{\\rm in}}\\phi_{1,n_{\\rm in}}(\\cdot) \\\\ \\vdots & & \\vdots \\\\ w_{n_{\\rm out},1}\\phi_{n_{\\rm out},1}(\\cdot) & \\cdots & w_{n_{\\rm out},n_{\\rm in}}\\phi_{n_{\\rm out},n_{\\rm in}}(\\cdot) \\end{pmatrix}$$

Kolmogorov-Arnold network:

$${\\rm KAN}({\\bf x})={\\bf \\Phi}_{L-1}\\circ\\cdots \\circ{\\bf \\Phi}_1\\circ{\\bf \\Phi}_0\\circ {\\bf x}$$

## Base Fuction

The KAN model's underlying architecture code is located in the `kans` folder. Their basis functions $\phi(x)$ are as follows

### BSplineKAN
- Base function: $\phi(x) = \sum_{i=1}^{n} c_i B_{i,k}(x)$
- Learnable parameters: The coefficients of control points $c_1 ,..., c_n$
- Configurable parameter: Grid count $n$, Order of the B-spline function $k$, The control points are determined by grid count and grid range.

### FourierKAN
- Base function: $\phi(x) = a_0 + \sum_{k=1}^{n} \left( a_k \cos(kx) + b_k \sin(kx) \right)$
- Learnable parameters: The coefficients of the Fourier series $a_0, a_1, b_1 ,..., a_n, b_n$
- Configurable parameter: Frequency limit $n$

### GaussianKAN
- Base function: $\phi(x) = \sum_{i=1}^{n} a_i \exp\left(-\frac{(x - \mu_i)^2}{2 \sigma_i^2}\right)$
- Learnable parameters: The coefficients $a_1,..., a_n$
- Configurable parameter: Grid count $n$, Parameters controlled by grid count and grid range $\mu_i,\sigma_i$

### JacobiKAN
- Base function: $\phi(x) = \sum_{k=0}^{n} c_k P_k^{(\alpha, \beta)}(x)$
- Learnable parameters: The coefficients of the Jacobi polynomials $c_0 ,..., c_n$
- Configurable parameter: Maximum order $n$, Parameters of Jacobi polynomials $\alpha, \beta$

### RationalKAN
- Base function: $\phi(x) = \frac{\sum_{i=0}^{m} a_i x^i}{1 + \lvert\sum_{j=1}^{n} b_j x^j\rvert}$
- Learnable parameters: The coefficients of the polynomials $a_i, b_j$
- Configurable parameter: Order of the numerator $m$, Order of the denominator $n$

### TaylorKAN
- Base function: $\phi(x) = \sum_{k=0}^{n} c_k x^k$
- Learnable parameters: The coefficients of the polynomial $c_0 ,..., c_n$
- Configurable parameter: Polynomial order $n$

### WaveletKAN
- Base function: $\phi(x) = \sum_{i=1}^{n} a_i \psi\left(\frac{x - b_i}{s_i}\right)$
- Learnable parameters: Magnitude, scale and translation parameters $a_i, b_i, s_i$
- Configurable parameter: Wave number $n$, Type of $\psi()$ including `'mexican_hat'`,`'morlet'`,`'dog'`