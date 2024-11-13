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

## Start with `notebook` folder

- `train_on_XX.ipynb`: These notebooks demonstrate how to train ,test and compare various KAN models and MLP using PyTorch, including examples on the XX datasets.
- `b_kan_train.ipynb`, `f_kan_train.ipynb`, `g_kan_train.ipynb`, `j_kan_train.ipynb`, `r_kan_train.ipynb`, `t_kan_train.ipynb`, `w_kan_train.ipynb`, `be_kan_train.ipynb`: These notebooks are specialized for training the BSplineKAN, FourierKAN, GaussianKAN, JacobiKAN, RationalKAN, TaylorKAN, WaveletKAN and BernsteinKAN models respectively, showcasing their specific configurations and training procedures.

## KAN

Kolmogorov-Arnold layer:

$${\bf \Phi}= \begin{pmatrix} w_{1,1}\phi_{1,1}(\cdot) & \cdots & w_{1,n_{\rm in}}\phi_{1,n_{\rm in}}(\cdot) \\\\ \vdots & & \vdots \\\\ w_{n_{\rm out},1}\phi_{n_{\rm out},1}(\cdot) & \cdots & w_{n_{\rm out},n_{\rm in}}\phi_{n_{\rm out},n_{\rm in}}(\cdot) \end{pmatrix}$$

Kolmogorov-Arnold network:

$${\rm KAN}({\bf x})={\bf \Phi}_{L-1}\circ\cdots \circ{\bf \Phi}_1\circ{\bf \Phi}_0\circ {\bf x}$$

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

### BernsteinKAN
- Base function: $\phi(x) = \sum_{k=0}^{n} c_k (x-a)^k (b-x)^{n-k}$
- Learnable parameters: The coefficients of the Bernstein polynomials $c_0 ,..., c_n$
- Configurable parameter:  Order of the polynomials $n$, Range of Interpolation $a, b$

## HB-KAN
- Based on the previous test results on various datasets, we found that: the comparison of the performance of various basis functions varies across different types of datasets.

For example, the TaylorKAN and RationalKAN models significantly outperform the other models in the wine dataset, but they perform poorly in the California Housing dataset; the WaveletKAN model has a significant advantage in the Iris dataset, but it does not perform well in the wine dataset and the California Housing dataset; BSplineKAN, which is used in the standard KAN model, performs well in the California Housing dataset, but is slightly inferior in the other datasets.

In addition, JacobiKAN, FourierKAN, and GaussianKAN consistently perform well in the various datasets, and are consistently moderately good or even better.

- Based on the above conclusions, we propose a new modeling architecture: **HB-KAN (Hybrid KAN)**.

Multiple basis functions are used separately for computation, and then the results are weighted and summed to obtain the final output. The weights are learnable parameters that can be automatically adjusted during the training process to assign higher weights to better choices.

For this purpose, we design two HB-KAN model architectures on different levels: **HybridKAN by Layer** and **HybridKAN by Net**

### HybridKAN by Layer
weighting at the layer level (K is the number of basis function species)

$${\bf \Phi_{l}} = \sum_{k=1}^{K} w_k{\bf \Phi}_{l,k}$$

$${\rm HB-KAN}({\bf x})={\bf \Phi}_{L-1}\circ\cdots \circ{\bf \Phi}_1\circ{\bf \Phi}_0\circ {\bf x}$$

### HybridKAN by Net
weighting at the net level (K is the number of basis function species)

$${\rm HB-KAN}({\bf x}) = \sum_{k=1}^{K} w_k {\rm KAN}_{k}({\bf x})$$