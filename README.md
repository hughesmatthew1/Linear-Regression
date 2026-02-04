# Linear Regression From Scratch

## Overview

This project implements **linear regression** with only the use of the **NumPy** library. The goal is to demonstrate a clear understanding of the mathematical underlyings of linear regression without reliance upon high-level libraries such as scikit-learn, TensorFlow, or PyTorch.

The implementation supports training a model using various optimization strategies, learning-rate schedules, and regularization techniques.

## Features
* Ordinary Least Squares (OLS) Regression
* Optimization methods:
  * Normal Equation
  * Batch Gradient Descent (bgd)
  * Mini-Batch Gradient Descent (mbgd)
  * Stochastic Gradient Descent (sgd)
* Learning rate schedules:
  * Constant learning rate
  * Inverse time decay
* Regularization
  * L1 (Lasso)
  * L2 (Ridge)
  * Elastic Net (L1 + L2)

## Mathematical Background

### Notation
| Symbol | Description |
| ------ | ----------- |
| $m$ | Number of samples |
| $n$ | Number of features |
| $X\in\{R}^{m \times n}$ | Design matrix |
| $y\in\{R}^{m \times 1}$ | Target vector |
| $\theta\in\{R}^{n \times 1}$ | Parameter vector |
| $\hat{y}\in\{R}^{m \times 1}$ | Model predictions |
| $L(\theta)$ | Loss function (MSE) |
| $R(\theta)$ | Penalty function |
| $J(\theta)$ | Objective function |

A bias term is incorporated by prepending a column of ones to the training data

## Linear Regression Model

Linear regression models aim to fit the relationship between input features and the target:

$$
\hat{y} = X\theta
$$

## Loss Function

The model is trained by adjusting weights and biases such that they minimize the Mean Standard Error (MSE):

### Loss (MSE) Function
$$
L(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{m} \lVert y_i - X\theta \rVert_2^2
$$
### Gradient W.R.T. Parameters
$$
\nabla_{\theta}L(\theta) = \frac{1}{m} X^T(y-X\theta)
$$

## Regularization
Let:
* $\lambda$ be the regularization coefficient
* $\alpha$ be the L1 ratio

Generally, the implementation of a regularization penalty takes the form:

$$
J(\theta) = L(\theta) + \lambda R(\theta)
$$

### L1 (Lasso) Regularization: 
L1 regularization penalizes the absolute magnitude of the parameters:

#### Regularization Function
$$
R_{L1}(\theta) = \lambda \lVert\theta\rVert_1
$$
#### Gradient W.R.T. Parameters
$$
\nabla_{\theta}R_{L1}(\theta) =\lambda sign(\theta)
$$
#### Properties
* Encourages sparsity of weights
* Implicit feature selection

### L2 (Ridge) Regularization
L2 regularization penalizes the squared magnitude of the parameters:

#### Regularization Function
$$
R_{L2}(\theta) = \lambda \lVert\theta\rVert_2^2
$$
#### Gradient W.R.T. Parameters
$$
\nabla_{\theta}R_{L2}(\theta) = 2 \lambda \theta
$$
#### Properties
* Shrinks coefficients smoothly to zero
* Handles collinearity

### Elastic Net Regularization
Elastic Net regularization combines the effects of L1 and L2 regularization:

#### Regularization Function
$$
R_{EN}(\theta) = \lambda \left( \alpha \lVert\theta\rVert_1) + (1 - \alpha) \lVert\theta\rVert_2^2 \right)
$$
#### Gradient W.R.T. Parameters
$$
\nabla_{\theta}R_{EN}(\theta) = \lambda \left( \alpha sign(\theta) + (1 - \alpha) \theta \right)
$$
#### Properties
* Captures benefits of both L1 and L2
* High stability
* Moderate sparsity

## Final Objective
Combining the loss and regularization functions, we obtain the objective function which the model must minimize:

### Objective Function
$$
J(\theta) = L(\theta) + \lambda R(\theta)
$$
### Gradient of the Objective
$$
\nabla_{\theta}J(\theta) = \frac{1}{m} X^T(y-X\theta) + \lambda \left( \alpha sign(\theta) + (1 - \alpha) \theta \right)
$$

## Optimization Methods

### Normal Equation
The closed form for Ordinary Least Squares regression is:

$$   
\theta = \left( X^t X \right)^{-1} X^T y   
$$

#### Properties
* No iterations necessary
* Computationally expensive for large n, m
* Does not support learning rate scheduling or regularization

### Gradient Descent
The remaining methods are different implementations of Gradient Descent.
Let $\eta$ be the learning rate

#### Batch Gradient Descent (BGD)
BGD calculates the gradient over the entire dataset at once

##### Properties
* Slow, but stable convergence
* Inefficient computationally

#### Mini-Batch Gradient Descent (MBGD)
MBGD calculates the gradient over subsets of the dataset

##### Properties
* Stable convergence
* Moderately efficient computationally

#### Stochastic Gradient Descent (SGD)
SGD calculates the gradient over subsets of the dataset

##### Properties
* Fast, but unstable convergance
* Highly efficient computationally

### Parameter Update Rule
Each of the gradient-based methods adhere to the following update rule:

$$
\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta}J(\theta)
$$

On each iteration, the paramaters are adjusted in the direction that most directly minimizes the objective function

## Learning Rate Scheduling
### Constant Learning Rate
In this schedule, the learning rate remains constant. 
#### Properties
* Simple and easily implemented
* Good baseline for experimentation
* Pros:
  * Minimal computation overhead
  * Works well for convex problems
* Cons:
  * Can lead to divergence or oscillation if too big
  * Can lead to slow convergence is too small

### Inverse Time Decay Scheduling
In this schedule, the learning rate diminishes as training progresses:

$$
\eta_t = \frac{\eta_0}{1+ d t}
$$

Where:
* $\eta_0$ is the initial learning rate
* $t$ is the index of epoch
* $d$ is the learning rate decay

#### Properties
* Smooth monotonic decay
  * Large steps initially to more quickly reach convergence
  * Smaller steps over time to reduce likelihood of oscillation
* Pros:
  * Improves stability near convergence
  * Reduces oscillations around minima
* Cons:
  * Requires tuning multiple hyperparameters
  * Can decay too quickly if misconfigured
 
