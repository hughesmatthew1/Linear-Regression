# Linear Regression From Scratch

This project implements **linear regression** with only the use of the **NumPy** library. The goal is to demonstrate a clear understanding of the mathematical underlyings of linear regression without reliance upon high-level libraries such as scikit-learn, TensorFlow, or PyTorch.

The implementation supports training (fitting) a model, and predicting.

## Mathematical Background

Linear regression models aim to fit the relationship between an input feature matrix $X \in \mathbb{R}^{nxm}$, and a target vector $Y \in \mathbb{R}^{n}$ as:

$$
\hat{Y} = Xw + b
$$

where:
* ${w} \in \mathbb{R}^{m}$ is the weight vector
* ${b} \in \mathbb{R}$ is the bias




The objective is to produce weights and biases which minimize the **Mean Squared Error (MSE)** loss.

$$
\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=i}^{n} \left(X_{i} w + b - y_{i} \right)^2
$$
