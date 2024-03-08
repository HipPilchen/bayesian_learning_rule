#the idea is to compare the smoothing effect of SGD to the one of the Bayesian Learning Rule
# we want to compare the similar effects of injection of noise in the optimization process

import numpy as np
import matplotlib.pyplot as plt
import torch

def modified_logistic_loss(theta, min_theta=2):
    """
    Modified logistic loss function that tends towards +infinity as theta approaches -infinity,
    descends to a global minimum, and then slightly rises to a plateau.
    
    Parameters:
    - theta: A real-valued parameter.
    
    Returns:
    - Modified logistic loss value for the given theta.
    """
    # Classic logistic loss component
    logistic_part = np.log(1 + np.exp(-theta))
    
    # Modifier component to slightly rise after a minimum and approach a plateau
    modifier_part = 0.1 * np.tanh(0.5 * (theta - min_theta))
    
    # Combine the components
    loss = logistic_part + modifier_part
    
    return loss

def f(theta):
    return (theta-2)**2

def gradient_f(theta):
    return 2*(theta-2) 

def gradient_logistic_loss(batch, min_theta=2):
    """
    Gradient of the modified logistic loss function.
    
    Parameters:
    - theta: A real-valued parameter.
    
    Returns:
    - Gradient of the modified logistic loss function at the given theta.
    """
    logistic_gradient = -(np.exp(-batch)) / (1 + np.exp(-batch))
    
    modifier_gradient = 0.1 * 0.5 * (1 - np.tanh(0.5 * (batch - min_theta))**2)
    
    gradient = logistic_gradient + modifier_gradient
    
    return gradient

class GD:
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def descent(loss_function, gradient_f, theta = None, step_size = 0.01, n_iter = 1000):
        """
        Gradient Descent algorithm.
        
        Parameters:
        - loss_function: The loss function to be minimized.
        - theta: The initial parameter value.
        - step_size: The learning rate.
        - n_iter: The number of iterations.
        - batch_size: The number of samples in each batch.
        
        Returns:
        - theta_values: The sequence of parameter values.
        - loss_values: The sequence of loss values.
        """
        theta_values = []
        loss_values = []
        
        if theta is None:
            theta = np.random.randn()
        
        for _ in range(n_iter):
            # Randomly sample a batch of data
            
            # Compute the loss and gradient
            loss = loss_function(theta)
            gradient = gradient_f(theta)
            
            # Update the parameter
            theta = theta - step_size * gradient
            
            # Store the parameter and loss values
            theta_values.append(theta)
            loss_values.append(loss)
        
        return theta_values, loss_values


class BLR_descent:
    def __init__(self, sigma, n_samples) -> None:
        self.sigma = sigma
        self.n_samples = n_samples

    def descent(self, loss_function, n_iter = 1000, step_size = 0.01, m = None):
        if m is None:
            m = np.random.randn()
        print('Initial m:', m)
        m_values = []
        for _ in range(n_iter):
            thetas = np.random.normal(m, self.sigma, self.n_samples)
            loss_values = loss_function(thetas)
            expected_loss1 = np.mean(loss_values) * m/(self.sigma**2)
            expected_loss2 = np.mean(thetas * loss_values) * 1/(self.sigma**2)
            m = m - step_size * (-expected_loss1 + expected_loss2)
            m_values.append(m)
        return m_values
    
