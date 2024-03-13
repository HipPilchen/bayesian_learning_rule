# Goal: To show that the Bayesian Learning Rule tends to look for the flatter regions of the loss landscape.
# This is similar to SGD whose noise can be seen as a way to escape the local minima.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def loss_landscape(mu, sigma, loss_function, n_samples=10000):

    """
    Args:
        mu: mean of the gaussian
        sigma: standard deviation of the gaussian
        loss_function: the loss function to be used
    Returns:
        loss_landscape: the expected loss landscape
    """
    thetas = np.random.normal(mu, sigma, n_samples)
    
    loss_values = np.array([loss_function(theta) for theta in thetas])
    
    expected_loss = np.mean(loss_values)
    
    return expected_loss

def plot_loss_landscape_heatmap(mu_range, sigma_range, loss_function, n_samples=1000):
    mu_values = np.arange(*mu_range)
    sigma_values = np.arange(*sigma_range)
    expected_loss_values = np.zeros((len(sigma_values), len(mu_values)))
    
    for i, sigma in tqdm(enumerate(sigma_values), total=len(sigma_values)):
        for j, mu in enumerate(mu_values):
            expected_loss = loss_landscape(mu, sigma, loss_function, n_samples)
            expected_loss_values[i, j] = np.round(expected_loss / 0.05) * 0.05
    
    plt.figure(figsize=(10, 8))
    plt.imshow(expected_loss_values, interpolation='nearest', cmap='viridis', aspect='auto',
               extent=[mu_values.min(), mu_values.max(), sigma_values.max(), sigma_values.min()])
    plt.contour(mu_values, sigma_values, expected_loss_values, levels=np.unique(expected_loss_values), colors='k', linewidths=0.5)
    plt.colorbar(label='Expected Loss')
    plt.xlabel('Mu')
    plt.ylabel('Sigma')
    plt.title('Heatmap of Expected Loss with Discretized Color Zones')
    plt.gca().invert_yaxis()
    plt.show()

