import numpy as np

def gaussian_density_1D(x, mu, sigma_2):
    """
    Evaluate the density of a Gaussian distribution at a point x.
    
    Parameters:
    - x: The point at which to evaluate the density
    - mu: The mean of the Gaussian distribution
    - sigma: The standard deviation of the Gaussian distribution
    
    Returns:
    - The density of the Gaussian distribution at point x
    """
    return (1 / (  np.sqrt(2 * np.pi * sigma_2))) * np.exp(-0.5 * ((x - mu)**2 / sigma_2) )


def combined_function(theta):
    sharp = np.exp(-30 * (theta + 1)**2)
    
    flat = 0.3 * (1 + np.cos(np.pi * (theta -0.8) / 2.5 ))
    
    combined = sharp + flat
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min())
    
    return -combined_norm
