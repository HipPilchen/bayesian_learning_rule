import numpy as np
import matplotlib.pyplot as plt

class function:
    def __init__(self, f, gradient, dom_size, hessian = None, optimal_value = None):
        self.f = f
        self.gradient = gradient
        self.dom_size = dom_size # If f(x) is defined on R^p, dom_size = p
        self.hessian = hessian
        self.optimal_value = optimal_value
    
    def f(self, x):
        return self.f(x)
    
    def gradient(self, x):
        return self.gradient(x)
    
    def hessian(self, x):
        return self.hessian(x)
    
    def plot(self, a, b , n_points = 100):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            x = np.linspace(a, b, n_points)
            y = [self.f(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
            
    def plot_gradient(self, a, b):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            x = np.linspace(a, b, 100)
            y = [self.gradient(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
            
    def plot_hessian(self, a, b, n_points = 100):
        if self.dom_size > 1:
            raise NotImplementedError
        elif self.hessian is None:
            raise ValueError("The function must have a hessian")
        else:
            x = np.linspace(a, b, n_points)
            y = [self.hessian(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
        
class Activated_function:
    def __init__(self, activation, principal_function):
        self.function = principal_function
        self.dom_size = self.function.dom_size # If f(x) is defined on R^p, dom_size = p
        # self.optimal_value = optimal_value
        if activation == "sigmoid":
                sigmoid = lambda x: 1 / (1 + np.exp(-x))
                gradient = lambda x: sigmoid(x) * (1 - sigmoid(x))
                hessian = lambda x: gradient(x) * (1 - sigmoid(x)) - sigmoid(x) * gradient(x)
                self.activation = function(sigmoid, gradient, 1, hessian=hessian)
                self.optimal_value = self.activation.f( self.function.optimal_value)
    
    def f(self, x):
        return self.activation.f(self.function.f(x))
    
    def gradient(self, x):
        return self.function.gradient(x) * self.activation.gradient(self.function.f(x))
    
    def hessian(self, x):
        return self.function.hessian(x) * self.activation.gradient(self.function.f(x)) + self.function.gradient(x) **2 * self.activation.hessian(self.function.f(x))
    
    def plot(self, a, b , n_points = 100):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            x = np.linspace(a, b, n_points)
            y = [self.f(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
            
    def plot_gradient(self, a, b):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            x = np.linspace(a, b, 100)
            y = [self.gradient(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
            
    def plot_hessian(self, a, b, n_points = 100):
        if self.dom_size > 1:
            raise NotImplementedError
        elif self.hessian is None:
            raise ValueError("The function must have a hessian")
        else:
            x = np.linspace(a, b, n_points)
            y = [self.hessian(x_) for x_ in x]
            plt.plot(x, y, '-')
            plt.show()
        
def combined_function(theta):
    sharp = np.exp(-30 * (theta + 1)**2)
    
    flat = 0.3 * (1 + np.cos(np.pi * (theta -0.8) / 2.5 ))
    
    combined = sharp + flat
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min())
    
    return -combined_norm
        
if __name__ == "__main__":
    f = lambda x: -x**4 + 4*x**2
    gradient= lambda x: -4*x**3 + 8*x
    hessian = lambda x: -12*x**2 + 8
    dom_f = 1
    X_4 = function(f, gradient, dom_f, hessian=hessian)
    
    f = lambda x: x**2
    gradient = lambda x: 2*x
    hessian = lambda x: 2.
    dom_f = 1
    X_square = function(f, gradient, dom_f, hessian=hessian)
    
    f = lambda x: 1/2 * (np.sin(13*x) * np.sin(27*x) + 1)
    gradient = lambda x: ( 13 * np.cos(13*x) * np.sin(27*x) + 27 * np.sin(13*x) * np.cos(27*x) )/2
    hessian = lambda x: ( 13 * 27 * np.cos(13*x) * np.cos(27*x) - 13*13*np.sin(13*x) * np.cos(27*x) + 27 * 13 * np.cos(13*x) * np.cos(27*x) - 13 * 13 * np.sin(13*x) * np.sin(27*x)  ) / 2
    dom_f = 1
    sin_cos = function(f, gradient, dom_f, hessian=hessian)
    sin_cos.plot(0, 1, n_points = 1000)
    sin_cos.plot_gradient(0, 1)
    sin_cos.plot_hessian(0, 1)
    
    
    
