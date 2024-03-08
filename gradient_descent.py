import numpy as np 
from functions import function
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, m = None, learning_rate = 0.001):
        self.m = m
        self.learning_rate = learning_rate
        self.m_values = []
        self.f_m_values = []
        
    def init_parameters(self):
        if self.function.dom_size > 1:
            raise NotImplementedError # TODO
        if self.m is None:
            self.m = np.random.randn(1)
        self.m_values.append(self.m)
        self.f_m_values.append(self.function.f(self.m))
        
    def optimize(self, function, n_iter = 100):
        self.function = function
        self.dom_size = function.dom_size
        self.init_parameters()
        for _ in range(n_iter):
            self.step()
        return self.m, self.function.f(self.m)
    
    def step(self):
        if self.dom_size > 1:
            echantillon = np.random.multivariate_normal(self.m, 1., 1)
        else:
            theta = np.random.normal(self.m, 1, 1)
            m = self.m - self.learning_rate * self.function.gradient(theta)
            self.m_values.append(self.m)
            self.m = m
            self.f_m_values.append(self.function.f(self.m))
        
    def plot(self):
        plt.plot(self.m_values, self.f_m_values, '-', label='f(m)')
        plt.show()
        
    def plot_residuals(self, optimal_value):
        residuals = np.array(self.f_m_values) - optimal_value
        residuals = np.abs(residuals)
        plt.plot(residuals, '-', label='|f(m) - f(m*)|')
        plt.show()
    
    
if __name__ == "__main__":
    f = lambda x: x**2
    gradient = lambda x: 2*x
    dom_f = 1
    X_square = function(f, gradient, dom_f)
    m = np.random.randn(1)
    gd = GradientDescent(learning_rate=0.01)
    m, fm = gd.optimize(X_square, 10000)
    X_square.plot(-10, 10)
    print("The optimisiation leeds to m = {m} and f(m) = {fm}".format(m=m, fm=fm))
    gd.plot_residuals(0)
    