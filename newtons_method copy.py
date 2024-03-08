import numpy as np
from functions import function
import matplotlib.pyplot as plt


# TODO : ADAPT THE CODE FOR FUNCTIONS OF R^p with p accissible with self.function.dom_size

class NewtonsMethod:
    def __init__(self, m = None, s = None, batch_size = 1, learning_rate = 0.001):
        self.m = m 
        self.s = s
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.m_values   = []
        self.f_m_values = []
        self.s_values   = []
        
    def init_parameters(self):
        if self.dom_size > 1:
            raise NotImplementedError # TODO
        if self.m is None:
            self.m = np.random.randn(1)
        if self.s is None:
            self.s = 1
        self.m_values.append(self.m)
        self.s_values.append(self.s)
        self.f_m_values.append(self.function.f(self.m))
        
    def check_function(self):
        if self.function.hessian is None:
            raise ValueError("The function must have a hessian")

    def optimize(self, function, n_iter = 100):
        self.function = function
        self.dom_size = function.dom_size
        self.check_function()
        self.init_parameters()
        for _ in range(n_iter):
            self.step()
        return self.m, self.function.f(self.m)

    def step(self):
        if self.dom_size > 1:
            samples = np.random.multivariate_normal(self.m, 1., self.batch_size)
            raise NotImplementedError # TODO
        else:
            samples = np.random.normal(self.m, 1/self.s, self.batch_size)
            self.update_cov(samples)  # We update the covariance matrix before the mean
            self.update_mean(samples)
            self.f_m_values.append(self.function.f(self.m))

    def update_mean(self, samples):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            grad_appro = self.approximate_gradient(samples)
            self.m = self.m - self.learning_rate * grad_appro / self.s
            self.m_values.append(self.m)
            print("new m : ", self.m)

    def update_cov(self, samples):   
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            hessian_appro = self.approximate_hessian(samples)
            self.s = max( (1 - self.learning_rate) * self.s + self.learning_rate * hessian_appro, 1e-2)
            self.s_values.append(self.s)
            print("new s : ", self.s)
            
    def approximate_hessian(self, samples):
        all_hessians = np.array([self.function.hessian(sample) for sample in samples])
        return all_hessians.mean(axis=0)
        
    def approximate_gradient(self, samples):
        all_grads = np.array([self.function.gradient(sample) for sample in samples])
        return all_grads.mean(axis=0)
        
    def plot(self):
        plt.plot(self.m_values, self.f_m_values, '-', label='f(m)')
        plt.show()
        
    def plot_residuals(self):
        if self.function.optimal_value is None:
            raise ValueError("The function must have an optimal value")
        residuals = np.array(self.f_m_values) - self.function.optimal_value
        residuals = np.abs(residuals)
        plt.plot(residuals, '-', label='|f(m) - f(m*)|')
        plt.show()
    
    
if __name__ == "__main__":
    # f = lambda x: x**2
    # gradient = lambda x: 2*x
    # hessian = lambda x: 2
    # f = lambda x: x**4 - 4*x**2
    # gradient= lambda x: 4*x**3 - 8*x
    # hessian = lambda x: 12*x**2 - 8
    # optimal_value = 
    # dom_f = 1
    
    # f = lambda x: 1/2 * (np.sin(13*x) * np.sin(27*x) + 1)
    # gradient = lambda x: ( 13 * np.cos(13*x) * np.sin(27*x) + 27 * np.sin(13*x) * np.cos(27*x) )/2
    # hessian = lambda x: ( 13 * 27 * np.cos(13*x) * np.cos(27*x) - 13*13*np.sin(13*x) * np.cos(27*x) + 27 * 13 * np.cos(13*x) * np.cos(27*x) - 13 * 13 * np.sin(13*x) * np.sin(27*x)  ) / 2
    # optimal_value = 0.03
    # dom_f = 1
    
    f = lambda x: np.sin(x) 
    gradient = lambda x: np.cos(x)
    hessian = lambda x: -np.sin(x)
    optimal_value = -1
    dom_f = 1

    X_square = function(f, gradient, dom_f, hessian=hessian, optimal_value=optimal_value)
    X_square.plot(-5, 5)
    # X_square.plot_gradient(-5, 5)
    # X_square.plot_hessian(-5, 5)
    newtons = NewtonsMethod(learning_rate=1.0, batch_size=20, m=1)
    m, fm = newtons.optimize(X_square, 1000)
    print("The optimisiation leeds to m = {m} and f(m) = {fm}".format(m=m, fm=fm))
    newtons.plot_residuals()
    