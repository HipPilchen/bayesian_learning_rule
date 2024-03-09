import numpy as np
from functions import function
import matplotlib.pyplot as plt


import math
from utils import gaussian_density_1D

# TODO : ADAPT THE CODE FOR FUNCTIONS OF R^p with p accissible with self.function.dom_size

class GMM_Descent:
    def __init__(self, number_of_mixtures = 2, m_k = None, S_k = None, pi_k = None, batch_size = 1, learning_rate = 0.001):
        self.m_k = m_k
        self.S_k = S_k
        self.pi_k = pi_k
        self.number_of_mixtures = number_of_mixtures
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.f_k_values = []
        self.m_values = []
        self.S_values = []

    def init_parameters(self):
        if self.dom_size > 1:
            raise NotImplementedError # TODO
        if self.m_k is None:
            self.m_k = np.random.randn(self.number_of_mixtures)
        if self.S_k is None:
            self.S_k = np.ones(self.number_of_mixtures) 
        if self.pi_k is None:
            self.pi_k = np.ones(self.number_of_mixtures) / self.number_of_mixtures
        
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
        self.f_k_values = np.array(self.f_k_values)
        self.m_values = np.array(self.m_values)
        self.S_values = np.array(self.S_values)
        return 
        # return self.m, self.function.f(self.m)
        
    def update_mean_k(self, samples, k):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            grad_density = self.appro_grad_log_gmm(samples)
            appro_gradient = self.approximate_gradient(samples)
            
            self.m_k[k] = self.m_k[k] - self.learning_rate * ( appro_gradient + grad_density ) / self.S_k[k]
        
    def update_cov_k(self, samples, k):  
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            hess_density = self.appro_hessian_log_gmm(samples)
            hessian_appro = self.approximate_hessian(samples)
            hess_density  = 0
            self.S_k[k] = max( (1 - self.learning_rate) * self.S_k[k] + self.learning_rate * ( hessian_appro + hess_density ), 1e-2) # NOT THE SAME EQU I NEED TO CHANGE + WHAT HAPPENS IF THE HESSIAN IS NEGATIVE
            

    def step_k(self, k):
        if self.dom_size > 1:
            sample = np.random.multivariate_normal(self.m_k[k], self.S_k[k], self.batch_size)
            raise NotImplementedError # TODO
        else:
            print("s_k", self.S_k[k])
            samples = np.random.normal(self.m_k[k], self.S_k[k], self.batch_size)
            self.update_cov_k(samples, k)  # We update the covariance matrix before the mean
            self.update_mean_k(samples, k)
            self.f_k_values[-1][k] = self.function.f(self.m_k[k])
            self.m_values[-1][k] = self.m_k[k]
            self.S_values[-1][k] = self.S_k[k]
            
    def appro_grad_log_gmm(self, samples):
        all_grads = np.array([self.grad_log_gmm(sample) for sample in samples])
        return all_grads.mean(axis=0)
    
    def appro_hessian_log_gmm(self, samples):
        all_grads = np.array([self.hessian_log_gmm(sample) for sample in samples])
        return all_grads.mean(axis=0)
        
    def hessian_log_gmm(self, theta):
        q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) for k in range(self.number_of_mixtures)])
        grad_q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) * ( - theta + self.m_k[k]) / self.S_k[k] for k in range(self.number_of_mixtures)])
        hessian_q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) * ( - 1 / self.S_k[k] + (theta - self.m_k[k])**2 / self.S_k[k]**2) for k in range(self.number_of_mixtures)])
        return hessian_q_theta/q_theta - (grad_q_theta/q_theta)**2
        
    def grad_log_gmm(self, theta):
        q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) for k in range(self.number_of_mixtures)])
        grad_q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) * ( - theta + self.m_k[k]) / self.S_k[k] for k in range(self.number_of_mixtures)])
        return grad_q_theta/q_theta
            
    def step(self):
        self.f_k_values.append( np.zeros(self.number_of_mixtures) )
        self.m_values.append( np.zeros(self.number_of_mixtures) )
        self.S_values.append( np.zeros(self.number_of_mixtures) )   
        for k in range(self.number_of_mixtures):
            self.step_k(k)
        # print("After step k m_1 = ", self.m_k[0], "and m_2 = ", self.m_k[1])
        # print("And the values of f_k are ", self.f_k_values[-1])
        
            
    def approximate_hessian(self, samples):
        all_hessians = np.array([self.function.hessian(sample) for sample in samples])
        return all_hessians.mean(axis=0)
        
    def approximate_gradient(self, samples):
        all_grads = np.array([self.function.gradient(sample) for sample in samples])
        return all_grads.mean(axis=0)
        return np.mean(self.function.gradient(samples)) # TODO : check if it is the right way to compute the gradient of the density
        
    def plot_residuals(self, k=0):
        if self.function.optimal_value is None:
            raise ValueError("The function must have an optimal value")
        residuals = self.f_k_values[:,k] - self.function.optimal_value
        residuals = np.abs(residuals)
        plt.plot(residuals, '-', label='|f(m) - f(m*)| for k = {k}'.format(k=k))
        plt.show()
        
    def plot_explo(self, k=0):
        x = np.linspace(-5, 5, 100)
        y = [self.function.f(x_) for x_ in x]
        plt.plot(x, y, '-')
        plt.plot(self.m_values[:,k], self.f_k_values[:,k], '-', label='f(m)')
        plt.show()
    
    
if __name__ == "__main__":
    f = lambda x: np.sin(x) 
    gradient = lambda x: np.cos(x)
    hessian = lambda x: -np.sin(x)
    optimal_value = -1.
    dom_f = 1
    # f = lambda x: 1/2 * (np.sin(13*x) * np.sin(27*x) + 1)
    # gradient = lambda x: ( 13 * np.cos(13*x) * np.sin(27*x) + 27 * np.sin(13*x) * np.cos(27*x) )/2
    # hessian = lambda x: ( 13 * 27 * np.cos(13*x) * np.cos(27*x) - 13*13*np.sin(13*x) * np.cos(27*x) + 27 * 13 * np.cos(13*x) * np.cos(27*x) - 13 * 13 * np.sin(13*x) * np.sin(27*x)  ) / 2
    # dom_f = 1
    # optimal_value = 0.003
    sin = function(f, gradient, dom_f, hessian=hessian, optimal_value=optimal_value)
    
    gamm_des = GMM_Descent(learning_rate=0.001, batch_size=10, number_of_mixtures = 2, m_k=np.array([0.2, 1.8]), S_k=np.array([0.5, 0.5]) )
    gamm_des.optimize(sin, 10000)
    print(gamm_des.m_values[:, 0])
    print(gamm_des.f_k_values[-1])
    gamm_des.plot_explo(0)
    gamm_des.plot_explo(1)
    