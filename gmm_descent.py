import numpy as np
from functions import function
import matplotlib.pyplot as plt

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
        self.f_k_values = [np.zeros(number_of_mixtures)]
        self.all_f_k_values = []

    def init_parameters(self):
        if self.dom_size > 1:
            raise NotImplementedError # TODO
        if self.m_k is None:
            self.m_k = np.random.randn(self.number_of_mixtures)
        if self.S_k is None:
            self.S_k = np.ones(self.number_of_mixtures) 
        if self.pi_k is None:
            self.pi_k = np.ones(self.number_of_mixtures) / self.number_of_mixtures
        print("S-k", self.S_k)
        # self.m_values.append(self.m)
        # self.s_values.append(self.s)
        # self.f_m_values.append(self.function.f(self.m))
        
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
            print("before", self.S_k[k])
            print("hess_density", hess_density)
            print("hessian_appro", hessian_appro)
            self.S_k[k] = (1 - self.learning_rate) * self.S_k[k] + self.learning_rate * ( hessian_appro + hess_density ) # NOT THE SAME EQU I NEED TO CHANGE + WHAT HAPPENS IF THE HESSIAN IS NEGATIVE
            print(self.S_k[k])
        
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
            self.all_f_k_values.append(self.function.f(self.m_k[k]))
            
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
        print("m and S :",self.m_k[0], self.S_k[0])
        q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) for k in range(self.number_of_mixtures)])
        grad_q_theta = np.sum([self.pi_k[k] * gaussian_density_1D(theta, self.m_k[k], self.S_k[k]) * ( - theta + self.m_k[k]) / self.S_k[k] for k in range(self.number_of_mixtures)])
        # print(grad_q_theta)
        # print(grad_q_theta/q_theta)
        return grad_q_theta/q_theta
            
    def step(self):
        for k in range(self.number_of_mixtures):
            self.step_k(k)
        self.f_k_values.append( np.zeros(self.number_of_mixtures) )
            
    def approximate_hessian(self, samples):
        all_hessians = np.array([self.function.hessian(sample) for sample in samples])
        return all_hessians.mean(axis=0)
        
    def approximate_gradient(self, samples):
        all_grads = np.array([self.function.gradient(sample) for sample in samples])
        return all_grads.mean(axis=0)
        return np.mean(self.function.gradient(samples)) # TODO : check if it is the right way to compute the gradient of the density
        
    def plot_residuals(self, optimal_value):
        residuals = np.array(self.f_m_values) - optimal_value
        residuals = np.abs(residuals)
        plt.plot(residuals, '-', label='|f(m) - f(m*)|')
        plt.show()
    
    
if __name__ == "__main__":
    f = lambda x: -x**4 + 4*x**2
    gradient= lambda x: -4*x**3 + 8*x
    hessian = lambda x: -12*x**2 + 8
    dom_f = 1
    X_square = function(f, gradient, dom_f, hessian=hessian)
    gamm_des = GMM_Descent(learning_rate=0.001, batch_size=10, number_of_mixtures = 2, m_k=np.array([-2, 2]), S_k=np.array([1, 1]) )
    gamm_des.optimize(X_square, 10000)
    # print("The optimisiation leeds to m = {m} and f(m) = {fm}".format(m=m, fm=fm))
    gamm_des.plot_residuals(0)
    