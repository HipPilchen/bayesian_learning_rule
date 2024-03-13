import numpy as np
from functions import function, Activated_function
import matplotlib.pyplot as plt

from utils import create_gif_function_eval_pillow_with_base

# TODO : ADAPT THE CODE FOR FUNCTIONS OF R^p with p accissible with self.function.dom_size
# This code should only work with convex functions
# https://arxiv.org/pdf/2002.10060.pdf

class NewtonsMethod:
    def __init__(self, m = None, s = None, nb_MC_eval = 1, learning_rate = 0.001, method = "bayes"):
        self.m = m 
        self.s = s
        self.method = method
        self.learning_rate = learning_rate
        self.nb_MC_eval = nb_MC_eval
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
        
    def step_deterministic(self):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            grad = self.function.gradient(self.m)
            hessian = self.function.hessian(self.m)
            self.m = self.m - grad / hessian
            self.m_values.append(self.m)
            print("grad", grad, "hessian", hessian, "m", self.m, "quotient", grad / hessian, "f(m)", self.function.f(self.m))
            self.f_m_values.append(self.function.f(self.m))
        
    def check_function(self):
        if self.function.hessian is None:
            raise ValueError("The function must have a hessian")

    def optimize(self, function, n_iter = 100, clip_sampling = 1.):
        self.clip_sampling = clip_sampling
        self.function = function
        self.dom_size = function.dom_size
        self.check_function()
        self.init_parameters()
        for _ in range(n_iter):
            self.step()
        return self.m, self.function.f(self.m)

    def step(self):
        if self.method == "bayes":
            if self.dom_size > 1:
                samples = np.random.multivariate_normal(self.m, 1., self.nb_MC_eval)
                raise NotImplementedError # TODO
            else:
                samples = np.random.normal(self.m, min(1/self.s, self.clip_sampling), self.nb_MC_eval)
                self.update_cov(samples)  # We update the covariance matrix before the mean
                self.update_mean(samples)
                self.f_m_values.append(self.function.f(self.m))
        else:
            self.step_deterministic()

    def update_mean(self, samples):
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            grad_appro = self.approximate_gradient(samples)
            self.m = self.m - self.learning_rate * grad_appro / self.s
            print("m", self.m, "grad_appro", grad_appro, "1/s", 1/self.s, "samples", samples)
            self.m_values.append(self.m)

    def update_cov(self, samples):   
        if self.dom_size > 1:
            raise NotImplementedError
        else:
            hessian_appro = self.approximate_hessian(samples)
            G_pd = self.s - hessian_appro
            pd_term = self.learning_rate**2 / 2 * G_pd * 1/self.s * G_pd
            # pd_term = 0
            self.s = (1 - self.learning_rate) * self.s + self.learning_rate * hessian_appro + pd_term
            self.s_values.append(self.s)
            # print("s", self.s, "last approx hessian", hessian_appro,  "pd_term", pd_term, "m :" , self.m, "G_pd", G_pd)
            
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
    f = lambda x: x**2
    gradient = lambda x: 2*x
    hessian = lambda x: 2
    optimal_value = 0
    dom_f = 1
    
    # f = lambda x: 1/2 * (np.sin(13*x) * np.sin(27*x) + 1)
    # gradient = lambda x: ( 13 * np.cos(13*x) * np.sin(27*x) + 27 * np.sin(13*x) * np.cos(27*x) )/2
    # hessian = lambda x: ( 13 * 27 * np.cos(13*x) * np.cos(27*x) - 13*13*np.sin(13*x) * np.cos(27*x) + 27 * 13 * np.cos(13*x) * np.cos(27*x) - 13 * 13 * np.sin(13*x) * np.sin(27*x)  ) / 2
    # optimal_value = 0.03
    # dom_f = 1
    
    # f = lambda x: np.sin(x) 
    # gradient = lambda x: np.cos(x)
    # hessian = lambda x: -np.sin(x)
    # optimal_value = -1
    # dom_f = 1
    
    # f = lambda x: x**4 - 4*x**2
    # gradient= lambda x: 4*x**3 - 8*x
    # hessian = lambda x: 12*x**2 - 8
    # optimal_value = -4
    # dom_f = 1

    
    # X_four = function(f, gradient, dom_f, hessian=hessian, optimal_value=optimal_value)
    # X_square.plot(-2, 2)
    # X_square.plot_gradient(-5, 5)
    # X_square.plot_hessian(-5, 5)
    
    f = lambda x: x**4
    gradient = lambda x: 4*x**3
    hessian = lambda x: 12*x**2
    optimal_value = 0
    dom_f = 1

    f_x = function(f, gradient, dom_f, hessian=hessian, optimal_value=optimal_value)

    final_f = Activated_function("sigmoid", f_x)
    # final_f.plot(-8, 8)
    # final_f.plot_gradient(-8, 8)
    # final_f.plot_hessian(-8, 8)
    newtons = NewtonsMethod(learning_rate=1., nb_MC_eval=100, m=0.5, method="bayes")
    m, fm = newtons.optimize(final_f,150,  clip_sampling = 0.1)
    print(np.min(newtons.m_values), np.max(newtons.m_values))
    # print(newtons.m_values)
    newtons.plot_residuals()
    # create_gif_function_eval_pillow_with_base(newtons.m_values, final_f.f, step_size=10, filename="newtons_method_PD.gif")
    print("The optimisiation leeds to m = {m} and f(m) = {fm}".format(m=m, fm=fm))
    