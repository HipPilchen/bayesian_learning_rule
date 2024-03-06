import numpy as np 
from functions import function
import matplotlib.pyplot as plt



class MC_Dropout:
    def __init__(self, loss, theta = None, learning_rate = 0.001, pi = 0.5, n_layer = 3, n_units = 4):
        
        self.learning_rate = learning_rate    
        
        # Row maps unit and column maps the layer
        if theta is None:
            self.n_layer = n_layer
            self.n_units = n_units
            self.theta = np.random.randn(self.n_units, self.n_layer, self.n_units)
        else:
            self.n_layer = theta.shape[1]
            self.n_units = theta.shape[0]
            self.dim_params = theta.shape[2]
            self.theta = theta
        
        self.loss = loss
        self.zs = np.random.binomial(1, pi, (self.n_units, self.n_layer))
        S = np.eye(self.n_units) 
        self.covs = np.array([[S for _ in range(self.n_layer)] for _ in range(self.n_units)]) 
        self.pi = pi
        self.f_m_values = []
            
    def step(self):
        theta_tilde = np.multiply(self.theta, np.tile(self.zs.T,reps=(self.n_units,1,1)))
        grad = self.loss.gradient(theta_tilde)
        for i in range(self.n_units):
            for j in range(self.n_layer):
                self.covs[i,j,:,:] = ((1-self.learning_rate)*self.covs[i,j,:,:] +
                                      self.learning_rate*np.outer(self.loss.gradient(theta_tilde)[i,j,:], self.loss.gradient(theta_tilde)[i,j,:])/self.pi)       
                self.theta[i, j,:] = self.theta[i, j,:] - self.learning_rate * np.linalg.pinv(self.covs[i,j,:,:])@self.loss.gradient(theta_tilde)[i,j,:]/self.pi
    
    def optimize(self, n_iter = 100):
        for _ in range(n_iter):
            self.step()
            self.f_m_values.append(np.mean(self.loss.f(self.theta)))
            if _ % 1000 == 0:
                print('Iteration :', _, 'Loss :', self.f_m_values[-1])
        return self.theta, self.covs
        
    def plot_residuals(self, optimal_value):
        residuals =  np.array(self.f_m_values) - optimal_value
        residuals = np.abs(residuals)
        plt.plot(residuals, '-', label='|f(m) - f(m*)|')
        plt.show()
    
    
if __name__ == "__main__":
    f = lambda x: x**2
    gradient = lambda x: 2*x
    dom_f = 1
    X_square = function(f, gradient, dom_f)
    MC_dropout = MC_Dropout(X_square)
    thetas, covs = MC_dropout.optimize(20000)
    MC_dropout.plot_residuals(0)
                