import numpy as np 
from functions import function
import matplotlib.pyplot as plt


class Loss:
    
    def __init__(self, loss, gradient, hessian = None):
        self.loss = loss
        self.gradient_f = gradient # gradient on the first coordinate (ftheta(x))
        self.hessian = hessian
    def loss(self, fx, y):
        return self.loss(fx, y)

    def gradient(self, fx, y):
        return self.gradient_f(fx, y)

    def hessian(self, fx, y):
        return self.hessian(fx, y)


class Predictor:
    def __init__(self, f, gradient, dim_theta, hessian = None):
        self.f = f
        self.dim_theta = dim_theta
        self.gradient_theta = gradient # gradient on theta
        self.hessian = hessian
    def f(self, theta, x):
        return self.f(theta, x)

    def gradient(self, theta, x):
        return self.gradient_theta(theta, x)
    def hessian(self, theta, x):
        return self.hessian(theta,x)

class SAG:
    
    def __init__(self, X, y, f, loss, batch_size = 2, m = None, learning_rate = 0.001, sigma = None):
        
        assert (X.shape[0]> batch_size)
  
        self.theta = m
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dim_params = f.dim_theta
        self.grad = np.zeros((X.shape[0],self.dim_params))
        self.theta_values = []
        self.loss_values = []
        self.predictor = f
        self.loss = loss
        self.S = sigma
        self.X = X # N x p 
        self.y = y # N x 1
        
    def init_parameters(self):
        if self.theta is None:
            self.theta = np.random.randn(self.dim_params)
        if self.S is None:
            self.S = np.eye(self.dim_params)
        else:
            assert(self.S.shape[0]==self.dim_params)
            print('Verify that Sigma is invertible !')
        self.theta_values.append(self.theta)
        self.loss_values.append(self.loss.loss(self.predictor.f(self.theta, self.X), self.y))
        
    def optimize(self, n_iter = 100):
        self.init_parameters()
        n = self.X.shape[0]
        for _ in range(n_iter):
            update_indices = np.random.choice(n, self.batch_size, replace=False)
            self.step(update_indices)
            self.theta_values.append(self.theta)
       
            self.loss_values.append(self.loss.loss(self.predictor.f(self.theta, self.X), self.y))
        return self.theta
    
    def step(self, update_indices):
        for i in update_indices:
            # Replace the gradient by the averaged stochastic approximation
            self.grad[i,:] = self.loss.gradient(self.predictor.f(self.theta, self.X[i,:]), self.y[i])*self.predictor.gradient(self.theta, self.X[i,:])
        self.theta -= self.learning_rate*np.linalg.inv(self.S)@np.sum(self.grad, axis = 0)
        
    def plot(self):
        plt.plot(self.loss_values, '-')
        plt.title('Evolution of the loss during learning with BLR SAG')
        plt.show()
        
    def samples_param(self, n_samples):
        return [np.random.multivariate_normal(mean = self.theta_values[-1], cov = self.S) for _ in range(n_samples)] # n_samples * dim_params
    

    
def viz_svm(theta, X, y):
    plt.figure()
    if isinstance(theta,list):
        
        for param in theta:
            a = -param[0] / param[1]
            xx = np.linspace(-5, 5)
            yy = a * xx - (param[2]) /param[1]
            plt.plot(xx,yy, alpha = 0.2)
            
        mean_theta = np.mean(np.array(theta), axis = 0)
        a = -mean_theta[0] / mean_theta[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (mean_theta[2]) / mean_theta[1]
        plt.plot(xx,yy, label = 'SVM Hyperplane')
        
    else:
        a = -theta[0] / theta[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (theta[2]) / theta[1]
        plt.plot(xx,yy, label = 'SVM Hyperplane')

    plt.scatter([x[0] for x in X], [x[1] for x in X], c = y)
    plt.title('SVM')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
def smoothed_hinge_loss(pred, y):
    if isinstance(y,np.int64):
        prod = pred*y
    else :
        prod = pred@y
    if prod<0:
        return 1/2 - prod
    elif prod<1:
        return (1-prod)**2/2
    else:
        return 0
    
def grad_smoothed_hinge_loss(pred, y):
    if isinstance(y,np.int64):
        prod = pred*y
    else :
        prod = pred@y
    if prod<0:
        return y
    elif prod<1:
        return (1-prod)*y
    else:
        return 0
    
def f_pred(theta, x):
    return np.tanh(x@theta[:-1] - theta[-1])

def grad_f_pred(theta, x):
    last_comp = np.zeros(theta.shape[0])
    last_comp[-1] = 1
    big_x = np.concatenate((x,np.zeros(1)))
    return (1-np.tanh(x@theta[:-1] - theta[-1])**2)*big_x - last_comp * (1-np.tanh(x.T@theta[:-1] - theta[-1])**2)