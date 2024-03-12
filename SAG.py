import numpy as np 
from functions import function
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from sklearn import datasets
from sklearn import metrics


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
    
    def __init__(self, X, y, f, loss, batch_size = 2, theta = None, learning_rate = 0.001, sigma = None):
        
        assert (X.shape[0]>=batch_size)
  
        self.theta = theta
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
            self.theta = np.ones(self.dim_params)
        if self.S is None:
            self.S = np.eye(self.dim_params)
        else:
            assert(self.S.shape[0]==self.dim_params)
            print('Verify that Sigma is invertible !')
        self.theta_values.append(self.theta)
        self.loss_values.append(self.loss.loss(self.predictor.f(self.theta, self.X)@self.y))
        
    def optimize(self, n_iter = 100, infotime = 1000, sgd = False, gd = False, sampled = False):
        self.init_parameters()
        n = self.X.shape[0]
        for _ in range(n_iter):
            if gd:
                update_indices = np.arange(n)
            else:
                update_indices = np.random.choice(n, self.batch_size, replace=False)
            self.step(update_indices, sgd)
            self.theta_values.append(self.theta)
            self.loss_values.append(self.loss.loss(self.predictor.f(self.theta, self.X)@self.y))
            if (_+1)%infotime==0:
                print('Iteration',_+1,'Loss',self.loss_values[-1])
        return self.theta
    
    def step(self, update_indices, sgd = False, sampled = False):
        if sgd:
            self.grad = np.zeros((self.X.shape[0],self.dim_params))
        for i in update_indices:
            # Replace the gradient by the averaged stochastic approximation
            self.grad[i,:] = self.loss.gradient(self.predictor.f(self.theta, self.X[i,:]), self.y[i])*self.predictor.gradient(self.theta, self.X[i,:])
        # print('Grad',self.grad)
        self.theta -= self.learning_rate*self.S@np.mean(self.grad, axis = 0)
        if sampled:
            self.theta = np.random.multivariate_normal(mean = self.theta, cov = self.S)
        
        
    def plot(self):
        plt.plot(np.arange(len(self.loss_values)),self.loss_values)
        plt.title('Evolution of the loss during learning with BLR SAG')
        plt.show()
        
    def samples_param(self, n_samples):
        return [np.random.multivariate_normal(mean = self.theta_values[-1], cov = self.S) for _ in range(n_samples)] # n_samples * dim_params
    

    
def viz_svm(theta, X, y, final_theta = None):
    plt.figure()
    xx = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    if isinstance(theta,list):
        
        for param in theta:
            a = -param[0] / param[1]
            yy = a * xx - (param[-1]) /param[1]
            plt.plot(xx,yy, alpha = 0.2)
            
        if final_theta is not None:
            a = -final_theta[0] / final_theta[1]
            yy = a * xx - (final_theta[-1]) / final_theta[1]
            plt.plot(xx,yy, label = 'Final SVM Hyperplane')
        else:
            mean_theta = np.mean(np.array(theta), axis = 0)
            a = -mean_theta[0] / mean_theta[1]
            yy = a * xx - (mean_theta[-1]) / mean_theta[1]
            plt.plot(xx,yy, label = 'SVM Hyperplane')
        
    else:
        a = -theta[0] / theta[1]
        yy = a * xx - (theta[-1]) / theta[1]
        plt.plot(xx,yy, label = 'SVM Hyperplane')

    plt.scatter([x[0] for x in X], [x[1] for x in X], c = y)
    plt.title('SVM')
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    
def smoothed_hinge_loss(prod):
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
        return -y
    elif prod<1:
        return -(1-prod)*y
    else:
        return 0
    
def f_pred(theta, x):
    return x@theta[:-1] - theta[-1]

def grad_f_pred(theta, x):
    last_comp = np.zeros(theta.shape[0])
    last_comp[-1] = 1
    big_x = np.concatenate((x,np.zeros(1)))
    return big_x - last_comp 
    
if __name__ == "__main__":
    X = np.r_[np.random.randn(20, 2) - [4, 4],np.random.randn(20, 2) + [4, 4]]
    y = np.array([-1] * 20 + [1] * 20)
    
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target*2-1

    sigma = 1e-4
    hinge_loss = Loss(smoothed_hinge_loss, grad_smoothed_hinge_loss)
    predictor = Predictor(f_pred, grad_f_pred, dim_theta = X.shape[1]+1)
    sag = SAG(X,y,predictor,hinge_loss, batch_size = 200, learning_rate=0.001, sigma=np.eye(X.shape[1]+1)*sigma)
    final_theta = sag.optimize(2500, infotime=500, sgd = False, gd = False, sampled = False)
    sag.plot()
    
    # To plot the SVM
    thetas_list = sag.samples_param(100)
    viz_svm(thetas_list, X, y, final_theta)
    
    # Scores 
    y_pred = f_pred(final_theta, X)
    print('Accuracy',metrics.accuracy_score(y, np.sign(y_pred)))
    
    # Plot Hinge loss and final point
    x = np.linspace(-1000000,1000000,100000)
    hinge_loss_values = [smoothed_hinge_loss(x_i) for x_i in x]
    plt.figure()
    plt.plot(x,hinge_loss_values)
    plt.scatter(y_pred@y,smoothed_hinge_loss(y_pred@y))
    plt.title('Smoothed Hinge Loss')
    plt.show()