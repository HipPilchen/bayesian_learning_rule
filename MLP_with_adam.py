import numpy as np


class predictor(): #this is gonna be the MLP
    
    def __init__(self, f, gradient, dim_theta, hessian=None):
        self.f = f
        self.dim_theta = dim_theta
        self.gradient_theta = gradient  # gradient on theta
        self.hess = hessian

    def f(self, theta, x):
        return self.f(theta, x)

    def gradient(self, theta, x):
        return self.gradient_theta(theta, x) # \grad_{\theta} f(\theta, x)

    def hessian(self, theta, x):
        return self.hess(theta, x)
    

class MyLinear(object):
    def __init__(self, n_input, n_output):
        self.n_output = n_output
        self.n_input = n_input
        self.mu_w = np.random.randn(n_output, n_input).reshape(-1)
        self.mu_b = np.random.randn(n_output)
        self.sigma_w = (np.random.randn(n_output, n_input).reshape(-1))**2
        self.sigma_b = np.random.randn(n_output)**2

    def forward(self, x):
        self.x = x
        weights = np.random.normal(self.mu_w, np.diag(np.sqrt(self.sigma_w))).reshape(self.n_output, self.n_input)
        bias = np.random.normal(self.mu_b, np.sqrt(self.sigma_b))
        
        return np.dot(x, weights.T) + bias

    def backward(self, grad_output):
        # y_i = \sum_j W_{i,j} x_j + b_i
        # d y_i / d W_{i, j} = x_j
        # d loss / d y_i = grad_output[i]
        # so d loss / d W_{i,j} = x_j * grad_output[i]  (by the chain rule)
        dl_dw = np.outer(grad_output, self.x)
        dl_dw = next(iter(dl_dw))
        
        # d y_i / d b_i = 1
        # d loss / d y_i = grad_output[i]
        dl_dy = grad_output
        
        
        # now we need to compute the gradient with respect to x to continue the back propagation
        # d y_i / d x_j = W_{i, j}
        # to compute the gradient of the loss, we have to sum over all possible y_i in the chain rule
        # d loss / d x_j = \sum_i (d loss / d y_i) (d y_i / d x_j)
        # YOUR CODE HERE
        #dl_dx = dl_dy @ self.weights
        #print(dl_dy)
        #print('&')
        #print(self.weights)
        dl_dx = [sum([dl_dy[i]*self.weights[i,j] for i in range(len(self.weights))]) for j in range(len(self.weights[0]))]
    
        self.grad_weights = dl_dw
        self.grad_bias = np.sum(grad_output, axis=0)
        return dl_dx
    
    def step(self, learning_rate = 0.01):

        self.sigma_w = (1 - learning_rate) * self.sigma_w + self.grad_weights**2 * learning_rate
        self.sigma_b = (1 - learning_rate) * self.sigma_b + self.grad_bias**2 * learning_rate

        self.mu_w -= (self.grad_weights / self.sigma_w) * learning_rate
        self.mu_b -= (self.grad_bias / self.sigma_b) * learning_rate

    
class MLP(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss3(self, out, label):    
        if out==0:
            out=1e-10
        loss=-np.mean((label * np.log(out) + (1-label) * np.log(1-out)))
        self.grad_output=-((label/out)+(label-1)/(1-out))
        return loss
    
    def compute_loss(self, out, label):
        BCE = -label*np.log(out[0])+(1-label)*np.log(1-out[0])
        self.grad_output = -(label/(out[0]+10**(-10)) - (1 - label) / (1 - out[0]+10**(-10)))
        return BCE
    
    def compute_loss2(self, out, label):
        # case where label is 0
        if label == 0:
            grad_output = -1 / (1 - out + 1e-10)
            loss0 = -np.mean(np.log(1 - out + 1e-10))
            self.grad_output = grad_output
            return loss0
        
        # case where label is 1
        grad_output = 1 / (out + 1e-10)
        loss1 = -np.mean(np.log(out + 1e-10))
        self.grad_output = grad_output
        return loss1

    def backward(self):
        grad_output = self.grad_output
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        # return grad_output

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)

    def train(self, X, y, n_iter, learning_rate):
        for _ in range(n_iter):
            loss = 0
            for x, label in zip(X, y):
                out = self.forward(x)
                loss += self.compute_loss(out, label)
                self.backward()
            loss /= len(X)
            self.step(learning_rate)
            print(loss)


class MyReLU(object):
    def forward(self, x):
        self.x = x
        return x * (x > 0)
        
    def backward(self, grad_output):
        return grad_output * (self.x > 0)
        
    def step(self, learning_rate):
        pass  # no parameters to update

class MySigmoid(object):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

    def step(self, learning_rate):
        pass  # no parameters to update



        


class bayesian_optimizer():

    def __init__(self, X, y, predictor, loss, n_samples, loss_function, n_iter = 1000, 
                 learning_rate = 0.01, batch_size = 100, m = None, s = None):
        self.n_samples = n_samples
        self.loss_function = loss_function
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.f = predictor
        self.loss = loss
        self.grad = np.zeros((X.shape[0], self.f.dim_theta))
        self.dim_params = self.f.dim_theta
        self.theta = m
        self.theta_values = []
        self.loss_values = []
        self.X = X
        self.y = y
        self.s = s

    def init_parameters(self):
        if self.theta is None:
            self.theta = np.random.randn(self.dim_params)
        self.theta_values.append(self.theta)
        self.loss_values.append(self.loss.loss(self.f.f(self.theta, self.X), self.y))

    def step():
        pass
    
    def zero_grad():
        pass
        
    
