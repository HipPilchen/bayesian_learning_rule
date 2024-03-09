import numpy as np
import torch

print(torch.__version__)

class DQN_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN_model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)



class MySoftmax(object):
    def forward(self, x):
        # Shift x for numerical stability
        shiftx = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shiftx)
        self.softmax_output = exps / np.sum(exps, axis=-1, keepdims=True)
        return self.softmax_output
    
    def backward(self, grad_output, a):
        # The gradient of softmax is the derivative of the output wrt the input,
        # which involves the softmax output itself. We use the Jacobian matrix for the calculation.
        # For each output, the gradient depends on all other outputs because of the normalization step in softmax.
        
        # Initialize the tensor for the Jacobian matrix of the gradients
        s = self.softmax_output.reshape((-1, 1))
        # Diagonal - probabilities for the derivative of softmax
        jacobian_m = np.diagflat(s) - np.dot(s, s.T)
        
        # For each class probability, compute the gradient with respect to each input feature

        grad_input = grad_output @ jacobian_m.T[a].reshape(1, -1)
        return grad_input
        grad_input = np.dot(jacobian_m, grad_output.reshape(s.shape[0], -1))
        
        # Reshape back to the original grad_output shape
        return grad_input.reshape(grad_output.shape)
    
    def step(self, learning_rate):
        # No parameters to update in softmax
        pass
    
    def copy(self):
        # create a copy of the layer
        return MySoftmax()
    
    def print(self):
        # print the parameters of the layer
        print("SoftMax layer")


class MyReLU(object):
    def forward(self, x):
        # the relu is y_i = max(0, x_i)
        self.save_for_backward = (x > 0).astype('float')
        return self.save_for_backward * x
        
    
    def backward(self, grad_output, a):
        # the gradient is 1 for the inputs that were above 0, 0 elsewhere
        self.grad = self.save_for_backward * grad_output
        # print('Relu end')
        return self.grad
    
    def step(self, learning_rate):
        # no need to do anything here, since ReLU has no parameters
        pass
    
    def copy(self):
        # create a copy of the layer
        return MyReLU()
    
    def print(self):
        # print the parameters of the layer
        print("Relu layer")

class MySigmoid(object):
    def forward(self, x):
        # the sigmoid is y_i = 1./(1+exp(-x_i))
        self.save_for_backward = np.exp(-x)
        return 1. / (1 + self.save_for_backward)
    
    def backward(self, grad_output, a):
        # the partial derivative is e^-x / (e^-x + 1)^2
        self.grad = (grad_output * self.save_for_backward) / (self.save_for_backward + 1) ** 2
        return self.grad
    
    def step(self, learning_rate):
        # no need to do anything here since Sigmoid has no parameters
        pass
    
    def copy(self):
        # create a copy of the layer
        return MySigmoid()
    
    def print(self):
        # print the parameters of the layer
        print("Sigmoid layer")
    
class MyLinear(object):
    def __init__(self, n_input, n_output):
        # initialize two random matrices for W and b (use np.random.randn)
        self.W = np.random.randn(n_input, n_output)
        self.b = np.random.randn(1,n_output)
        # self.W = np.ones((n_input, n_output))
        # self.b = np.ones((1,n_output))
        

    def forward(self, x):
        # save a copy of x, you'll need it for the backward
        # return xW + b
        self.saved_for_backward = x.copy()
        return x @ self.W + self.b

    def backward(self, grad_output, a):
        # y_i = \sum_j x_jW_{j,i} + b_i
        # d y_i / d W_{j, i} = x_j
        # d loss / d y_i = grad_output[i]
        # so d loss / d W_{j,i} = x_j * grad_output[i]  (by the chain rule)
        self.grad_W = (grad_output.T @ self.saved_for_backward).T
    
        # d y_i / d b_i = 1
        # d loss / d y_i = grad_output[i]
        self.grad_b = grad_output.copy()
        
        # now we need to compute the gradient with respect to x to continue the back propagation
        # d y_i / d x_j = W_{j, i}
        # to compute the gradient of the loss, we have to sum over all possible y_i in the chain rule
        # d loss / d x_j = \sum_i (d loss / d y_i) (d y_i / d x_j)
        
        # print("Pb here, shapes :")
        # print(grad_output.shape, self.W.shape)
        try:
            return grad_output @ self.W.T
        except:
            return grad_output @ self.W.T[a].reshape(1, -1)
    
    def step(self, learning_rate):
        # update self.W and self.b in the opposite direction of the stored gradients, for learning_rate
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b
        pass
    
    def copy(self):
        # create a copy of the layer
        new_layer = MyLinear(self.W.shape[0], self.W.shape[1])
        new_layer.W = self.W.copy()
        new_layer.b = self.b.copy()
        return new_layer
    
    def print(self):
        # print the parameters of the layer
        print("The layers has a shape : ", self.W.shape)
    
class MySequential_model(object):
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def compute_loss(self, out, label):
        # Calculate the MSE loss
        self.loss = 0.5 * (out - label) ** 2
        
        # Calculate the gradient of the MSE loss
        # The gradient does not depend on the value of the label in a case-by-case manner as in BCE,
        # so we don't need to split the computation.
        self.loss_grad = out - label
        
        return self.loss
    
    def compute_loss_BCE(self, out, label):
        # use the BCE loss
        # -(label * log(output) + (1-label) * log(1-output))
        # save the gradient, and return the loss      
        # beware of dividing by zero in the gradient.
        # split the computation in two cases, one where the label is 0 and another one where the label is 1
        # add a small value (1e-10) to the denominator
        self.loss = - (label * np.log(out) + (1-label) * np.log(1-out))
        if label == 0:
            self.loss_grad =  (1-label) / (1-out + 1e-10)
        else:
            self.loss_grad = - label / (out + 1e-10)
        
        return self.loss

    def backward(self, a):
        # apply backprop sequentially, starting from the gradient of the loss
        current_grad = self.loss_grad.reshape((-1, 1))
        for layer in reversed(self.layers):
            # layer.print()
            current_grad = layer.backward(current_grad, a)
    
    def print(self):
        for layer in reversed(self.layers):
            layer.print()
    
    def step(self, learning_rate):
        # take a gradient step for each layers
        for layer in self.layers:
            layer.step(learning_rate)
            
    def copy(self):
        # create a copy of the model
        new_layers = [layer.copy() for layer in self.layers]
        return MySequential_model(new_layers)

    def optimize_my_model(self, X, A, R, Y, D, gamma=0.99, lr=1.0):
        QYmax = np.max(self.forward(Y), axis=1)
        update_np = R + (1 - D) * QYmax * gamma
        QXA_np =  self.forward(X)[np.arange(X.shape[0]), A.astype(np.int64)]
        # print(QXA_np)
        # print(update_np)
        loss = self.compute_loss(QXA_np, update_np)
        for a in A:
            self.backward(a)
        self.step(lr)
        # QXA = self.forward(X).gather(1, A.to(torch.long).unsqueeze(1))
        # update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
        # QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        # loss = self.criterion(QXA, update.unsqueeze(1))
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step() )
       
def generate_my_model(input_dim, hidden_dim, output_dim):
    layers = [MyLinear(input_dim, hidden_dim), MyReLU(), MyLinear(hidden_dim, hidden_dim), MyReLU(), MyLinear(hidden_dim, output_dim), MySoftmax()]
    return MySequential_model(layers)     
