import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class MyReLU_horder(object):
    def forward(self, x, sample):
        self.x = x
        return x * (x > 0)
        
    def backward(self, grad_output, grad_output2):
        return grad_output * (self.x > 0), np.zeros_like(self.x)
        
    def step(self, learning_rate):
        pass  # no parameters to update

class MySigmoid_horder(object):
    def forward(self, x, sample = False):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output, grad_output_2):
        return grad_output * self.output * (1 - self.output), grad_output_2 * self.output * (1 - self.output) * (1 - 2 * self.output)

    def step(self, learning_rate):
        pass  # no parameters to update


class MyLinear_1st_order(object):
    def __init__(self, n_input, n_output, eps):
        self.n_output = n_output
        self.n_input = n_input
        self.mu_w = np.random.randn(n_output, n_input).reshape(-1, 1)
        self.mu_b = np.random.randn(n_output, 1)
        self.sigma_w = (np.random.randn(n_output, n_input).reshape(-1, 1))**2
        self.sigma_b = np.random.randn(n_output, 1)**2
        self.eps = eps

    def forward(self, x, sample = True, n_samples = 1):
        self.x = x
        #self.weights = np.random.normal(self.mu_w, np.sqrt(self.sigma_w)).reshape(self.n_output, self.n_input)
        if n_samples == 1:
            self.weights = np.random.multivariate_normal(mean = self.mu_w.reshape(-1), cov = np.diag(self.sigma_w.flatten()))
            self.weights = self.weights.reshape(self.n_output, self.n_input)

            self.bias = np.random.multivariate_normal(self.mu_b.reshape(-1), np.diag(self.sigma_b.flatten()))

            return np.dot(x, self.weights.T) + self.bias

        self.weights = np.random.multivariate_normal(mean = self.mu_w.reshape(-1), cov = np.diag(self.sigma_w.flatten())
                                                    , size=n_samples)
        self.weights = self.weights.reshape(n_samples, self.n_output, self.n_input) # shape (n_samples, n_output, n_input)


        #self.bias = np.random.normal(self.mu_b, np.sqrt(self.sigma_b))
        self.bias = np.random.multivariate_normal(self.mu_b.reshape(-1), np.diag(self.sigma_b.flatten())
                                                , size=n_samples) #shape (n_samples, mu_b shape)
        return np.dot(x[np.newaxis, :], self.weights.T) + self.bias



    def backward(self, grad_output, grad_output_2):
        self.grad_weights = np.outer(grad_output, self.x)

        self.grad_bias = np.sum(grad_output, axis = 0)
        return self.weights.T@grad_output, grad_output_2
    
    def step(self, learning_rate):
        

        self.sigma_w = ((1 - learning_rate) * self.sigma_w + (self.grad_weights**2).reshape(-1, 1) * learning_rate) + self.eps
        self.sigma_b = (1 - learning_rate) * self.sigma_b + self.grad_bias**2 * learning_rate +  self.eps

        self.mu_w -= (self.grad_weights.reshape(-1, 1) / self.sigma_w) * learning_rate
        self.mu_b -= (self.grad_bias / self.sigma_b) * learning_rate


class MyLinear_2nd_order(object):
    def __init__(self, n_input, n_output, eps):
        self.n_output = n_output
        self.n_input = n_input
        self.mu_w = np.random.randn(n_output, n_input).reshape(-1, 1)
        self.mu_b = np.random.randn(n_output, 1)
        self.sigma_w = (np.random.randn(n_output, n_input).reshape(-1, 1))**2
        self.sigma_b = np.random.randn(n_output, 1)**2
        self.eps = eps

    def forward(self, x, sample = False):
        self.x = x
        #self.weights = np.random.normal(self.mu_w, np.sqrt(self.sigma_w)).reshape(self.n_output, self.n_input)
        if sample:
            self.weights = np.random.multivariate_normal(mean = self.mu_w.reshape(-1), cov = np.diag(self.sigma_w.flatten()))
            self.weights = self.weights.reshape(self.n_output, self.n_input)

        #self.bias = np.random.normal(self.mu_b, np.sqrt(self.sigma_b))
            self.bias = np.random.multivariate_normal(self.mu_b.reshape(-1), np.diag(self.sigma_b.flatten()))


        return np.dot(x, self.weights.T) + self.bias

    def backward(self, grad_output, grad_output_2):
        self.grad_weights = np.outer(grad_output, self.x)
        self.grad_weights_2 = np.outer(grad_output_2, self.x**2)

        #self.grad_bias = np.sum(grad_output, axis = 0)
        #self.grad_bias_2 = np.sum(grad_output_2, axis = 0)
        self.grad_bias = grad_output.copy()
        self.grad_bias_2 = grad_output_2.copy()

        return self.weights.T@grad_output, self.weights.T@grad_output_2
    
    def step(self, learning_rate = 0.001):

        self.sigma_w = ((1 - learning_rate) * self.sigma_w + (self.grad_weights_2).reshape(-1, 1) * learning_rate) + self.eps
        self.sigma_b = (1 - learning_rate) * self.sigma_b + self.grad_bias_2.reshape(-1, 1) * learning_rate + self.eps

        self.mu_w -= (self.grad_weights.reshape(-1, 1) / self.sigma_w) * learning_rate
        self.mu_b -= (self.grad_bias / np.diag(self.sigma_b)).reshape(-1, 1) * learning_rate

class MLP_horder(object):
    def __init__(self, layers, order):
        self.layers = layers
        self.order = order

    def forward(self, x, sample = True):
        for layer in self.layers:
            x = layer.forward(x, sample)
        return x
    
    def compute_loss(self, out, label):
        BCE = -label*np.log(out[0])+(1-label)*np.log(1-out[0])
        self.grad_output = -(label/(out[0]+1e-6) - (1 - label) / (1 - out[0]+1e-6))
        self.grad_output_2 = (label/(out[0]+1e-6)**2 + (1 - label) / (1 - out[0]+1e-6)**2)

        return BCE

    def backward(self):
        grad_output = self.grad_output
        grad_output_2 = self.grad_output_2
        for layer in reversed(self.layers):
            if self.order == 1:
                grad_output, grad_output_2 = layer.backward(grad_output, grad_output_2)
            elif self.order == 2:
                grad_output, grad_output_2 = layer.backward(grad_output, grad_output_2)
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

def plot_data(ax, X, Y):
    plt.axis('off')
    ax.scatter(X[:, 0], X[:, 1], s=1, c=Y, cmap='bone')




# plot the decision boundary of our classifier
def plot_decision_boundary(ax, X, Y, classifier, x_min=-1.5, x_max=2.5, y_min=-1, y_max=1.5):
    # forward pass on the grid, then convert to numpy for plotting
    # Define the grid on which we will evaluate our classifier
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                        np.arange(y_min, y_max, .1))

    to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))
    Z = []
    for x in to_forward:
        Z.append(classifier.forward(x))
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    
    # plot contour lines of the values of our classifier on the grid
    ax.contourf(xx, yy, Z>0.5, cmap='Blues')
    
    # then plot the dataset
    plot_data(ax, X,Y)

def plot_decision_boundary_comparison(ax, X, Y, classifier1, classifier2, x_min=-1.5, x_max=2.5, y_min=-1, y_max=1.5):
    # Définition de la grille pour évaluer nos classificateurs
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                         np.arange(y_min, y_max, .1))

    to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))
    Z1 = []
    Z2 = []
    for x in to_forward:
        Z1.append(classifier1.forward(x))
        Z2.append(classifier2.forward(x))
    Z1 = np.array(Z1).reshape(xx.shape)
    Z2 = np.array(Z2).reshape(xx.shape)
    
    # Tracer les lignes de frontière de décision pour chaque classificateur
    contour1 = ax.contour(xx, yy, Z1, levels=[0.5], colors='blue', linestyles='solid', label='BLR')
    contour2 = ax.contour(xx, yy, Z2, levels=[0.5], colors='red', linestyles='dashed', label='SGD')
    
    # Créer des entrées de légende spécifiques pour chaque classificateur
    lines = [plt.Line2D([0], [0], color='blue', linestyle='solid'),
             plt.Line2D([0], [0], color='red', linestyle='dashed')]
    labels = ['BLR', 'SGD']
    
    # Ajouter la légende avec les lignes et étiquettes spécifiées
    ax.legend(lines, labels)
    
    # Puis tracer le jeu de données
    plot_data(ax, X, Y)


def train_and_plot_horder(net, X, Y, x_min=-1.5, x_max=2.5,
y_min=-1, y_max=1.5, n_iter=20000, learning_rate=1e-2):

    fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    losses = []
    for it in tqdm(range(n_iter)):
        
        j = np.random.randint(1, len(X))


        example = X[j:j+1][0]
        
        label = Y[j]

        sample = True

        out = net.forward(example, sample=sample)

        loss = net.compute_loss(out, label)
        
        losses.append(loss)
        
        
        net.backward()
        

        net.step(learning_rate)
        
        
        # draw the current decision boundary every 250 examples seen
        if it % 1000 == 0 : 
            fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plot_decision_boundary(ax, X, Y, net)
            plt.show()
    return losses

def train_and_plot(net, X, Y, x_min=-1.5, x_max=2.5,
                   y_min = -1, y_max = 1.5, n_iter = 20000, learning_rate = 1e-3):
    # unfortunately animation is not working on colab
    # you should comment the following line if on colab
    fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    losses = []
    for it in range(n_iter):
        
        # pick a random example id
        j = np.random.randint(1, len(X))


        # select the corresponding example and label
        example = X[j:j+1][0]
        
        label = Y[j]
        

        # do a forward pass on the example
        out = net.forward(example)

        # compute the loss according to your output and the label
        # YOUR CODE HERE
        loss = net.compute_loss(out, label)
        
        losses.append(loss)
        
        
        # backward pass
        # YOUR CODE HERE
        net.backward()
        
        # gradient step
        # YOUR CODE HERE
        net.step(learning_rate)

        # draw the current decision boundary every 250 examples seen
        if it % 1000 == 0 : 
            fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plot_decision_boundary(ax, X,Y, net)
            plt.show()
    return losses

def train_and_plot_comparisons(net1, net2, X, Y, x_min=-1.5, x_max=2.5,
                   y_min = -1, y_max = 1.5, n_iter = 20000, learning_rate = 1e-3, save = False):
    # unfortunately animation is not working on colab
    # you should comment the following line if on colab
    fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    losses1 = []
    losses2 = []

    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)

    subfolder_name = 'comparison'

    # Chemin complet du sous-dossier
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Créer le sous-dossier
    os.makedirs(subfolder_path, exist_ok=True)
    indice = 0


    for it in range(n_iter):
        
        # pick a random example id
        j = np.random.randint(1, len(X))


        # select the corresponding example and label
        example = X[j:j+1][0]
        
        label = Y[j]
        

        # do a forward pass on the example
        out1 = net1.forward(example)
        out2 = net2.forward(example)

        # compute the loss according to your output and the label
        loss1 = net1.compute_loss(out1, label)
        loss2 = net2.compute_loss(out2, label)        
        losses1.append(loss1)
        losses2.append(loss2)        
        
        # backward pass
        net1.backward()
        net2.backward() 

        # gradient step
        net1.step(learning_rate)
        net2.step(0.001)
        # draw the current decision boundary every 250 examples seen
        if it % 1000 == 0 : 
            fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plot_decision_boundary_comparison(ax, X,Y, net1, net2)
            if save:
                plt.savefig(f'{subfolder_path}/{indice}_decision_boundary.png')
                plt.close()
                indice += 1
            else:
                plt.show()
    return losses1, losses2

def train_and_register(net, X, Y, x_min=-1.5, x_max=2.5,
                       y_min = -1, y_max = 1.5, n_iter = 20000, learning_rate = 1e-3):
    fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    losses = []
    learning_rate = 1e-3

    # Créer un dossier pour sauvegarder les images
    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)

    subfolder_name = f'{learning_rate}'

    # Chemin complet du sous-dossier
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Créer le sous-dossier
    os.makedirs(subfolder_path, exist_ok=True)


    # Boucle principale d'apprentissage
    for it in tqdm(range(20000)):
        j = np.random.randint(1, len(X))
        example = X[j:j+1][0]
        label = Y[j]

        out = net.forward(example)
        loss = net.compute_loss(out, label)
        losses.append(loss)
        
        net.backward()
        net.step(learning_rate)
        # Enregistrer l'image de la frontière de décision tous les 250 exemples
        if it % 200 == 0:
            fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            plot_decision_boundary(ax, x_min, x_max, y_min, y_max, X, Y, net)
            plt.savefig(f'{subfolder_path}/{it}_decision_boundary.png')
            plt.close()


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

class MyLinear(object):
    def __init__(self, n_input, n_output):
        self.n_output = n_output
        self.n_input = n_input
        self.mu_w = np.random.randn(n_output, n_input).reshape(-1, 1)
        self.mu_b = np.random.randn(n_output, 1)
        self.sigma_w = (np.ones(n_output * n_input).reshape(-1, 1))*1e-3
        self.sigma_b = np.ones(n_output).reshape(-1, 1)*1e-3

    def forward(self, x):
        self.x = x
        #self.weights = np.random.normal(self.mu_w, np.sqrt(self.sigma_w)).reshape(self.n_output, self.n_input)
        self.weights = np.random.multivariate_normal(mean = self.mu_w.reshape(-1), cov = np.diag(self.sigma_w.flatten()))
        self.weights = self.weights.reshape(self.n_output, self.n_input)


        #self.bias = np.random.normal(self.mu_b, np.sqrt(self.sigma_b))
        self.bias = np.random.multivariate_normal(self.mu_b.reshape(-1), np.diag(self.sigma_b.flatten()))


        return np.dot(x, self.weights.T) + self.bias

    def backward(self, grad_output):
        self.grad_weights = np.outer(grad_output, self.x)

        self.grad_bias = np.sum(grad_output, axis = 0)
        return self.weights.T@grad_output
    
    def step(self, learning_rate = 0.01):

        self.mu_w -= (self.grad_weights.reshape(-1, 1)) * learning_rate
        self.mu_b -= (self.grad_bias) * learning_rate

class Sequential(object):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def compute_loss(self, out, label):
        BCE = -label*np.log(out[0])+(1-label)*np.log(1-out[0])
        self.grad_output = -(label/(out[0]+10**(-10)) - (1 - label) / (1 - out[0]+10**(-10)))
        return BCE
    
    def backward(self):
        grad_output = self.grad_output
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        # return grad_output

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)



        


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

class MyLinear_deterministic(object):
    def __init__(self, n_input, n_output):
        self.weights = np.random.randn(n_output, n_input)
        self.bias = np.random.randn(n_output)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights.T) + self.bias

    def backward(self, grad_output):
        # y_i = \sum_j W_{i,j} x_j + b_i
        # d y_i / d W_{i, j} = x_j
        # d loss / d y_i = grad_output[i]
        # so d loss / d W_{i,j} = x_j * grad_output[i]  (by the chain rule)
        dl_dw = np.outer(grad_output, self.x)
        
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
    
    def step(self, learning_rate):
        self.weights -= self.grad_weights * learning_rate
        self.bias -= self.grad_bias * learning_rate
        
    
