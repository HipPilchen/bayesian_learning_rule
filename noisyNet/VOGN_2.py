import random
import torch
import numpy as np
from tqdm import trange
import gymnasium as gym
import torch
import torch.nn as nn
import gymnasium as gym

import gymnasium as gym

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models_RL import BNN, MLP, IndividualGradientMLP
import torchsso
from optimizers2 import VOGN, Vadam, goodfellow_backprop_ggn


# import torchsso

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

cartpole = gym.make('CartPole-v1', render_mode="rgb_array")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = cartpole.observation_space.shape[0]
n_action = cartpole.action_space.n 
nb_neurons=24

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        # print("forward :",Q)
        # print("forward :",torch.argmax(Q).item())
        # print(e+1)
    return torch.argmax(Q).item()
    
# Declare network
state_dim = cartpole.observation_space.shape[0]
n_action = cartpole.action_space.n 
nb_neurons=24

class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.config = config
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.criterion = torch.nn.MSELoss()
        # self.optimizer = torchsso.optim.VOGN(self.model, dataset_size= 30, lr=config['learning_rate'])
        self.optimizer = Vadam(self.model.parameters(), train_set_size=70, lr=config['learning_rate'])
        # self.optimizer = VOGN(self.model.parameters(), train_set_size=6, lr=config['learning_rate'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
            # def __init__(self, model: nn.Module, dataset_size: float, curv_type: str, curv_shapes: dict, curv_kwargs: dict,
            #      lr=0.01, momentum=0., momentum_type='preconditioned',
            #      grad_ema_decay=1., grad_ema_type='raw', weight_decay=0.,
            #      normalizing_weights=False, weight_scale=None,
            #      acc_steps=1, non_reg_for_bn=False, bias_correction=False,
            #      lars=False, lars_type='preconditioned',
            #      num_mc_samples=10, val_num_mc_samples=10,
            #      kl_weighting=1, warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=None,
            #      prior_variance=1, init_precision=None,
            #      seed=1, total_steps=1000):
                # default_kwargs = dict(lr=1e-3,
                #               curv_type='Cov',
                #               curv_shapes={
                #                   'Linear': 'Diag',
                #                   'Conv2d': 'Diag',
                #                   'BatchNorm1d': 'Diag',
                #                   'BatchNorm2d': 'Diag'
                #               },
                #               curv_kwargs={'ema_decay': 0.01, 'damping': 1e-7},
                #               warmup_kl_weighting_init=0.01, warmup_kl_weighting_steps=1000,
                #               grad_ema_decay=0.1, num_mc_samples=50, val_num_mc_samples=100)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            
            def closure1():
                QYmax = self.model(Y).max(1)[0].detach()
                #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                outputs = self.model(X)
                QXA = outputs.gather(1, A.to(torch.long).unsqueeze(1))
                
                self.optimizer.zero_grad()
                # output = self.model(data)
                loss = self.criterion(QXA, update.unsqueeze(1))
                loss.backward()
                return loss, outputs
            
            def closure2():
                QYmax = self.model(Y).max(1)[0].detach()
                #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                outputs = self.model(X)
                QXA = outputs.gather(1, A.to(torch.long).unsqueeze(1))
                
                self.optimizer.zero_grad()
                # output = self.model(data)
                loss = self.criterion(QXA, update.unsqueeze(1))
                loss.backward()
                return loss

            def closure():
                # QYmax, H_list, Z_list = self.model.forward(Y, individual_grads=True)
                QYmax = self.model.forward(Y, individual_grads=False).max(1)[0].detach()
                # QYmax = QYmax.max(1)[0].detach()
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                self.optimizer.zero_grad()
                outputs, H_list, Z_list =  self.model.forward(X, individual_grads=True)
                QXA = outputs.gather(1, A.to(torch.long).unsqueeze(1))
                
                loss = self.criterion(QXA, update.unsqueeze(1))

                linearGrads = torch.autograd.grad(loss, Z_list)

                grad, ggn = goodfellow_backprop_ggn(H_list, linearGrads)
                
                del H_list, Z_list
                return loss, grad, ggn, X.size()[0]

            # loss1 = optimizer1.step(closure1)
            loss = self.optimizer.step(closure1)
            
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step() 
            # if self.config['noisy']:
            #     self.model.reset_noise()
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            
            # select epsilon-greedy action
            if self.config['noisy']:
                action = greedy_action(self.model, state)
            else:
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

# DQN config
config = {'nb_actions': cartpole.action_space.n,
          'learning_rate': 0.01,
          'gamma': 0.95,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.0,
          'epsilon_decay_period': 3000,
          'epsilon_delay_decay': 20,
          'noisy' : False,
          'batch_size': 30}
# DQN = 
# if config['noisy']:
#     DQN = Network(state_dim, nb_neurons, n_action)
# else:
#     DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
#                           nn.ReLU(),
#                           nn.Linear(nb_neurons, nb_neurons),
#                           nn.ReLU(), 
#                           nn.Linear(nb_neurons, n_action)).to(device)
# model_kwargs = dict(input_size=2, output_size=None, hidden_sizes=[128])
# model1 = BNN(input_size = state_dim,
#                          hidden_sizes = [nb_neurons],
#                          output_size = n_action,
#                          )
# model1 = MLP(input_size = state_dim,
#                          hidden_sizes = [nb_neurons],
#                          output_size = n_action,
#                          )
model1 = IndividualGradientMLP(input_size = state_dim,
                         hidden_sizes = [nb_neurons],
                         output_size = n_action,
                         )
model1 = model1.to(device)
# Train agent
agent = dqn_agent(config, model1)
scores = agent.train(cartpole, 2000)
plt.plot(scores)