import random

import numpy as np
from tqdm import trange

import gymnasium as gym

from models import generate_my_model

import matplotlib.pyplot as plt

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity # capacity of the buffer
#         self.data = []
#         self.index = 0 # index of the next cell to be filled
#     def append(self, s, a, r, s_, d):
#         if len(self.data) < self.capacity:
#             self.data.append(None)
#         self.data[self.index] = (s, a, r, s_, d)
#         self.index = (self.index + 1) % self.capacity
#     def sample(self, batch_size):
#         batch = random.sample(self.data, batch_size)
#         print(batch)
#         return batch
#     def __len__(self):
#         return len(self.data)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        data = random.sample(self.data, batch_size)
        X, A, R, Y, D = [], [], [], [], []

        for x, a, r, y, d in data:
            X.append(x)
            A.append(a)
            R.append(r)
            Y.append(y)
            D.append(d)
        
        D = np.array(D).astype(int)
        return np.array(X), np.array(A), np.array(R), np.array(Y), D 
    def __len__(self):
        return len(self.data)
    



cartpole = gym.make('CartPole-v1', render_mode="rgb_array")

state_dim = cartpole.observation_space.shape[0]
n_action = cartpole.action_space.n 
nb_neurons=24

# DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
#                           nn.ReLU(),
#                           nn.Linear(nb_neurons, nb_neurons),
#                           nn.ReLU(), 
#                           nn.Linear(nb_neurons, n_action)).to(device)
MyDQN = generate_my_model(state_dim, nb_neurons, n_action)


def greedy_action(network, state):
    # with torch.no_grad(): #TODO: check how to do this with my model
    # Q = network(state).unsqueeze(0)
    Q = network.forward(state)
    print(Q)
    return np.argmax(Q)
    

class dqn_agent:
    def __init__(self, config, model):
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'])
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        # self.criterion = torch.nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            self.model.optimize_my_model(X, A, R, Y, D, gamma=self.gamma, lr=1.0)
            # QYmax = self.model(Y).max(1)[0].detach()
            # #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            # update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            # QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            # loss = self.criterion(QXA, update.unsqueeze(1))
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
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
          'learning_rate': 4.0,
          'gamma': 0.99,
          'buffer_size': 10000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 4000,
          'epsilon_delay_decay': 20,
          'batch_size': 1}

# Train agent
agent = dqn_agent(config, MyDQN)
scores = agent.train(cartpole, 10000)
plt.plot(scores)
# plt.show()