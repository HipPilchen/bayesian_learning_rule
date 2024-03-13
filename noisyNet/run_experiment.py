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

from models_RL import BNN, MLP, IndividualGradientMLP, Network
# import torchsso
from optimizers import VOGN, Vadam, MyVOGN, MyVadam,  goodfellow_backprop_ggn


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


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        # print("forward :",Q)
        # print("forward :",torch.argmax(Q).item())
        # print(e+1)
    return torch.argmax(Q).item()
    

class dqn_agent:
    def __init__(self, config, model, seed):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
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

        if config["optimizer"] == "Vadam":
            print("Vadam")
            self.optimizer = MyVadam(self.model.parameters(), train_set_size=90, lr=config['learning_rate'], num_samples = 7)
        elif config["optimizer"] == "Adam":
            print("Adam")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        elif config["optimizer"] == "SGD":
            print("SGD")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['learning_rate'])
        else:
            raise ValueError("Invalid optimizer type")
        
        # self.optimizer = torchsso.optim.VOGN(self.model, dataset_size= 70, lr=config['learning_rate'])
        # self.optimizer = optimizer
        
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
                
                if self.config['noisy']:
                    self.model.reset_noise()
                    
                return loss
        
            if self.config["optimizer"] == "SGD":
                QYmax = self.model(Y).max(1)[0].detach()
                #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                outputs = self.model(X)
                QXA = outputs.gather(1, A.to(torch.long).unsqueeze(1))
                
                self.optimizer.zero_grad()
                # output = self.model(data)
                loss = self.criterion(QXA, update.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                if self.config['noisy']:
                    self.model.reset_noise()
            else:
                self.optimizer.step(closure1)
            
            # loss = self.optimizer.step(closure1)

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

def generate_model(config):
    if config['noisy']:
        model = Network(in_dim = state_dim, hidden_dims = [nb_neurons], out_dim = n_action)
        # model = BNN(input_size = state_dim,
        #                     hidden_sizes = [nb_neurons],
        #                     output_size = n_action,
        #                     )
    elif config['model_type'] == 'MLP':
        model = MLP(input_size = state_dim,
                            hidden_sizes = [nb_neurons],
                            output_size = n_action,
                            )
    elif config['model_type'] == 'IndividualGradientMLP':
        model = IndividualGradientMLP(input_size = state_dim,
                            hidden_sizes = [nb_neurons],
                            output_size = n_action,
                            )    
    else:
        raise ValueError("Invalid model type")
    return model

if __name__ == "__main__":
    cartpole = gym.make('CartPole-v1', render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = cartpole.observation_space.shape[0]
    n_action = cartpole.action_space.n 
    nb_neurons=24

    # DQN config
    config = {'nb_actions': cartpole.action_space.n,
            'learning_rate': 0.01,
            'gamma': 0.95,
            'buffer_size': 1000000,
            'epsilon_min': 0.01,
            'epsilon_max': 1.0,
            'epsilon_decay_period': 3000,
            'epsilon_delay_decay': 20,
            'noisy' : True,
            'model_type': 'MLP',
            'number_of_run_per_optimizer': 20,
            'number_of_episode': 1300,
            'batch_size': 30}
    
    model1 = generate_model(config)
    model1 = model1.to(device)
        
    optimizers_to_test = [
        "Vadam",
        "Adam",
        "SGD",
    ]
        
    tab_average_ep_return = []
    for optimizer in optimizers_to_test:
        avr_ep_per_optimizer = np.zeros(config['number_of_episode'])
        print("optimizer : ", optimizer)
        config['optimizer'] = optimizer
        for seed in range(config['number_of_run_per_optimizer']):
            model1 = generate_model(config)
            model1 = model1.to(device)
            agent = dqn_agent(config, model1, seed)   
            episode_return = agent.train(cartpole, config['number_of_episode'])
            avr_ep_per_optimizer += np.array(episode_return)
        
        tab_average_ep_return.append(avr_ep_per_optimizer/config['number_of_run_per_optimizer'])
    
    if config['noisy']:
        name = "true"
    else:
        name = "false"
    np.save(f'results/results_noisy{name}.npy', np.array(tab_average_ep_return))
    for i in range(len(optimizers_to_test)):
        plt.plot(tab_average_ep_return[i], label=str(optimizers_to_test[i]))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    plt.title('Average Rewards after 50 Iterations on the CartPole environment')
    plt.legend()
    plt.savefig(f'results/plot_noisy{name}.png')
    plt.show()