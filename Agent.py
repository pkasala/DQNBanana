import numpy as np
import random

from Model import QNetwork
from PriorityMemory import PriorityReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 80000  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

EPS_START= 1
EPS_MIN = 0.001
EPS_DECAY= 0.99991

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        #Replay memory
        self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # number of steps over all experiment
        self.t_step = 0

        self.eps = EPS_START

    def greedy_action(self, state):
        #convet from numpy to tensor
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        #perform prediction
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        #calculate the current epsilon
        self.eps = max(EPS_MIN,EPS_DECAY * self.eps) 
        # Epsilon-greedy action selection
        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        #increase the number of steps
        self.t_step += 1
        # calcualte the error pro priority memory
        error =  self.calc_td_error(state, action, reward, next_state, done)
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done,error)
        #learn
        if  (self.t_step) % UPDATE_EVERY == 0:
            # If enough samples are available in memory start learning
            if len(self.memory) > BATCH_SIZE:
                self.learn()

    def calc_td_error(self,state, action, reward, next_state, done):

        #convert to tensor
        next_state = torch.from_numpy(np.vstack([next_state])).float()
        state = torch.from_numpy(np.vstack([state])).float()
        max_action_local = self.qnetwork_local(next_state).detach().max(1)[1]
        Q_target = self.qnetwork_target(next_state).detach()[0][max_action_local]
        # Compute Q targets for current states
        Q_target = reward + (GAMMA * Q_target * (1 - done))
        # Get expected Q values from local model, detach from autogradient
        Q_local = self.qnetwork_local(state).detach()[0][action]
        #calculate error and return
        return (Q_local - Q_target).item()

    def learn(self):

        experiences = self.memory.sample()
        idxs, states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        max_action_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,max_action_local)
        # Compute Q targets for current states
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_local = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_local, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #copy weights from local network to target
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        #calculate the errors for priority
        errors = torch.abs(Q_local - Q_targets).data.numpy()
        #update the experiments with new priorities
        for i in range(len(idxs)):
            self.memory.update(idxs[i], errors[i])

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

