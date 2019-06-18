import random
from collections import namedtuple, deque
import torch
import numpy as np
from SumTree import SumTree

class PriorityReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):
        self.priority_eps = 0.005
        self.priority_alfa = 0.3
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.sumtree = SumTree(buffer_size)

    def add(self, state, action, reward, next_state, done,error):
        #convert to experience
        e = self.experience(state, action, reward, next_state, done)
        #calculate the priority
        priority = self.error2priority(error)
        #append to memory
        self.sumtree.add(priority,e)

    def update(self, idx, error):
        #calculate the priority and update the leaf in sum tree
        p = self.error2priority(error)
        self.sumtree.update(idx, p)

    def error2priority(self, errors):
        #convert td-error to priority
        return np.power(np.abs(errors) + self.priority_eps, self.priority_alfa)

    def sample(self):
        #lists for result
        experiences = []
        idxs = []
        priorities = []
        #number of segment
        segment = self.sumtree.total() / self.batch_size

        for i in range(self.batch_size):
            segment_bottom_boundery = segment * i
            segment_upper_boundery = segment * (i + 1)
            # get the random value with each segment, this ensures randomnes within priorities
            uniform_sample = random.uniform(segment_bottom_boundery, segment_upper_boundery)
            (idx, p, data) = self.sumtree.get(uniform_sample)
            #append to result
            idxs.append(idx)
            priorities.append(p)
            experiences.append(data)
        #convert to tensor
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (idxs,states, actions, rewards, next_states, dones)

    def __len__(self):
        # the current size of sumtree
        return self.sumtree.size()