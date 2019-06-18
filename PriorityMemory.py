import random
from collections import namedtuple, deque
import torch
import numpy as np
from SumTree import SumTree

class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.priority_eps = 0.005
        self.priority_alfa = 0.3
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.sumtree = SumTree(buffer_size)

    def add(self, state, action, reward, next_state, done,error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        priority = self.error2priority(error)
        #self.memory.append(e)
        self.sumtree.add(priority,e)

    def update(self, idx, error):
        p = self.error2priority(error)
        self.sumtree.update(idx, p)

    def error2priority(self, errors):
        return np.power(np.abs(errors) + self.priority_eps, self.priority_alfa)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        '''
        experiences = []
        idxs = []
        priorities = []

        segment = self.sumtree.total() / self.batch_size

        for i in range(self.batch_size):
            segment_bottom_boundery = segment * i
            segment_upper_boundery = segment * (i + 1)
            # get the random value with each segment, this ensures randomnes within priorities
            uniform_sample = random.uniform(segment_bottom_boundery, segment_upper_boundery)
            (idx, p, data) = self.sumtree.get(uniform_sample)
            idxs.append(idx)
            priorities.append(p)
            experiences.append(data)


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (idxs,states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.sumtree.size()