import random
import numpy as np
import os

class ReplayBuffer(object):
    def __init__(self, capacity, process_id=None):
        self.capacity = capacity
        self.process_id = process_id
        self.buffer = []
        self.lp_buffer = []

    def add(self, s0, a, r, s1, done, index, fixline_id):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done, index, fixline_id))

    def sample(self, batch_size):
        s0, a, r, s1, done, index, fixline_id = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done, index, fixline_id

    def lp_add(self, s0, a, r):
        if len(self.lp_buffer) >= self.capacity:
            self.lp_buffer.pop(0)
        self.lp_buffer.append((s0[None, :], a, r))

    def lp_sample(self, batch_size):
        s0, a, r = zip(*random.sample(self.lp_buffer, batch_size))
        return np.concatenate(s0), a, r

    def size(self):
        return len(self.buffer)
