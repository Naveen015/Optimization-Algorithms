import numpy as np
from collections import deque
import random


class ppoMemory:
    def __init__(self, replay_memory_size=5000, mini_batch_size=8):
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.replay_memory = deque(maxlen=self.replay_memory_size)

    def save_transition(self, current_state, action, reward, new_state, done):
        transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def return_mini_buffer(self):
        mini_batch = random.sample(self.replay_memory, self.mini_batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        new_current_states = np.array([transition[3] for transition in mini_batch])
        return mini_batch, current_states, new_current_states

