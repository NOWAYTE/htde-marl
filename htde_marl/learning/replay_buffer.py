import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: dict):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
