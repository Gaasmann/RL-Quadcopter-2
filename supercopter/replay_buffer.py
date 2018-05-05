"""Replay buffer for DDPG algo"""

from collections import namedtuple, deque
import numpy as np

Record = namedtuple('Record', field_names=['start_state', 'action', 'reward',
                                           'end_state'])


class ReplayBuffer:
    """Replay buffer"""
    def __init__(self, maxlen=100):
        self._buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self._buffer)

    def add(self, record):
        """Add a new record"""
        if not isinstance(record, Record):
            raise TypeError('record is not an object of type Record')
        self._buffer.append(record)

    def sample(self, size):
        """Get a sample of elements contained in the buffer"""
        if size > self._buffer.maxlen:
            raise ValueError('Size can\'t be > ReplayBuffer\'s maxlen')
        if size == 0 or len(self) == 0:
            return []
        size = len(self) if len(self) < size else size
        elected = np.random.choice(len(self), size, replace=False)
        res = []
        for idx in elected:
            res.append(self._buffer[idx])
        return res
