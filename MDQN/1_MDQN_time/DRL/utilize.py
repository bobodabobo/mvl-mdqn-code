from collections import deque
import numpy as np
from typing import Tuple
from copy import deepcopy as copy


class CostDelayCache:
    '''A cache for delayed cost assignment.'''
    def __init__(self, delay_time:int, gamma:float=1.0):
        self.delay_time = delay_time
        self.gamma = gamma
        self.maxlen = delay_time + 1
        self.clear()

    def push(self, frame:tuple): # frame: (s, a, c_upf, c_sub, s_next)
        self.cache.append(frame)
        if len(self.cache) == self.maxlen:
            s, a, s_next = self.cache[0][0], self.cache[0][1], self.cache[0][4]
            c = self.cache[0][2] + (self.gamma ** self.delay_time) * self.cache[-1][3]
            r = -c
            transfered_frame = (s, a, r, s_next)
        else:
            transfered_frame = None
        return transfered_frame
    
    def clear(self):
        self.cache = deque(maxlen = self.maxlen)


class ReplayBuffer:
    def __init__(self, max_size:int, seed:int=0):
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)
        self.clear()
    
    def clear(self):
        self.size = 0
        [self.obs_buf, 
         self.action_buf, 
         self.reward_buf, 
         self.obs_next_buf] = [deque(maxlen=self.max_size) for _ in range(4)]

    def store(self, transition: Tuple[np.ndarray, float, float, np.ndarray]):
        transition = copy(transition)
        self.obs_buf.append(transition[0])
        self.action_buf.append(transition[1])
        self.reward_buf.append(transition[2])
        self.obs_next_buf.append(transition[3])
        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample_batch(self, batch_size:int):
        indices = self.rng.choice(self.size, size=batch_size, replace=True)
        s, a, r, s_next = [], [], [], []
        for idx in indices:
            s.append(self.obs_buf[idx])
            a.append(self.action_buf[idx])
            r.append(self.reward_buf[idx])
            s_next.append(self.obs_next_buf[idx])
        s = np.stack(s, 0)
        a = np.array(a).astype(np.int32)
        r = np.array(r).astype(np.float32)
        s_next = np.stack(s_next, 0)
        return s, a, r, s_next
    
    def concate(self, new_buffer):
        len_new_buffer = len(new_buffer.obs_buf)
        for i in range(len_new_buffer):
            self.store((new_buffer.obs_buf[i],
                        new_buffer.action_buf[i],
                        new_buffer.reward_buf[i],
                        new_buffer.obs_next_buf[i]))

    def __len__(self) -> int:
        return self.size
