import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def epsilon_greedy(q_values, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(q_values)
    else:
        action = np.random.choice(len(q_values))
    return action