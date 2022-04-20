import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def assign_roles(num_players, num_wolves, pwh):
  status = []
  # Prophet, witch, hunters
  status.append(3)
  status.append(4)
  status.append(5)
  for i in range(num_wolves):
    status.append(1)
  for i in range(num_players-3-num_wolves):
    status.append(2)
  random.shuffle(status)
  return(status)

def check_end_reason(alive_status):
  if alive_status[0] == 0:
    return "Wolves Lost!"
  elif alive_status[1] == 0:
    return "Villagers Lost!"
  elif alive_status[2] == 0:
    return "Deities Killed!"
  else:
    return "Wait... It shouldn't end here."

def check_ended(alive_status):
  return min(alive_status) == 0

def check_alives(status):
  # Checking the remaining alives for each of the powers
  wolves = [id for id, player in enumerate(status) if player==1]
  villagers = [id for id, player in enumerate(status) if player==2]
  deities = [id for id, player in enumerate(status) if player > 2]
  return (len(wolves), len(villagers), len(deities))

def epsilon_greedy(q_values, epsilon):
    if np.random.random() > epsilon:
        action = np.argmax(q_values)
    else:
        action = np.random.choice(len(q_values))
    return action