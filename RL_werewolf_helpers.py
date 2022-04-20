import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def epsilon_greedy(n_actions, epsilon, device):
  def policy_fn(q_net, state):
    if torch.rand(1) < epsilon:
      return torch.randint(n_actions, size=(1,), device=device)
    else:
      with torch.no_grad():
        q_pred = q_net(state)
        return torch.argmax(q_pred).view(1,)
  return policy_fn

# reward for each player at end of game
# returns 12*1 array, reward for each player
def compute_final_reward(game):
  if not game.ended:
    raise Exception("reward given before game ended")
  
  pass