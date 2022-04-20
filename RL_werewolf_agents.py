# Code source: Worksheet W12D2
# Heavily Modified

import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Q Network: given the game states, decide what to do
# Structure: input -> 1024 -> 128 -> n_actions, linear + ReLU, first 2 with dropout

class QNetwork(nn.Module):
  def __init__(self, input_size, n_actions):
    super().__init__()
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc3 = nn.Linear(128, n_actions)
    self.drop = nn.Dropout()

  def forward(self, x):
    x = torch.relu((self.drop(self.fc1(x))))
    x = torch.relu((self.drop(self.fc2(x))))
    x = torch.relu((self.fc3(x)))
    # Returning something that is not softmax-ed.
    return x

class QNetworkAgent:
  def __init__(self, policy, q_net, optimizer):
    self.policy = policy
    self.q_net = q_net
    self.optimizer = optimizer
  
  def act(self, state):
    # on selecting, we do not grad
    with torch.no_grad():
      return self.policy(self.q_net, state)
  
  def train(self, state, action, reward, discount, next_state, frame):
    # Predicted Q value
    q_pred = self.q_net(state) # .gather(1, action)

    # Now compute the q-value target (also called td target or bellman backup) (no grad) 
    with torch.no_grad():
      # get the best Q-value from the next state (there are still multiple action choices, so we still need to choose among these)
      """TODO: 问题是 我们一轮结束之后才能知道下一次的state，不是吗（当然狼人也可以说，我知道刀了谁之后下一个再刀谁最好，云云；
      这些确实也对。但是，也许我们可以只用states的value? 然后根据这些去train state values
      或者，就是直接假定刀完之后没有其他因素介入（至少在算的时候），然后选择最好的"""
      q_target = self.q_net(next_state).max(dim=1)[0].view(-1, 1)
      """TODO:看一下这个.view函数"""
      # Next apply the reward and discount to get the q-value target
      q_target = reward + discount * q_target
    # Compute the MSE loss between the predicted and target values
    loss = F.mse_loss(q_pred, q_target)

    # backpropogation to update the q network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

n_steps = 50000
gamma = 0.99
epsilon = 0.1

# q_net = QNetwork(300, 12).to(device)
# policy = epsilon_greedy(env.num_actions(), epsilon)
# optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
# agent = QNetworkAgent(policy, q_net, optimizer)
# eps_b_qn = learn_env(env, agent, gamma, n_steps)


# Some scripts for training the agents