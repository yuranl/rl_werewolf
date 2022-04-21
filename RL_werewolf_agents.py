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
    self.flatten = nn.Flatten(start_dim=0)
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc_act = nn.Linear(128, n_actions)
    self.fc_identity = nn.Linear(128, n_actions)
    self.fc_evaluation = nn.Linear(128, n_actions)
    self.fc_vote = nn.Linear(128, n_actions)
    self.drop = nn.Dropout()
    self.decoders = {"act": self.fc_act, "identity": self.fc_identity, "evaluation": self.fc_evaluation, "vote": self.fc_vote}

    torch.nn.init.xavier_uniform_(self.fc1.weight)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    torch.nn.init.xavier_uniform_(self.fc_act.weight)
    torch.nn.init.xavier_uniform_(self.fc_identity.weight)
    torch.nn.init.xavier_uniform_(self.fc_evaluation.weight)
    torch.nn.init.xavier_uniform_(self.fc_vote.weight)

  def forward(self, x, act_type):
    x = self.flatten(x)
    x = torch.relu((self.drop(self.fc1(x))))
    x = torch.relu((self.drop(self.fc2(x))))
    x = self.decoders[act_type](x)
    # Returning something that is not softmax-ed.
    return x

class QNetworkAgent:
  def __init__(self, policy, q_net, optimizer):
    self.policy = policy
    self.q_net = q_net
    self.optimizer = optimizer
  
  def act(self, state, action):
    # on selecting, we do not grad
    with torch.no_grad():
      return self.q_net(state, action) # self.policy(self.q_net, state)
  
  def train(self, state, action, reward, next_state, decoder_type):
    # Predicted Q value
    # action: type of decoder
    q_pred = self.q_net(state, decoder_type)[int(action)]
    # print(sum(torch.abs(next_state - state)))

    # Now compute the q-value target (also called td target or bellman backup) (no grad) 
    with torch.no_grad():
      # get the best Q-value from the next state (there are still multiple action choices, so we still need to choose among these)
      q_target = max(self.q_net(next_state, decoder_type))
      # Next apply the reward and discount to get the q-value target
      q_target = reward + q_target
    # Compute the MSE loss between the predicted and target values
    loss = F.mse_loss(q_pred, q_target)
    print(loss.is_leaf)

    # backpropogation to update the q network
    self.optimizer.zero_grad()
    # print(loss.grad)
    loss.backward()
    # print(loss.grad)
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