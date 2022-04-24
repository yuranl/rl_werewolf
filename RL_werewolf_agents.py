# Code source: Worksheet W12D2
# Heavily Modified

import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_werewolf_helpers import epsilon_greedy


# Q Network: given the game states, decide what to do
# Structure: input -> 1024 -> 128 -> n_actions, linear + ReLU, first 2 with dropout

class QNetwork_sigmoid(nn.Module):
  def __init__(self, input_size, n_actions, n_roles):
    super().__init__()
    self.flatten = nn.Flatten(start_dim=0)
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc_act = nn.Linear(128, n_actions)
    self.fc_identity = nn.Linear(128, n_roles)
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
    x = torch.sigmoid((self.drop(self.fc1(x))))
    x = torch.sigmoid((self.drop(self.fc2(x))))
    x = self.decoders[act_type](x)
    # Returning something that is not softmax-ed.
    return x

class QNetwork_relu(nn.Module):
  def __init__(self, input_size, n_actions, n_roles):
    super().__init__()
    self.flatten = nn.Flatten(start_dim=0)
    self.fc1 = nn.Linear(input_size, 1024)
    self.fc2 = nn.Linear(1024, 128)
    self.fc_act = nn.Linear(128, n_actions)
    self.fc_identity = nn.Linear(128, n_roles)
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

class QNetwork_convolution(nn.Module):
  def __init__(self, input_size, n_actions, n_roles):
    super().__init__()
    # input shape: 10 * 12 * 37
    self.flatten = nn.Flatten()
    self.flatten1 = nn.Flatten(0)
    self.conv = nn.Conv1d(12 * 37, 64, 10, stride=5)
    self.fc2 = nn.Linear(192, 128)
    self.fc_act = nn.Linear(128, n_actions)
    self.fc_identity = nn.Linear(128, n_roles)
    self.fc_evaluation = nn.Linear(128, n_actions)
    self.fc_vote = nn.Linear(128, n_actions)
    self.drop = nn.Dropout()
    self.decoders = {"act": self.fc_act, "identity": self.fc_identity, "evaluation": self.fc_evaluation, "vote": self.fc_vote}

    torch.nn.init.xavier_uniform_(self.conv.weight)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    torch.nn.init.xavier_uniform_(self.fc_act.weight)
    torch.nn.init.xavier_uniform_(self.fc_identity.weight)
    torch.nn.init.xavier_uniform_(self.fc_evaluation.weight)
    torch.nn.init.xavier_uniform_(self.fc_vote.weight)

  def forward(self, x, act_type):
    x = self.flatten(x).permute(1,0)[None,:,:]
    x = self.flatten1(self.drop(self.conv(x)))
    # print(x.shape)
    x = torch.sigmoid((self.drop(self.fc2(x))))
    x = self.decoders[act_type](x)
    x /= torch.sum(x)
    # Returning something that is not softmax-ed.
    return x

class QNetwork_random(nn.Module):
  def __init__(self, input_size, n_actions, n_roles):
    super().__init__()
    self.decoder_dims = {"act": 12, "identity": 6, "evaluation": 12, "vote": 12}

  def forward(self, x, act_type):
    res = torch.rand(self.decoder_dims[act_type])
    res /= torch.sum(res)
    # Returning something that is not softmax-ed.
    return res

class QNetworkAgent:
  def __init__(self, policy1, policy2, q_net, optimizer):
    self.policy1 = policy1
    self.policy2 = policy2
    self.q_net = q_net
    self.optimizer = optimizer
  
  def act(self, state, action):
    # on selecting, we do not grad
    with torch.no_grad():
      if action != "identity":
        return self.policy1(self.q_net, state, action)
      else:
        return self.policy2(self.q_net, state, action)
  
  def train(self, state, action, reward, next_state, decoder_type):
    # Predicted Q value
    # action: type of decoder
    q_pred = self.q_net(state, decoder_type)
    q_pred = q_pred[int(action)]
    with torch.no_grad():
      q_target = torch.max(self.q_net(next_state, decoder_type))
      q_target = reward + q_target

    loss = F.mse_loss(q_pred, q_target)
    if not isinstance(self.q_net, QNetwork_random):
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

# n_steps = 50000
# gamma = 0.99
# epsilon = 0.1

# q_net = QNetwork(300, 12).to(device)
# policy = epsilon_greedy(env.num_actions(), epsilon)
# optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
# agent = QNetworkAgent(policy, q_net, optimizer)
# eps_b_qn = learn_env(env, agent, gamma, n_steps)


# Some scripts for training the agents