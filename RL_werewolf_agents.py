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
    self.conv_act = nn.Conv1d(12 * 37, 64, 10, stride=5)
    self.conv_identity = nn.Conv1d(12 * 37, 64, 10, stride=5)
    self.conv_evaluation = nn.Conv1d(12 * 37, 64, 10, stride=5)
    self.conv_vote = nn.Conv1d(12 * 37, 64, 10, stride=5)
    self.fc2_act = nn.Linear(192, 128)
    self.fc2_identity = nn.Linear(192, 128)
    self.fc2_evaluation = nn.Linear(192, 128)
    self.fc2_vote = nn.Linear(192, 128)
    self.fc_act = nn.Linear(128, n_actions)
    self.fc_identity = nn.Linear(128, n_roles)
    self.fc_evaluation = nn.Linear(128, n_actions)
    self.fc_vote = nn.Linear(128, n_actions)
    self.drop = nn.Dropout()
    self.decoders = {"act": self.fc_act, "identity": self.fc_identity, "evaluation": self.fc_evaluation, "vote": self.fc_vote}
    self.fc2s = {"act": self.fc2_act, "identity": self.fc2_identity, "evaluation": self.fc2_evaluation, "vote": self.fc2_vote}
    self.convs = {"act": self.conv_act, "identity": self.conv_identity, "evaluation": self.conv_evaluation, "vote": self.conv_vote}

    torch.nn.init.xavier_uniform_(self.conv_act.weight)
    torch.nn.init.xavier_uniform_(self.conv_identity.weight)
    torch.nn.init.xavier_uniform_(self.conv_evaluation.weight)
    torch.nn.init.xavier_uniform_(self.conv_vote.weight)
    torch.nn.init.xavier_uniform_(self.fc2_act.weight)
    torch.nn.init.xavier_uniform_(self.fc2_identity.weight)
    torch.nn.init.xavier_uniform_(self.fc2_evaluation.weight)
    torch.nn.init.xavier_uniform_(self.fc2_vote.weight)
    torch.nn.init.xavier_uniform_(self.fc_act.weight)
    torch.nn.init.xavier_uniform_(self.fc_identity.weight)
    torch.nn.init.xavier_uniform_(self.fc_evaluation.weight)
    torch.nn.init.xavier_uniform_(self.fc_vote.weight)

  def forward(self, x, act_type):
    x = self.flatten(x).permute(1,0)[None,:,:]
    x = self.flatten1(self.drop(self.convs[act_type](x)))
    # print(x.shape)
    x = torch.sigmoid((self.drop(self.fc2s[act_type](x))))
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
  
  # Act_type <==> decoder_type
  def act(self, state, act_type):
    # on selecting, we do not grad
    with torch.no_grad():
      if act_type != "identity":
        return self.policy1(self.q_net, state, act_type)
      else:
        return self.policy2(self.q_net, state, act_type)
  
  def train(self, state, action, reward, next_state, decoder_type):
    # Predicted Q value
    # action: type of decoder
    q_pred = self.q_net(state, decoder_type)
    q_pred = q_pred[int(torch.argmax(action))]
    with torch.no_grad():
      q_target = torch.max(self.q_net(next_state, decoder_type))
      q_target = reward + q_target

    loss = F.mse_loss(q_pred, q_target)
    if not isinstance(self.q_net, QNetwork_random):
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()