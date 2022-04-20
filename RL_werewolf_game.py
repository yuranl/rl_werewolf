from RL_werewolf_helpers import *
from RL_werewolf_agents import *

import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# Initialization
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##@title Game Class

class Game:
  def __init__(self, num_players=12, num_wolves=4, pwh=True, total_round=10):

    # Read input
    self.num_players = num_players
    self.num_wolves = num_wolves
    self.pwh = pwh
    self.total_round = total_round

    # Initialize status array
    self.status = assign_roles(self.num_players, self.num_wolves, self.pwh)
    self.ended = False
    self.witch_poison = True
    self.witch_antedote = True
    self.vote_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players -> vote (in terms of player id)
    self.evaluation_history = torch.zeros((self.total_round, self.num_players, self.num_players)) # size: total_round * num_players * num_players (turn number, player id, evaluated player id) -> evaluation
    self.kill_vote_history = torch.zeros((self.total_round, self.num_players, self.num_wolves)) # size: total_round * num_players * num_wolves (turn number, player id, which wolf) -> probability distribution
    self.prophet_history = torch.zeros((self.total_round, 2)) # size: 2 * total_round (each turn, record id detected and good / bad)
    self.witch_history = torch.zeros((self.total_round, 2)) # size: 2 * total_round (poison / antedote used at which round)
    self.hunter_history = torch.zeros(self.total_round) # length: total_round


  def next_round(self):

    ####################
    ### The Night ###
    ####################
    # Prepare situation for the night
    still_alive = [id for id, player in enumerate(self.status) if player > 0]
    vote_kill = [0] * self.num_players

    #########################################
    ### Every Player does his thing ###
    #########################################
    # Wolves


    """
    Meta-code:
    (in environment: self.status)
    act = player.act(self.status) # player acts based on the current environment
    # Do we share the same player network across the same group?
    # i.e. for 5 villagers do we train the same network, and update 5 times each round?
    # also, we could do something like Double DQN - keep 5 networks, update only after
    # each turn - to avoid doing such things. This seems good
    reward, new_turn_number = self.act(act) / nextround(act)
    player.train(act, reward, current_status) # or anything else that we need
    """
    for player in self.status:
      if player > 0:
        # Still alive
        if player == 1:
          # This will be later updated (can do votes on distribution)
          vote = random.choice(still_alive)
          vote_kill[vote] += 1
    to_kill = random.choice([id for id, count in enumerate(vote_kill) if count==max(vote_kill)])

    # Prophet

    # Witch

    # Hunter


    ##################
    ### The Day ###
    ##################
    # Ordinary Villager (voting to out people)

    # Update the situation
    if self.status[to_kill] > 0:
      self.status[to_kill] = -self.status[to_kill]


    # What are the results? i.e. rewards
    reward = get_reward(self.status) # or, conditional evaluation
    
    # Check situation
    alives = check_alives(self.status)
    if check_ended(alives):
      return check_end_reason(alives), False
    return "", True
  
  def run(self):
    message, running = "", True
    while running:
      message, running = self.next_round()
    return message


# Running - Random Results
results = defaultdict(int)
for i in range(100):
  game = Game()
  results[game.run()] += 1
print(results)