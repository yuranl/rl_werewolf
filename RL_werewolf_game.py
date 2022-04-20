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
  def __init__(self, num_players=12, num_wolves=4, roles="swhf", total_round=10):

    # Read input
    self.num_players = num_players
    self.num_wolves = num_wolves
    self.roles = roles # default villager roles: seer, witch, hunter, fool (swhf)
    self.num_roles = 6 # number of different roles
    self.total_round = total_round # maximum game rounds, game will be forced to end after this many rounds

    # true player identities (not given to players)
    self.roles = self.assign_roles() # size: num_roles * num_players (one hot encoding for players' true identities)

    # global state variables (not given to players)
    self.ended = False
    self.witch_poison = True
    self.witch_antedote = True

    # public information
    self.alive = torch.ones((self.total_round, self.num_players)) # size: total_round * num_players (turn number, player id) -> whether still alive at this turn
    self.hunter_kill = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players (turn number, player id) -> whether hunter killed this player at this turn
    self.vote_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players -> vote (in terms of player id)
    self.identity_claim_history = torch.zeros((self.total_round, self.num_roles, self.num_players)) # size: total_round * num_players (turn number, role, player id) -> whether player claims to be this role
    self.testimony_history = torch.zeros((self.total_round, self.num_players, self.num_players)) # size: total_round * num_players * num_players (turn number, player id, evaluated player id) -> evaluation
    self.eliminate_vote_history = torch.zeros((self.total_round, self.num_players, self.num_wolves)) # size: total_round * num_players * num_wolves (turn number, player id, which wolf) -> probability distribution
    
    # private information
    self.seer_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players (which player checked and identity given [0: unchecked, 1: villager, -1: werewolf])
    self.witch_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players (werewolf kill target given, poison, antedote used [0: not killed, 1: killed not saved, 2: killed saved, -1: poisoned])
    self.hunter_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players ([all 0 if cannot use ability, all 1 if can use ability, -1 for ability target])
    # self.wolf_history = torch.zeros((self.total_round, self.num_players * 2)) # size: total_round * num_players (average of kill discussion record, space for strategy discussion)
    self.wolf_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players (average of kill discussion record)
    self.civilian_history = torch.zeros((self.total_round, self.num_players)) # size: total_round * num_players (ALL EMPTY) (fool also has no access to private information)


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
    reward = self.get_reward(self.status) # or, conditional evaluation
    
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

  def assign_roles(self):
    status = []
    # Seer, witch, hunters
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


# Running - Random Results
results = defaultdict(int)
for i in range(100):
  game = Game()
  results[game.run()] += 1
print(results)