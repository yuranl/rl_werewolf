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
  def __init__(self, num_players=12, num_wolves=4, roles="swhf", total_round=10, agent_nets=None):

    # Read input
    self.num_players = num_players
    self.num_wolves = num_wolves
    self.roles = roles # default villager roles: seer, witch, hunter, fool (swhf)
    # dict_identity = {0: "werewolf", 1: "villager", 2: "seer", 3: "witch", 4: "hunter", 5: "fool"}
    self.num_roles = 6 # number of different roles
    self.total_round = total_round # maximum game rounds, game will be forced to end after this many rounds
    self.curr_round = 0

    # true player identities (not given to players)
    self.roles = self.assign_roles() # size: num_players * num_roles (one hot encoding for players' true identities)

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


    # Player Agents
    self.villager_net, self.werewolf_net, self.seer_net, self.witch_net, self.hunter_net, self.fool_net = agent_nets

  def next_round(self):

    ####################
    ### The Night ###
    ####################
    # Prepare situation for the night

    # Get current alive players
    curr_alive = self.alive[self.curr_round]

    #########################################
    ### Every Player does his thing ###
    #########################################

    # Wolves
    """
    Meta-code:
    act = player.act(self.status) # player acts based on the current environment
    # Do we share the same player network across the same group?
    # i.e. for 5 villagers do we train the same network, and update 5 times each round?
    # also, we could do something like Double DQN - keep 5 networks, update only after
    # each turn - to avoid doing such things. This seems good
    reward, new_turn_number = self.act(act) / nextround(act)
    player.train(act, reward, current_status) # or anything else that we need
    """
    vote_kill = torch.zeros(self.num_players)
    for i in range(len(self.roles)):
      if curr_alive[i] > 0 and self.roles[i][0] == 1:
          # This will be later updated (can do votes on distribution)
          wolf_input = self.prep_input(self.wolf_history)
          vote = F.softmax(self.werewolf_net(wolf_input)) # softmax
          vote_kill += vote
    to_kill = torch.argmax(vote_kill)

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

    self.alive[self.curr_round + 1] = curr_alive
    self.curr_round += 1


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
    status = torch.zeros((self.num_players, self.num_roles))
    status[0][2] = 1 # seer
    status[1][3] = 1 # witch
    status[2][4] = 1 # hunter
    status[3][5] = 1 # fool
    status[4:8, 1] = 1 # villagers
    status[8:12, 0] = 1 # werewolves
    status = status[torch.randperm(status.size()[0])]
    return(status)


  # Checks for final conditions

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
    
  def prep_input(self, private_info):
    return torch.cat([self.alive, self.hunter_kill, self.vote_history,
      self.identity_claim_history, self.testimony_history, self.eliminate_vote_history, private_info])


# Running - Random Results
results = defaultdict(int)
for i in range(1):
  game = Game(agent_nets=())
  results[game.run()] += 1
print(results)