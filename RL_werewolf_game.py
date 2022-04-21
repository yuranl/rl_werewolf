from xxlimited import foo
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
  def __init__(self, num_players=12, num_wolves=4, roles="swhf", total_round=10, agents=None):

    # Read input
    self.num_players = num_players
    self.num_wolves = num_wolves
    self.roles = roles # default villager roles: seer, witch, hunter, fool (swhf)
    self.num_roles = 6 # number of different roles
    self.total_round = total_round # maximum game rounds, game will be forced to end after this many rounds
    self.curr_round = 0

    # true player identities (not given to players)
    self.roles_compact, self.roles = self.assign_roles() # size: num_players * num_roles (one hot encoding for players' true identities)

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
    self.villager_agent, self.werewolf_agent, self.seer_agent, self.witch_agent, self.hunter_agent, self.fool_agent = agents

    # Rewards for each game
    self.villager_reward, self.werewolf_reward, self.seer_reward, self.witch_reward, self.hunter_reward, self.fool_reward = 0,0,0,0,0,0

    # Helpers
    self.dict_info = {0: self.wolf_history, 1: self.civilian_history, 2: self.seer_history, 3: self.witch_history, 4: self.hunter_history, 5: self.civilian_history}
    self.dict_agent = {0: self.werewolf_agent, 1: self.villager_agent, 2: self.seer_agent, 3: self.witch_agent, 4: self.hunter_agent, 5: self.fool_agent}

  def next_round(self):

    ####################
    ### The Night ###
    ####################
    # Prepare situation for the night

    # Get current alive players
    curr_alive = self.alive[self.curr_round]
    beginning_all_states = self.get_all_states()

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
          vote = self.werewolf_agent.act(wolf_input, "act") # softmax
          vote = F.softmax(vote, dim=0)
          vote_kill += vote
    to_kill = torch.argmax(vote_kill)

    # Prophet

    # Witch

    # Hunter


    ##################
    ### The Day ###
    ##################
    # Who died?
    curr_alive[to_kill] = min(curr_alive[to_kill], 0)

    # Everyone (showing identity, giving evaluations, voting)

    # Voting
    vote_out = torch.zeros(self.num_players)
    for i in range(len(self.roles)):
      if curr_alive[i] > 0:
          # This will be later updated (can do votes on distribution)
          role = int(torch.argmax(self.roles[i]))
          input = self.prep_input(self.dict_info[role])
          agent = self.dict_agent[role]
          vote = torch.argmax(agent.act(input, "vote"))
          vote_out[vote] += 1
    to_vote = torch.argmax(vote_out)
    curr_alive[to_vote] = min(curr_alive[to_vote], 0)

    # Update the situation

    self.alive[self.curr_round + 1] = curr_alive
    self.vote_history[self.curr_round] = vote_out
    self.curr_round += 1


    # What are the results? i.e. rewards
    # reward = self.get_reward(self.status) # or, conditional evaluation
    
    # Check situation
    # print([int(role) if self.alive[self.curr_round][i] == 1 else 9 for i, role in enumerate(self.roles_compact)])
    alives = self.check_alives()

    end_all_states = self.get_all_states()

    # Summarize all losts (incorporate this into the other things, later)
    if self.check_ended(alives):
      result_str = self.check_end_reason(alives)
      if result_str == "Wolves Lost!":
        self.villager_reward += 5
      if result_str == "Deities Killed!":
        self.werewolf_reward += 5
      if result_str == "Villagers Lost!":
        self.werewolf_reward += 5

    
    self.update_all_models(beginning_all_states, end_all_states, actions, rewards)

    if self.check_ended(alives):
      return result_str, False
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
    return torch.flatten(torch.nonzero(status)[:,1]), status

  def get_all_states(self):
    common_state = self.prep_input(self.civilian_history)
    wolf_state = self.prep_input(self.wolf_history)
    seer_state = self.prep_input(self.seer_history)
    witch_state = self.prep_input(self.witch_history)
    hunter_state = self.prep_input(self.hunter_history)
    fool_state = self.prep_input(self.civilian_history)
    return common_state, wolf_state, seer_state, witch_state, hunter_state, fool_state
  # Checks for final conditions
  def check_end_reason(self, alive_status):
    if alive_status[0] == 0:
      return "Wolves Lost!"
    elif alive_status[1] == 0:
      return "Villagers Lost!"
    elif alive_status[2] == 0:
      return "Deities Killed!"
    else:
      return "A tie... no party died out, so far."

  def check_ended(self, alive_status):
    return min(alive_status) == 0 or self.curr_round >= self.total_round - 1

  def check_alives(self):
    # Checking the remaining alives for each of the powers
    alive = self.alive[self.curr_round]
    wolves = [1 for alive, role in zip(alive, self.roles_compact) if role==0 and alive==1]
    villagers = [1 for alive, role in zip(alive, self.roles_compact) if role==1 and alive==1]
    deities = [1 for alive, role in zip(alive, self.roles_compact) if role>1 and alive==1]
    # print(len(wolves), len(villagers), len(deities))
    return (len(wolves), len(villagers), len(deities))
    
  def prep_input(self, private_info):
    result = torch.cat([self.alive[:,:,None], self.hunter_kill[:,:,None], self.vote_history[:,:,None],
      self.identity_claim_history.permute(0,2,1), self.testimony_history, self.eliminate_vote_history,
      private_info[:,:,None]], dim=2)
    return result
