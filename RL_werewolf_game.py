from turtle import end_fill
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
    self.role2_id = self.player_id_role(2)
    self.role3_id = self.player_id_role(3)
    self.role4_id = self.player_id_role(4)
    self.role5_id = self.player_id_role(5)
    self.hunter_night_ability = False # True only if hunter just died and can use his ability at the start of next day

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
    self.dict_rewards = {0: self.werewolf_reward, 1: self.villager_reward, 2: self.seer_reward, 3: self.witch_reward, 4: self.hunter_reward, 5: self.fool_reward}

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
      if self.alive[self.curr_round][i] > 0 and self.roles[i][0] == 1: # alive and is wolf
          # This will be later updated (can do votes on distribution)
          wolf_input = self.prep_input(self.wolf_history)
          vote = self.werewolf_agent.act(wolf_input, "act") # softmax
          vote = F.softmax(vote, dim=0)
          vote_kill += vote
    to_kill = torch.argmax(vote_kill)
    # self.eliminate_vote_history[self.curr_round] = vote_kill

    curr_alive[to_kill] = min(curr_alive[to_kill], 0) # update target to dead
    self.wolf_history[self.curr_round] = vote_kill # update wolf history

    # Seer
    """
    Seer will check highest value target from action array returned by seer network.
    Seer given -1 for a wolf check, and 1 for a villager check.
    """
    seer_id = self.role2_id
    if self.alive[self.curr_round][seer_id] > 0: # action if alive
      seer_input = self.prep_input(self.seer_history)
      seer_action = self.seer_agent.act(seer_input, "act")
      to_check = torch.argmax(seer_action)
      if self.roles[to_check][0] > 0: # checked player is wolf
        self.seer_history[self.curr_round][to_check] = -1
      else: # checked player is villager
        self.seer_history[self.curr_round][to_check] = 1

    # Witch
    """
    Determining witch action:
    In action array returned by witch network:
    If highest value target = kill target, then witch action is SAVE (use antedote)
    Else if lowest value target has value < 0, then witch action is KILL (use poison)
    Else: no action.
    """
    witch_id = self.role3_id
    if self.alive[self.curr_round][witch_id] > 0: # action if alive

      if self.witch_antedote == True: # update witch information with werewolf kill this round
        self.witch_history[self.curr_round][to_kill] = 1
      
      witch_input = self.prep_input(self.witch_history)
      witch_action = self.witch_agent.act(witch_input, "act")
      highest = torch.argmax(witch_action)
      lowest = torch.argmin(witch_action)
      poisoned = None # needed for hunter information
      if highest == to_kill and self.witch_antedote == True:
        curr_alive[to_kill] = 1 # SAVE
        self.witch_antedote = False
        self.witch_history[self.curr_round][to_kill] = 2
      elif witch_action[lowest] < 0 and self.witch_poison == True:
        curr_alive[lowest] = 0 # KILL
        poisoned = lowest
        self.witch_poison = False
        self.witch_history[self.curr_round][lowest] = -1
    
    # Hunter
    """
    Hunter only given information at night:
    By default given that they can use their ability.
    If hunter is poisoned by witch then they can't use their ability.
    """
    hunter_id = self.role4_id
    if self.alive[self.curr_round][hunter_id] > 0: # action if alive
      self.hunter_history[self.curr_round] = torch.ones(self.num_players) # assume can use ability
      if curr_alive[hunter_id] <= 0: # hunter is killed and not saved, can use ability next day
        self.hunter_night_ability = True
      if hunter_id == poisoned: # can't use ability if poisoned by witch
        self.hunter_history[self.curr_round] = torch.zeros(self.num_players) # can't use ability
        self.hunter_night_ability = False
    
    ##################
    ### Process night time results ###
    ##################
    self.alive[self.curr_round] = curr_alive


    ##################
    ### The Day ###
    ##################
    # Who died?
    #curr_alive[to_kill] = min(curr_alive[to_kill], 0)

    # Hunter action time
    """
    If hunter can use ability and dies, he will kill the lowest value target from the action array, if the lowest value is < 0
    Otherwise, ability will not be used.
    """
    if self.hunter_night_ability == True:
        hunter_input = self.prep_input(self.hunter_history)
        hunter_action = self.hunter_agent.act(hunter_input, "act")
        hunter_lowest = torch.argmin(hunter_action)
        if hunter_action[hunter_lowest] < 0:
          self.alive[self.curr_round][hunter_lowest] = 0 # hunter kill
          self.hunter_kill[self.curr_round][hunter_lowest] = 1 # update public information

    # Everyone (showing identity, giving evaluations, voting)
    curr_alive = self.alive[self.curr_round]
    for i in range(self.num_players): 
      if curr_alive[i] > 0: # testimony for each player alive
        role = int(torch.argmax(self.roles[i]))
        input = self.prep_input(self.dict_info[role])
        agent = self.dict_agent[role]
        player_claim = agent.act(input, "identity")
        player_evalution = agent.act(input, "evaluation")
        self.identity_claim_history[self.curr_round][i] = player_claim # update claim and evalution as public information available to players speaking after this player
        self.testimony_history[self.curr_round][i] = player_evalution

    # Voting
    vote_out = torch.zeros(self.num_players)
    actions = torch.zeros(self.num_players)
    for i in range(len(self.roles)):
      if curr_alive[i] > 0:
          # This will be later updated (can do votes on distribution)
          role = int(torch.argmax(self.roles[i]))
          input = self.prep_input(self.dict_info[role])
          agent = self.dict_agent[role]
          vote = torch.argmax(agent.act(input, "vote"))
          vote_out[vote] += 1
          actions[i] = int(vote)
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

    # actions = self.get_actions()
    rewards = self.get_rewards()
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
    wolf_state = self.prep_input(self.wolf_history)
    common_state = self.prep_input(self.civilian_history)
    seer_state = self.prep_input(self.seer_history)
    witch_state = self.prep_input(self.witch_history)
    hunter_state = self.prep_input(self.hunter_history)
    fool_state = self.prep_input(self.civilian_history)
    return wolf_state, common_state, seer_state, witch_state, hunter_state, fool_state

  def get_actions(self):
    # getting the actions of all players
    # Not used right now!
    actions = self.vote_history[self.curr_round-1]
    return actions

  def get_rewards(self):
    # maybe later?
    self.fool_reward = self.villager_reward
    return self.werewolf_reward, self.villager_reward, self.seer_reward, self.witch_reward, self.hunter_reward, self.fool_reward

  def update_all_models(self, beginning_all_states, end_all_states, actions, rewards):

    # Training the voting 
    voting = actions
    for i in range(self.num_players):
      role = int(self.roles_compact[i])
      self.dict_agent[role].train(beginning_all_states[role], voting[i], rewards[role], end_all_states[role], "vote")

    # Training the votekill, explain, etc. everything else, as we have


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
