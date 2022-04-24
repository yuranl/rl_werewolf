import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def epsilon_greedy(n_actions, epsilon, device):
  def policy_fn(q_net, state, action):
    if torch.rand(1) < epsilon:
      res = torch.rand(n_actions)
      return res
    else:
      with torch.no_grad():
        q_pred = q_net(state, action)
        return q_pred
  return policy_fn

# reward for each player at end of game
# returns 12*1 array, reward for each player
"""
Team rewards:
Base reward: 100 for winning and 0 for losing
Turns ahead reward: 10 for each turn ahead [i.e., minimum number of eliminations opponent need to win instead] (40 MAX)
Turns behind penalty: -10 for each turn behind (-40 MAX)
Fast win reward: 5 for each round before the 8th around for the winning team (game needs at least 2 rounds to end, 30 MAX)
    Note that minimum team reward for winning a game is 110 (at least ahead by one turn)
              maximum team reward for winning a game is 170
    Note that maximum team reward for losing a game is -10 (at least behind by one turn)
              minimum team reward for losing a game is -40

Role-specific rewards (if applied):
Villagers:
  correctly vote on a werewolf during elimination votes: +4 (60 MAX)
  incorrectly vote on a villager during elimination votes: -2 (-30 MAX)
Werewolves:
  successfully voting out a villager: +8 (56 MAX)
  voting out a teammate: -10 (-30 MAX)
"""
def compute_final_reward(game):
  REWARD_WIN = 100
  REWARD_TURN = 10
  PENALTY_TURN = -10
  REWARD_FAST = 5
  REWARD_FAST_THRESHOLD = 8

  ROLE_REWARDS = False # whether to give role-specific rewards


  if not game.ended:
    raise Exception("final reward called before game ended")

  reward = torch.zeros(game.num_players)

  # Team rewards
  wol, civ, dei = game.check_alives() # final survival counts

  wolf_team_reward = 0
  villager_team_reward = 0

  if game.curr_round < game.total_round: # if game ended with one team wining; if game ended after too long, both teams considered lost
    fast_win_reward = REWARD_FAST * max(REWARD_FAST_THRESHOLD - game.curr_round, 0)
    if wol > 0: # wolf team wins
      wolf_team_reward += REWARD_WIN
      wolf_team_reward += fast_win_reward
      wolf_turns_ahead = wol
      if not (wolf_turns_ahead == max(wol-civ, wol-dei)):
        print(wolf_team_reward)
        print(wol, civ, dei)
        print(game.curr_round)
        print(game.total_round)
      wolf_team_reward += (REWARD_TURN * wolf_turns_ahead)
      villager_team_reward += (PENALTY_TURN * wolf_turns_ahead)

    else:
      villager_team_reward += REWARD_WIN
      villager_team_reward += fast_win_reward
      villager_turns_ahead = min(civ, dei)
      # assert(villager_turns_ahead > 0)
      villager_team_reward += (REWARD_TURN * villager_turns_ahead)
      wolf_team_reward += (PENALTY_TURN * villager_turns_ahead)
    
  else: #still compute turn rewards when game ends after running too long
    wolf_turns_ahead = max(wol-civ, wol-dei)
    villager_turns_ahead = -wolf_turns_ahead
    if wolf_turns_ahead > 0: # wolf is ahead
      wolf_team_reward += (REWARD_TURN * wolf_turns_ahead)
      villager_team_reward += (PENALTY_TURN * wolf_turns_ahead)
    else: # villager is ahead
      villager_team_reward += (REWARD_TURN * villager_turns_ahead)
      wolf_team_reward += (PENALTY_TURN * villager_turns_ahead)

  for i in range(game.num_players):
    role = int(torch.argmax(game.roles[i]))
    player_reward = 0
    if role < 1: # wolf team
      player_reward += wolf_team_reward
    else: # villager team
      player_reward += villager_team_reward


  return wolf_team_reward, villager_team_reward