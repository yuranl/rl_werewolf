from RL_werewolf_agents import *
from RL_werewolf_game import *
from RL_werewolf_helpers import *


# Initializing agents


# Running - Random Results
results = defaultdict(int)
for i in range(1):
  game = Game(agent_nets=())
  results[game.run()] += 1
print(results)