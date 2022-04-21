from RL_werewolf_agents import *
from RL_werewolf_game import *
from RL_werewolf_helpers import *
import time

# Initializing agents

def generate_trios():
    net = QNetwork(3120, 12, 6)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    agent = QNetworkAgent(None, net, opt)
    return net, opt, agent

villager_net, villager_opt, villager_agent = generate_trios()
werewolf_net, werewolf_opt, werewolf_agent = generate_trios()
seer_net, seer_opt, seer_agent = generate_trios()
witch_net, witch_opt, witch_agent = generate_trios()
hunter_net, hunter_opt, hunter_agent = generate_trios()
fool_net, fool_opt, fool_agent = generate_trios()

# Running - Random Results
results = defaultdict(int)

start_time = time.time()
prev_weights = villager_agent.q_net.fc1.weight
for i in range(10):

  game = Game(total_round=10, agents=(villager_agent, werewolf_agent, seer_agent, witch_agent, hunter_agent, fool_agent))
  results[game.run()] += 1
  print(results)
print(results)
after_weights = villager_agent.q_net.fc1.weight
print(torch.sum(torch.abs(after_weights - prev_weights)))
print("--- %s seconds ---" % (time.time() - start_time))