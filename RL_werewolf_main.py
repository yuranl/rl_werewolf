from RL_werewolf_agents import *
from RL_werewolf_game import *
from RL_werewolf_helpers import *
import time
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt

# Initializing agents

def generate_trios(network_type=QNetwork_sigmoid):
    net = network_type(4440, 12, 6)
    opt = None
    if network_type is not QNetwork_random:
      opt = torch.optim.Adam(net.parameters(), lr=1e-1)
    agent = QNetworkAgent(None, net, opt)
    return net, opt, agent

villager_net, villager_opt, villager_agent = generate_trios()
werewolf_net, werewolf_opt, werewolf_agent = generate_trios(QNetwork_random)
seer_net, seer_opt, seer_agent = generate_trios()
witch_net, witch_opt, witch_agent = generate_trios()
hunter_net, hunter_opt, hunter_agent = generate_trios()
fool_net, fool_opt, fool_agent = generate_trios()

# Running - Random Results
results = np.zeros(4,)
dict_res = {"Wolves Lost!" : 1, "Deities Killed!" : 2, "Villagers Lost!": 3, "Tie!": 0}
epoch = 500
results_history = np.zeros((epoch, 4))

start_time = time.time()
prev_weights = torch.clone(villager_agent.q_net.fc1.weight).detach_()
for i in tqdm(range(epoch)):
  game = Game(total_round=10, agents=(villager_agent, werewolf_agent, seer_agent, witch_agent, hunter_agent, fool_agent))
  results[dict_res[game.run()]] += 1
  results_history[i] = copy.deepcopy(results)
after_weights = villager_agent.q_net.fc1.weight

print(torch.sum(torch.abs(after_weights - prev_weights)))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(results_history[:,0], label="Tie")
plt.plot(results_history[:,1], label="Wolves Lost")
plt.plot(results_history[:,2], label="Deities Killed")
plt.plot(results_history[:,3], label="Villagers Lost")
plt.legend()
plt.show()