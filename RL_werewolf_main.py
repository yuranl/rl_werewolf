from RL_werewolf_agents import *
from RL_werewolf_game import *
from RL_werewolf_helpers import *
import time
from tqdm import tqdm
import copy
from matplotlib import pyplot as plt

# Initializing agents

def generate_trios(network_type=QNetwork_sigmoid, lr=1e-3):
    net = network_type(8880, 12, 6)
    opt = None
    if network_type is not QNetwork_random:
      opt = torch.optim.Adam(net.parameters(), lr=lr)
    agent = QNetworkAgent(None, net, opt)
    return net, opt, agent

villager_net, villager_opt, villager_agent = generate_trios(QNetwork_convolution, lr=1e-4)
werewolf_net, werewolf_opt, werewolf_agent = generate_trios(QNetwork_random)
seer_net, seer_opt, seer_agent = generate_trios(QNetwork_convolution, lr=1e-4)
witch_net, witch_opt, witch_agent = generate_trios(QNetwork_convolution, lr=1e-4)
hunter_net, hunter_opt, hunter_agent = generate_trios(QNetwork_convolution, lr=1e-4)
fool_net, fool_opt, fool_agent = generate_trios(QNetwork_convolution, lr=1e-4)

# Running - Random Results
results = np.zeros(4,)
dict_res = {"Wolves Lost!" : 1, "Deities Killed!" : 2, "Villagers Lost!": 3, "Tie!": 0}
epoch = 500
results_history = np.zeros((epoch, 4))

start_time = time.time()
# prev_weights = torch.clone(villager_agent.q_net.fc2.weight).detach_()
for i in tqdm(range(epoch)):
  game = Game(total_round=20, agents=(villager_agent, werewolf_agent, seer_agent, witch_agent, hunter_agent, fool_agent))
  print_info = (i % 5 == 0)
  results[dict_res[game.run(print_info)]] += 1
  results_history[i] = copy.deepcopy(results)
  if i % 100 == 0:
    plt.plot(results_history[:,0], label="Tie")
    plt.plot(results_history[:,1], label="Wolves Lost")
    plt.plot(results_history[:,2], label="Deities Killed")
    plt.plot(results_history[:,3], label="Villagers Lost")
    plt.legend()
    plt.show()

# after_weights = villager_agent.q_net.fc2.weight

# print(torch.sum(torch.abs(after_weights - prev_weights)))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(results_history[:,0], label="Tie")
plt.plot(results_history[:,1], label="Wolves Lost")
plt.plot(results_history[:,2], label="Deities Killed")
plt.plot(results_history[:,3], label="Villagers Lost")
plt.legend()
plt.show()