# Code source: Worksheet W12D2

class QNetwork(nn.Module):
  def __init__(self, n_channels, n_actions):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=n_channels, out_channels=16,
                          kernel_size=3, stride=1)
    self.fc1 = nn.Linear(in_features=1024, out_features=128)
    self.fc2 = nn.Linear(in_features=128, out_features=n_actions)

  def forward(self, x):
    ####################################################################
    # Fill in missing code below (...),
    # then remove or comment the line below to test your function
    # raise NotImplementedError("Q network")
    ####################################################################

    # Pass the input through the convnet layer with ReLU activation
    x = torch.relu((self.conv(x)))
    # Flatten the result while preserving the batch dimension
    x = torch.flatten(x, 1)
    # Pass the result through the first linear layer with ReLU activation
    x = torch.relu(self.fc1(x))
    # Finally pass the result through the second linear layer and return
    x = self.fc2(x)
    return x

# Uncomment below to test your module
env = Environment('breakout', random_seed=522)
q_net = QNetwork(env.n_channels, env.num_actions()).to(device)
env.reset()
state = env.state()
# note: phi() transforms the state to make it compatible with PyTorch
q_net(phi(state))

# Code Source: Worksheet W12D2
# Modified

class QNetworkAgent:
  def __init__(self, policy, q_net, optimizer):
    self.policy = policy
    self.q_net = q_net
    self.optimizer = optimizer
  
  def act(self, state):
    # on selecting, we do not grad
    with torch.no_grad():
      return self.policy(self.q_net, state)
  
  def train(self, state, action, reward, discount, next_state, frame):
    # Predicted Q value
    q_pred = self.q_net(state) # .gather(1, action)

    # Now compute the q-value target (also called td target or bellman backup) (no grad) 
    with torch.no_grad():
      # get the best Q-value from the next state (there are still multiple action choices, so we still need to choose among these)
      TODO: 问题是 我们一轮结束之后才能知道下一次的state，不是吗（当然狼人也可以说，我知道刀了谁之后下一个再刀谁最好，云云；
      这些确实也对。但是，也许我们可以只用states的value? 然后根据这些去train state values
      或者，就是直接假定刀完之后没有其他因素介入（至少在算的时候），然后选择最好的
      q_target = self.q_net(next_state).max(dim=1)[0].view(-1, 1)
      TODO:看一下这个.view函数
      # Next apply the reward and discount to get the q-value target
      q_target = reward + discount * q_target
    # Compute the MSE loss between the predicted and target values
    loss = F.mse_loss(q_pred, q_target)

    # backpropogation to update the q network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()