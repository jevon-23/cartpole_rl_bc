import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import gym
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style
import os.path
import random

#making the env
env = gym.make("CartPole-v0")
action_space_size = env.action_space.n  # Discrete 2
observation_space_size = env.observation_space.shape[0] # Box (4, )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Loading in expert data, states and actions if they exist
expert_states = torch.tensor(np.load("states_expert.npy"), dtype=torch.float) if os.path.isfile('./states_expert.npy') else None
expert_actions = torch.tensor(np.load("actions_expert.npy"), dtype=torch.float) if os.path.isfile('./states_expert.npy') else None
# print('expert states ', expert_states.shape, 'expert actions', expert_actions.shape)

"""
Data preperation and filtering.
INPUTS:
-states: expert states as tensor
-actions: actions as tensor
-n: window size (how many states needed to predict the next action)
-compare: for filtering the data
OUTPUTS:
-output_states: filtered states as tensor
-output_actions: filtered actions as tensor
"""
def to_input(states, actions, n=2, compare=1):
    if (states is None or actions is None):
        print('there is no expert inputs.')
        return None, None
    count = 0
    index = []
    ep, t, state_size = states.shape    # Might have to take state_size off of thie line, and set it equal to ep * t
    _, _, action_size = actions.shape
    output_states = torch.zeros((ep*(t-n+1), state_size*n), dtype = torch.float)
    output_actions = torch.zeros((ep*(t-n+1), action_size), dtype = torch.float)

    for i in range(ep):
        for j in range(t-n+1):
            #Checking to see if this state or the state after this is equal to that value compare think
            if (states[i, j] == -compare*torch.ones(state_size)).all() or (states[i, j+1] == -compare.torch.ones(state_size)).all():
                index.append([i, j])
            else:
                output_states[count] = states[i, j:j+n].view(-1)
                output_actions[count] = actions[i, j]
                count += 1
    output_states = output_states[:count]
    output_actions = output_actions[:count]
    return output_states, output_actions

#Determining how many trajectories to take from the expert data
if (expert_states is not None and expert_actions is not None):
    num_exp_traj = 50
    a = np.random.randint(expert_states.shape[0] - num_exp_traj)
    print(a)
    expert_states, expert_actions = to_input(expert_states[a: a+num_exp_traj], expert_actions[a: a+num_exp_traj])
    print("expert_state", expert_states.shape)
    print("expert_action", expert_actions.shape)

#concatenating expert states and actions together, 70% training, 30% testing
if (expert_states is not None and expert_actions is not None):
    # adding the end of one row on to theo end of the next
    new_data = np.concatenate((expert_states[:, : observation_space_size], expert_actions), axis=1)
    np.random.shuffle(new_data)
    new_data = torch.tensor(new_data, dtype=torch.float)
    n_samples = int(new_data.shape[0]*0.7)  # 70% of data
    training_set = new_data[:n_samples]
    testing_set = new_data[n_samples:]
    print("training_set", training_set.shape)
    print("testing_set", testing_set.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_actions = action_space_size
        self.num_obs = observation_space_size
        self.fc1 = nn.Linear(self.num_obs, 256)
        self.fc2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Building the nn, Binary Cross Entropy Loss Function, Adams optimizer w/ a learning rate of .01
net = Net()
if (os.path.isfile('./cart2nn')):
    print('prev nn is found')
    net.load_state_dict(torch.load('./cart2nn'))

crit = nn.BCELoss()
opt = optim.Adam(net.parameters(), lr=.01)

loss_list = []
test_list = []
batch_size = 256
n_epoch = 200
lr = .001

for i in range(n_epoch):
    total_loss = 0
    b = 0
    for batch in range(0, training_set.shape[0], batch_size):

        #grab the data
        data = training_set[batch: batch+batch_size, :observation_space_size]
        y = training_set[batch: batch+batch_size, observation_space_size:]

        #train the nn
        y_pred = net(data)
        loss = crit(y_pred, y)
        total_loss += loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        b += 1

    print("Epoch: %i, bceloss: %.6f" % (i+1, total_loss / b))
    display.clear_output(wait=True)
    loss_list.append(total_loss/b)
    x = testing_set[:, :observation_space_size]
    y = testing_set[:, observation_space_size:]
    y_pred = net(x)
    test_list.append(crit(y_pred, y).item())

#Saving the nn to a file
# torch.save(net, "cart2nn")

# uncomment these lines to plot the loss
# plt.plot(test_list, label="Testing Loss")
# plt.xlabel("iterations")
# plt.ylabel("loss")
# plt.legend()
# plt.show()

#Testing inferred actions against real actions
p = 85
print(net(testing_set[p, :observation_space_size]))
print(testing_set[p, observation_space_size:])
crit(net(testing_set[p, :observation_space_size]), testing_set[p, observation_space_size:]).item()

# ======== Testing nn in env =========
# parameters
n = 2 # Window size
n_iter = 5  # max num of interacting w/ env
n_ep = 1000 # num of epochs
max_steps = 500 # Max time steps
gamma = 1.0 # discount factor
seeds = [0, 0, 0, 0, 0]
for i in range(5):
    seeds[i] = random.randrange(1000)
seed_reward_mean = []
seed_reward = []

#Interacting with env
for i in range(n_iter):
    G = []
    G_mean = []
    env.seed(seeds[i])
    torch.manual_seed(seeds[i])
    torch.cuda.manual_seed_all(int(seeds[i]))

    for ep in range(n_ep):
        state = env.reset()
        rew = []
        R = 0

        for step in range(max_steps):
            act = net(torch.tensor(state, dtype=torch.float))
            act = np.clip(act.detach().numpy(), -1, 2)

            next, r, done, _ = env.step(act)
            env.render()
            rew.append(r)
            state = next
            if (done):
                break
        R = sum([rew[i]*gamma**i for i in range(len(rew))])
        G.append(R)
        G_mean.append(np.mean(G))
        if (ep % 10 == 0):
            print("ep = ", ep, ", Mean Reward = ", R)
        display.clear_output(wait=True)
    seed_reward.append(G)
    seed_reward_mean.append(G_mean)

print("iter = ", i, "overall rew = ", np.mean(seed_reward_mean[-1]))
print('closing env')
env.close()
