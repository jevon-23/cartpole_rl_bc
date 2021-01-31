import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import os
import cnn
import gym
import numpy as np
from torch.distributions.categorical import Categorical
import sys
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style
plt.style.use("ggplot")

#making the env
env = gym.make("CartPole-v0")
action_space_size = env.action_space.n  # Discrete 2
observation_space_size = env.observation_space.shape[0] # Box (4, )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_checking = __name__ == 'main'
if len(sys.argv) == 2:
    is_checking = sys.argv[1] == 't'    # If a 't' is passed in, that means that we are trying using learning based on checkpoint

""" 4 layered network."""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_actions = action_space_size
        self.num_obs = observation_space_size
        self.fc1 = nn.Linear(self.num_obs, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#Loading in previous net
bc_net_path = './bcnet.pth'
bc_net = Net()
#
# if (os.path.isfile(bc_net_path) and not is_checking):
#     print('previous BC net is found, loading in')
#     try:
#         bc_net.load_state_dict(torch.load(bc_net))
#     except RuntimeError:
#         print('could not load in previous bc_net')

#Loading in the RL net
rl_net_path = './cnet.pth'
rl_net = cnn.Net()
if (not is_checking):
    try:
        print('attempting to load in RL net\n')
        rl_net.load_state_dict(torch.load(rl_net_path))
        print('loaded in RL net')
    except RuntimeError:
        print('could not load in previous dictionary, ./cnet.pth may not exist\n')

#Bincary Cross Entropy Function, Adam optimizer for our net
crit = nn.MSELoss()
opt = optim.Adam(bc_net.parameters(), lr=.001)

"""
Not training the rl_net, but we are getting 'expert data' from the rl_net

Inputs:
    network: rl_net, the net that the behavioral cloning net will be learning from
    batch_size: how many iterations for the episode
    episodes: How many episodes we want to run this net

Returns:
    batch_obs: states that we have seen throughout our iterations
    batch_acts: actions we took based on the states that we saw
    batch_loss: the values of the loss that we were getting back from the states
"""
def run_RL_net(network, episodes, batch_size=256):
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_loss = []         # for the loss
    batch_rew = []          # rewards for plotting

    #Reset episode specific variables, all return values of step()
    obs = env.reset()   # First obs, gets updated as we go on
    done = False    # Checking to see if we are done

    for _ in range(episodes):
        while True:
            batch_obs.append(obs.copy())    # saving the observation

            #Taking a step w/ the nn
            act = cnn.get_action(network, torch.as_tensor(obs, dtype=torch.float))
            out = network(torch.as_tensor(obs, dtype=torch.float))

            obs, rew, done, _ = env.step(act)

            #Saving the action and the loss and the rew
            batch_acts.append(act)
            batch_loss.append(out)
            batch_rew.append(rew)

            if done:

                #Reset the vars
                obs, done = env.reset(), False

                if len(batch_obs) > batch_size:
                    break

    return batch_obs, batch_acts, batch_loss

"""
Creates a rl_network from cnn, and stores the networks in ./networks
Inputs:
    path: name of file that we want to store networks in. Will be stored as ./networks/path[int].pth
    episodes: how many iterations we will run

Outputs:
    all_batch_obs: the observations from all of the networks
    all_batch_acts: the actions from all of the networks
    all_batch_loss: the loss from all of the networks
 """
def run_rl_net_check_points(episodes, check_points, path='nets', batch_size=256):

    #Creating a directory to hold the network that we have, I might have to change this if we want to save multiple networks
    os.mkdir("./network") if not os.path.isdir('./network') else None
    new_path = "./network/" + path  # Making it so that we are putting the files that we are creating in the network folder

    #Creating the network and running it with the checkpoints
    trained_nets = cnn.train(epis=episodes, save_checkpoints=True, check_paths=new_path, check_points=check_points)
    print('\n ran the neural networks. Networks have been stored. files = ', trained_nets)

    #Storing batch_obs, batch_acts, and batch_loss
    all_batch_obs, all_batch_acts, all_batch_loss = [], [], []

    for net_index in range(len(trained_nets)):

        #Loading in the network from the file
        curr_net = cnn.Net()
        cp = torch.load(trained_nets[net_index])
        curr_net.load_state_dict(cp['model_state_dict'])
        curr_net.eval()

        #Running the net, Saving the next states the network took for expert data
        batch_obs, batch_acts, batch_loss = run_RL_net(curr_net, 10)
        all_batch_obs += batch_obs
        all_batch_acts += batch_acts
        all_batch_loss += batch_loss

    return all_batch_obs, all_batch_acts, all_batch_loss

"""
Training the behavioral cloning network on the expert data. Observations for the states, loss for the crit functions.
Inputs:
    expert_obs: the expert observations that we will be training the bc on
    expert_acts: The expert acts that we will be training the bc on
    expert_loss: The expert loss that wew ill be training the bc on
"""
test_loss = []
def train_bc(expert_obs, expert_loss, batch_size=256, network_path=''):
    loss_list = []  # A list of the loss on each step
    counter = 0 # Index for expert_obs and expert_loss
    # print(batch_size)

    while counter < len(expert_obs):
        total_loss = 0
        val = 0 # used for calculating the loss at the current step

        for _ in range(batch_size):

            if (counter == len(expert_obs)):
                break

            #Updating our bc_net, comparing our output to the rl_net
            curr_obs = expert_obs[counter]   # current observation
            exp = expert_loss[counter]  # what the expert did in this state

            bc_net = Net()

            if (os.path.isfile(network_path) and not is_checking):
                print('previous BC net is found, loading in')
                bc_net.load_state_dict(torch.load(bc_net))

                print('could not load in previous bc_net')

            out = bc_net(torch.as_tensor(curr_obs, dtype=torch.float))  # what we did in this state
            loss = crit(out, exp)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            val += 1
            counter += 1

        loss_list.append(total_loss / val)
        # x = expert_obs[:observation_space_size]
        # y = expert_obs[observation_space_size:]
        #
        # z = expert_loss[:, :observation_space_size]
        # output = bc_net(torch.as_tensor(x, dtype=torch.float))
        # test_loss.append(crit(output, y).item())
        print("step ", counter, " of bc. Loss for this step = ", (total_loss / val))
    return loss_list

#Printing the loss for the bc_net, with the loss for the rl_net
def plot(expert_loss, test_loss, ep_rew):

    plt.plot(np.array(expert_loss), label="Expert Loss")
    # plt.plot(test_loss, label="Testing Loss")
    plt.plot(ep_rew, label="Episode Reward")
    plt.plot()
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

""" One episode of the network after being trained """
def test_bc(batch_size=25):

    #Setting episode specific vars
    obs = env.reset()
    done = False
    finished = False
    counter = 0
    ep_rew = []
    theRews = []

    #Running simulation
    while True:
        if (not finished):
            env.render()
        else:
            finished = False

        act = Categorical(logits=bc_net(torch.as_tensor(obs, dtype=torch.float)))
        act = act.sample().item()
        obs, rew, done, _ = env.step(act)
        ep_rew.append(rew)
        # print(rew)

        if (done):
            finished = True
            ep_len = len(ep_rew)
            theRews.append(ep_len)
            obs, done, ep_rew = env.reset(), False, []

            if (counter > batch_size):
                break
            else:
                print("finished episode ", counter, "episode length was ", ep_len)
                counter += 1
    return theRews

""" Trainnig the behavioral cloning network """
def train(checkpoint=False):
    # Obtaining the expert data
    if not checkpoint:
        print("Getting expert data from rl_net without checkpoints")
        expert_obs, expert_acts, expert_loss = run_RL_net(rl_net, 60)
    else:
        print("Getting expert data from rl_net with checkpoints")
        expert_obs, expert_acts, expert_loss = run_rl_net_check_points(episodes=20, check_points=3,  path='first_nets')

    #Training the bc_net based on the expert data
    print("training bc_net from rl_net.")
    bc_loss = train_bc(expert_obs, expert_loss)
    print("finished training bc_net.")

    #Testing the bc network
    print("now testing bc.")

    ep_rew = test_bc()
    print("finished tesitng bc")
    # torch.save(bc_net.state_dict(), bc_net_path)

    print(bc_loss, "\n")
    # print(test_loss, "\n")
    print(ep_rew, "\n")
    plot(bc_loss, test_loss, ep_rew)

if __name__ == "__main__":
    train(checkpoint=True) if is_checking else train()

    torch.save(rl_net.state_dict(), rl_net_path)
env.close()
