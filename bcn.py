import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os.path
import cnn
import gym
from torch.distributions.categorical import Categorical


#making the env
env = gym.make("CartPole-v0")
action_space_size = env.action_space.n  # Discrete 2
observation_space_size = env.observation_space.shape[0] # Box (4, )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

if (os.path.isfile(bc_net_path)):
    print('previous BC net is found, loading in')
    bc_net.load_state_dict(torch.load(bc_net))
#Loading in the RL net
rl_net_path = './cnet.pth'
rl_net = cnn.Net()

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
    net: rl_net, the net that the behavioral cloning net will be learning from
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
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_loss = []

    #Reset episode specific variables, all return values of step()
    obs = env.reset()   # First obs, gets updated as we go on
    done = False    # Checking to see if we are done
    ep_rew = [] # list of rew throughout episode
    finished = False    # finished this episode
    for counter in range(episodes):
        while True:
            # if (not finished):
            #     env.render()    # Render env
            batch_obs.append(obs.copy())    # saving the observation

            #Taking a step w/ the nn
            act = cnn.get_action(torch.as_tensor(obs, dtype=torch.float))

            batch_loss.append(network(torch.as_tensor(obs, dtype=torch.float)))
            obs, rew, done, _ = env.step(act)

            #Saving the action and the reward
            batch_acts.append(act)
            ep_rew.append(rew)

            if done:
                #Saving the details about the episode
                ep_ret, ep_len = sum(ep_rew), len(ep_rew)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                finished = True   # finished this episode

                #Reset the vars
                obs, done, ep_rew = env.reset(), False, []

                if len(batch_obs) > batch_size:
                    break

        b_loss = cnn.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weight=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        # batch_loss.append(b_loss)
        # print("finished episode ", counter, "of rl_net")
    # env.reset()
    return batch_obs, batch_acts, batch_loss

def train_bc(expert_obs, expert_acts, expert_loss, batch_size=256):
    loss_list = []
    test_loss = []

    counter = 0

    while counter < len(expert_obs):
        total_loss = 0
        val = 0
        for _ in range(batch_size):
            # if (expert_obs[obs_ind] == 50):
            if (counter == len(expert_obs)):
                break

            #Updating our bc_net, comparing our output to the rl_net
            curr_obs = expert_obs[counter]   # current observation
            exp = expert_loss[counter]  # what the expert did in this state
            out = bc_net(torch.as_tensor(curr_obs, dtype=torch.float))  # what we did in this state

            # print("out = ", out, "exp = ", exp)
            loss = crit(out, exp)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            val += 1
            counter += 1

        loss_list.append(total_loss / val)
        print("step ", counter, " of bc. Loss for this step = ", (total_loss / val))

        x = expert_obs[observation_space_size]
        y = expert_loss[observation_space_size]

        # out = bc_net(x)
        # test_loss.append(crit(out, y).item())
""" One episode of the network after being trained """
def test_bc(batch_size=100):
    obs = env.reset()
    done = False
    finished = False
    counter = 0
    ep_rew = []
    # batch_obs = []
    while True:
        if (not finished):
            env.render()
        else:
            finished = False

        act = Categorical(logits=bc_net(torch.as_tensor(obs, dtype=torch.float)))
        act = act.sample().item()
        obs, rew, done, _ = env.step(act)
        ep_rew.append(rew)

        if (done):
            finished = True
            ep_len = len(ep_rew)
            obs, done, ep_rew = env.reset(), False, []

            if (counter > batch_size):
                break
            else:
                print("finished episode ", counter, "episode length was ", ep_len)
                counter += 1


""" Trainnig the behavioral cloning network """
def train(epis=2):

    print("running rl_net for expert data.")
    expert_obs, expert_acts, expert_loss = run_RL_net(rl_net, 60)
    print("training bc_net from rl_net.")

    train_bc(expert_obs, expert_acts, expert_loss)
    print("finished training bc_net, now testing bc.")

    test_bc()
    print("finished tesitng bc")
    torch.save(bc_net.state_dict(), bc_net_path)

if __name__ == "__main__":
    train()

    # torch.save(rl_net.state_dict(), rl_net_path)
env.close()
