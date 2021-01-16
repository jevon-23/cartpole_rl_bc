#https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import gym
import os.path


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
        self.fc1 = nn.Linear(self.num_obs, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
path = './cnet.pth'
if (os.path.isfile(path)):
    print('loaded prev neunet')
    net.load_state_dict(torch.load(path))

#Binary Cross Entropy Loss function; heard that BCE is the best for choosing between 2 diff actions
#Adams optimizer, good for noisy problems
crit = nn.BCELoss()
opt = optim.Adam(net.parameters(), lr=.001)

#Getting the neural net, making it type categorical because those are used in discrete action spaces
def get_policy(obs):
    return Categorical(logits=net(obs))

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weight):
    #via: https://stackoverflow.com/questions/54635355/what-does-log-prob-do, I think this only available for Categorical
    #log_prob returns the log of the probability density/mass function evaluated at the given sample value.
    logprob = get_policy(obs).log_prob(act)
    return -(logprob * weight).mean()

def train_one(batch_size=5000):
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    #Reset episode specific variables, all return values of step()
    obs = env.reset()   # First obs, gets updated as we go on
    done = False    # Checking to see if we are done
    ep_rew = [] # list of rew throughout episode
    finished = False    # finished this episode

    while True:
        if (not finished):
            env.render()    # Render env
        batch_obs.append(obs.copy())    # saving the observation

        #Taking a step w/ the nn
        act = get_action(torch.as_tensor(obs, dtype=torch.float))
        obs, rew, done, _ = env.step(act)

        #Saving the action and the reward
        batch_acts.append(act)
        ep_rew.append(rew)

        if done:
            #Saving the details about the episode
            ep_ret, ep_len = sum(ep_rew), len(ep_rew)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len
            finished = True   # finished this episode

            #Reset the vars
            obs, done, ep_rew = env.reset(), False, []

            if len(batch_obs) > batch_size:
                break

    opt.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weight=torch.as_tensor(batch_weights, dtype=torch.float32)
    )
    batch_loss.backward()
    opt.step()
    return batch_loss, batch_rets, batch_lens

def train(epis=450):
    for i in range(epis):
        batch_loss, batch_rets, batch_lens = train_one()
        print('episode: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
            (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        print(batch_lens[len(batch_lens) - 1])
if __name__ == "__main__":
    train()
    torch.save(net.state_dict(), path)
env.close()
