#https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
import os.path
import sys


#making the env
env = gym.make("CartPole-v0")
action_space_size = env.action_space.n  # Discrete 2
observation_space_size = env.observation_space.shape[0] # Box (4, )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
2 layered network.
"""
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
# valp = len(sys.argv) == 2
# if (os.path.isfile(path) and __name__ == 'main'):
#     print('loaded prev neunet')
#     net.load_state_dict(torch.load(path))

#Binary Cross Entropy Loss function; heard that BCE is the best for choosing between 2 diff actions
#Adams optimizer, good for noisy problems
crit = nn.BCELoss()
opt = optim.Adam(net.parameters(), lr=.001)

"""
Getting the neural net, making it type categorical because those are used in discrete action spaces.
"""
def get_policy(network, obs):

    return Categorical(logits=network(obs))

"""
Make action selection function (outputs int actions, sampled from policy.
"""
def get_action(network, obs):
    return get_policy(network, obs).sample().item()

"""
Compute the loss for the current step.
"""
def compute_loss(network, obs, act, weight):
    #via: https://stackoverflow.com/questions/54635355/what-does-log-prob-do, I think this only available for Categorical
    #log_prob returns the log of the probability density/mass function evaluated at the given sample value.
    logprob = get_policy(network, obs).log_prob(act)
    return -(logprob * weight).mean()

"""
Trains one step of the network
Outputs:
    batch_loss: The losses that were computed for the episode
    batch_rets: The sum of the rewards that were produced in the episode
    batch_lens: The length of the episode
"""
def train_one(network, batch_size=5000):
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    #Reset episode specific variables, all return values of step()
    obs = env.reset()   # First obs, gets updated as we go on
    done = False    # Checking to see if we are done
    ep_rew = [] # list of rew throughout episode
    finished = False    # finished this episode

    while True:
        if (not finished): # and not valp):
            env.render()    # Render env

        batch_obs.append(obs.copy())    # saving the observation

        #Taking a step w/ the nn
        act = get_action(network, torch.as_tensor(obs, dtype=torch.float))
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
    opt = optim.Adam(network.parameters(), lr=.001)
    opt.zero_grad()
    batch_loss = compute_loss(network=network, obs=torch.as_tensor(batch_obs, dtype=torch.float32),
        act=torch.as_tensor(batch_acts, dtype=torch.int32),
        weight=torch.as_tensor(batch_weights, dtype=torch.float32)
    )
    batch_loss.backward()
    opt.step()
    return batch_loss, batch_rets, batch_lens

""" Makes check_points
Inputs:
    EPIS = Number of episodes to be ran for the training
    check_points = how many checkpoints
Outputs:
    rv = List of numbers that are evenly divided based on epis, -1 for zero index,
        -Ex: epis = 21 check_points = 3 -> rv = [6, 13, 20]
"""

def make_check_points(epis, check_points):

    # rv = []
    # counter = 1
    # for i in range(check_points):
    #     rv.append(counter * (epis // check_points) - 1)
    #     counter += 1
    # return rv

    # bad = 10 - check_points  # what bad episodes we will be saving
    bad = check_points
    # good = epis - check_points

    good = epis - 10 + check_points
    # what good episodes we will be saving
    rv = []

    #Getting bad checkpoints
    for x in range(bad):
        rv.append(x)

    #Getting good checkpoints
    for x in range(good, epis):
        rv.append(x)

    return rv

"""
Trains the network.
Inputs:
    epis: Number of episodes that we will learn from
    save_checkpoints: True if we want to create checkpoints, false otherwise
    check_paths: the name of the path that we want to name the checkpoints after
    checkpoints: How many checkpoints we want to create, only set if save_checkpoints is true
Outputs:
    files: a list of the files that were created from the process of the training. Is empty unless save_checkpoints is true
"""
def train(epis, save_checkpoints, check_paths, check_points):
    counter = 0 # Used for the file that we will store the checkpoints in
    network = Net()
    print(epis, check_points)
    check_points = make_check_points(epis, check_points) if save_checkpoints else None
    print(check_points)
    files = []  # The files where the checkpoints are saved

    for i in range(epis):

        batch_loss, batch_rets, batch_lens = train_one(network)    # Train network

        #If save_checkpoints is true, that meanst that the other parameters are set
        if ((save_checkpoints) and (i in check_points)):
            cp_place = check_paths + str(counter) + '.pth'
            try:
                torch.save({
        'model_state_dict': net.state_dict(),
        }, cp_place)
                counter += 1
                print('path created at ', cp_place)
                files.append(cp_place)

            except RuntimeError:
                print('could not save network @ ', i, 'pathName = ', cp_place)

        # print('episode: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
        #     (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        print('episode ', i, '= ', batch_lens[len(batch_lens) - 1])

    return files

if __name__ == "__main__":
    train(None, False, 0, 200)
    torch.save(net.state_dict(), path)
env.close()
