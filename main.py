import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import cnn
import bcn
import sys
import re
from matplotlib import pyplot as plt


"""
Making check_points for the graph over time. At the moment only made for 10 check_points, ill change it lager, taken from cnn
Inputs:
    epis: number of episodes
    good_checks: how many good checkpoints do we want to save
"""
def make_check_points2(epis, good_checks):
    bad = good_checks
    # good = epis - check_points

    good = epis - 10 + good_checks

    # what good episodes we will be saving
    rv = []

    #Getting bad checkpoints
    for x in range(bad):
        rv.append(x)

    #Getting good checkpoints
    for x in range(good, epis):
        rv.append(x)

    return rv

#Printing the loss for the bc_net, with the loss for the rl_net
def plot(expert_loss, ep_rew):
    for loss_ind in range(len(expert_loss)):

        plt.plot(np.array(expert_loss[loss_ind]), label="Expert Loss" + str(loss_ind))

    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

def plot2(ep_rew):
    plt.plot(ep_rew, label="Episode Reward")
    plt.xlabel("iterations")
    plt.ylabel("ep_rew")
    plt.legend()
    plt.show()

def processCLI():
    if (len(sys.argv) != 2):
        print('failed1')
        return -1

    # if (type(sys.argv[1]) != int):
    #     print('failed')
    #     return -1

    episodes = int(sys.argv[1])

    return episodes


def main():
    episodes = processCLI()
    bc_net_path = 'network0.pth'
    if (episodes == -1):
        print('usage: main.py { number of episodes }')
        return -1

    all_loss = [] # Storing the loss for the plot
    for good_checks in range(16):
        checkpoints = make_check_points2(episodes, good_checks)
        print('checkpoints for batch #', good_checks, checkpoints)

        print("Getting expert data from rl_net with checkpoints")
        expert_obs, expert_acts, expert_loss = bcn.run_rl_net_check_points(episodes=episodes, check_points=good_checks)

        #Training the bc_net based on the expert data
        print("training bc_net from rl_net.")
        bc_loss = bcn.train_bc(expert_obs, expert_loss, network_path=bc_net_path)
        print("finished training bc_net.")

        #Testing the bc network
        print("now testing bc.")

        ep_rew = bcn.test_bc(network_path=bc_net_path)
        print("finished tesitng bc")

        all_loss.append(ep_rew)

    plot(bc_loss, all_loss)
    plot2(all_loss)
# if (__name__ == 'main'):
print('hello')
main()
