import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import spaces
import os.path
import numpy as np
import math
import random

#Training Parameters
epis = 100
ticks = 195
max_env_steps = None

gamma = 1.0 # Discount factor
epsilon = 1.0   # Explore
epsilon_min = .01
epsilon_max = .995
epsilon_decay = .995
alpha = .01 # Learning Rate
a_decay = .01

batch_size = 64
monitor = False
quiet = False

#Creating enviroment, and getting the action & observation spaces
#action_spcace = Discrete(2)
#Observation_space = (4, )
env = gym.make("CartPole-v0")
mem = []
if max_env_steps is not None:
    env.max_episode_steps = max_env_steps


#Building the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_actions = env.action_space.n
        self.num_obs = env.observation_space.shape[0]
        self.fc1 = nn.Linear(self.num_obs, 256)
        self.fc2 = nn.Linear(self.num_obs, 1)

    def forward(self, x):
        F.relu(self.fc1(x))
        self.fc2(x)
        return x

net = Net()
# print(net)
path = './cpt_net.pth'

# if (os.path.isfile(path)):
#     print('prev file is found')
#     net.load_state_dict(torch.load(path))

#Binary Cross Entropy Loss function; heard that BCE is the best for choosing between 2 diff actions
#SGD optimizer; need to do more research on an optimizer other than SGD

loss_fn = nn.BCELoss()
# opt = optim.SGD(net.parameters(), lr=.001, momentum=0.9)
opt = optim.SGD(net.parameters(), lr=alpha, momentum=a_decay)

def remember(state, act, rew, next, done):
    mem.append((state, act, rew, next, done))

def choose_act(state, epsilon):
    #Not sure if this is right, i dont know what exactly is getting passed in as state here.
    #in the keras ex, he is using model.predict(), so i assume thats the same as calling the nn
    # print('state = ', state, 'eps = ', epsilon)
    if (np.random.random() <= epsilon):
        # print('if')
        return env.action_space.sample()
    else:
        p = net(torch.tensor(state, dtype=torch.float))
        opt.zero_grad()
        loss = loss_fn(p, None)
        loss.backward()
        opt.step()
        p = np.clip(p.detach().numpy(), -1, 2)
        # print(p[0][0:2], np.argmax(p[0][0:2]))
        # print(p, 'vs', p.numpy())
        # print('np.argmax = ', np.ndarray.argmax(p.numpy()))
        return np.argmax(p[0][0:2])

    # return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(net(torch.tensor(state, dtype=torch.float)).numpy())

def get_eps(t):
    #look in to this
    return max(epsilon_min, min(epsilon, 1.0 - math.log10(t+1)*epsilon_decay))

def preprocess_state(state):
    return np.reshape(state, [1, 4])

def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibactch = random.sample(mem, min(len(mem), batch_size))

    for state, act, rew, next, done in minibactch:
        y_targ = net(torch.tensor(state, dtype=torch.float))
        # print(torch.tensor(next, dtype=torch.float)[0].dim())
        # if (done):
        #     print('in done')
        #     y_targ[0][act] = rew
        # else:
        #     print('in else', next)
        #     y_targ[0][act] = rew + gamma + np.max(net(torch.tensor(next, dtype=torch.float)).numpy())
        # y_targ[0][act] = rew if done else rew + gamma * np.max(net(torch.tensor(next, dtype=torch.float))[0])
        y_targ[0][act] = rew if done else (rew + gamma * np.max(net(torch.tensor(next, dtype=torch.float)).numpy()))
        x_batch.append(state[0])
        y_batch.append(y_targ[0])

    #model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
    if (epsilon > epsilon_min):
        epsilon *= epsilon_decay

#run func
def run():
    scores = []
    for e in range(epis):
        state = preprocess_state(env.reset())
        done = False
        i = 0
        while not done:

            # Currently the act that we are getting back from this ends up being > 1, which is outside of our range of possibilities
            act = choose_act(state, get_eps(e))
            # print('finished choosing, act = ', act)
            # print('act = ', act)
            next, rew, done, _ = env.step(act)
            # print('finished stepping, next act = ', next, 'rew = ', rew)
            env.render()
            next = preprocess_state(next)
            remember(state, act, rew, next, done)
            state = next
            i += 1
        # print('out of loop')

        scores.append(i)
        mean_score = np.mean(scores)

        if (mean_score >= ticks and e > 100):
            if not quiet:
                print('Ran ', e, 'episodes. Solved after ', e-100, 'trials')
                return e - 100
        if e % 20 == 0 and not quiet:
            print('episode ', e, 'mean survival after last dub was', mean_score)

        replay(batch_size, get_eps(e))
    if (not quiet):
        print('did not finish after ', e)
        return e
run()

#Learn

# for _ in range(5):
#     tloss = 0.0
#     rEnv = env.reset()
#     d = False
#     counter = 0
#     while not d:
#         env.render()
#         opt.zero_grad()
#         act = env.action_space.sample()
#         obs, rew, d, info = env.step(act)
#         print(act)
#         out = net(torch.tensor(rEnv, dtype=torch.float))
#         print(out)
#         # env.step(rEnv)
#
#         if (d):
#             print('finished step, rew = ', rew, 'obs = ', obs, 'count = ', counter)
#         counter += 1
#
#Saving the data that we learned
torch.save(net.state_dict(), path)
env.close()
