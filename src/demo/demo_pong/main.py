import gym
import random
import numpy as np
from itertools import count
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.autograd import Variable

from atari_wrappers import WarpFrame, FrameStack, ClipRewardEnv


use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# Experience replay memory
class ExperienceMemory:
    def __init__(self, n):
        self.memory = deque(maxlen=n)

    def add_transition(self, s, a, r, next_s, done):
        self.memory.append((s, a, r, next_s, done))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        # https://stackoverflow.com/questions/40181284/how-to-get-random-sample-from-deque-in-python-3
        # Since python3.5 you can just do a random sample on a deque with a size
        sample_batch = random.sample(self.memory, batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for s in sample_batch:
            state_batch.append(s[0])
            action_batch.append(s[1])
            reward_batch.append(s[2])
            next_state_batch.append(s[3])
            done_batch.append(s[4])

        return np.asarray(state_batch), np.asarray(action_batch), np.asarray(reward_batch), \
               np.asarray(next_state_batch), np.asarray(done_batch)


# Environment according to deepmind's paper "Human Level Control Through Deep Reinforcement Learning"
def deepmind_env(env_id, m=4):
    env = gym.make(env_id)

    # Wrap the frames to 84x84 and grayscale
    env = WarpFrame(env)

    # Stack the 4 most recent frames
    env = FrameStack(env, m)

    # Clip rewards to -1 and 1
    env = ClipRewardEnv(env)

    return env


# The neural network
class Model(nn.Module):
    def __init__(self, possible_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, possible_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class Agent:
    def __init__(self, game_id):
        # initialize the game environment
        self.env = deepmind_env(game_id)

        # Init Q
        self.Q = Model(self.env.action_space.n)

        # Init target Q with the same weights as self.Q
        self.target_Q = Model(self.env.action_space.n)
        self.sync_target_q()

        if use_cuda:
            self.Q.cuda()
            self.target_Q.cuda()

        self.memory = ExperienceMemory(1000000)
        self.gamma = 0.99

        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=0.0001)

    def sync_target_q(self):
        # Syncs the Q target with the target Q function
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/5
        copy_from = list(self.Q.parameters())
        copy_to = list(self.target_Q.parameters())
        n = len(copy_from)
        for i in range(0, n):
            copy_to[i].data[:] = copy_from[i].data[:]

    def get_eps(self, current_steps, max_exploration, start_eps, end_eps):
        # Gets the current epsilon value
        # linearly decline
        return max(end_eps, start_eps - current_steps / max_exploration)

    def get_action(self, current_eps, states):
        # Get an action based on the current eps and the state
        if random.random() > current_eps:

            # Our states are 84 x 84 x 4 but pytorch expects a 4D tensor
            # so we add an extra dimension
            states = np.expand_dims(states, 0)
            actions = self.Q(Variable(torch.from_numpy(states)).type(FloatTensor))
            return np.argmax(actions.data.cpu().numpy())
        else:
            return LongTensor([[random.randrange(self.env.action_space.n)]])

    def get_yi(self, next_states, rewards, done):
        q_target_vals = self.target_Q(Variable(torch.from_numpy(next_states)).type(FloatTensor))

        # We get a batch size x 1 tensor back
        # We want the values from the last dimension
        q_target_vals = np.max(q_target_vals.data.cpu().numpy(), axis=1)

        # For every state that is done, set Q to zero
        mask = (done == 1)
        q_target_vals[mask] = 0

        yi = rewards + self.gamma * q_target_vals
        return Variable(torch.from_numpy(yi)).type(FloatTensor)

    def update_weights(self, batch_size):
        if len(self.memory) < batch_size:
            return

        # get a random minibatch of transitions
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(batch_size)

        # Get our yi's
        yi = self.get_yi(next_state_batch, reward_batch, done_batch)

        # Now we need to get our normal q values
        q_values = self.Q(Variable(torch.from_numpy(state_batch).type(FloatTensor)))

        # Now select the actions we took
        actions_taken = torch.gather(q_values, 1,
                                     Variable(torch.from_numpy(action_batch)).type(LongTensor).view(-1, 1))

        loss = self.loss(actions_taken, yi)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_agent(self, file):
        self.Q.load_state_dict(torch.load(file))
        self.sync_target_q()

    def save_agent(self, episode):
        if not os.path.exists("saved_model/"):
            os.makedirs("saved_model/")

        torch.save(self.Q.state_dict(), "saved_model/agent_episode_{}.pth".format(episode))

    def play(self, episodes):
        for episode in range(1, episodes+1):
            state = self.env.reset()
            for _ in count(start=1):
                action = self.get_action(0, state)
                state, reward, done, _ = self.env.step(action)
                self.env.render()
                if done:
                    break

    def train(self, episodes, sync_target=10000, max_eploration=10**5, end_eps=0.1, start_eps=1, batch_size=32):
        steps = 0
        self.save_agent(0)
        for episode in range(1, episodes + 1):
            state = self.env.reset()

            current_reward = 0
            for t in count(start=1):
                # select action with prob eps
                current_eps = self.get_eps(steps, max_eploration, start_eps, end_eps)
                action = self.get_action(current_eps, state)
                # execute action in emulator
                next_state, reward, done, _ = self.env.step(action)
                # Add this to our memory
                self.memory.add_transition(state, action, reward, next_state, done)

                # Update our weights now
                self.update_weights(batch_size)

                steps += 1
                current_reward += reward
                state = next_state
                # every C steps we reset target Q
                if (steps % sync_target) == 0:
                    print("Sync target network")
                    self.sync_target_q()

                if done:
                    break

            print("Episode: {} finished".format(episode))
            # information stuff
            if (episode % 10) == 0:
                print("--- Saving episode {} ---".format(episode))
                self.save_agent(episode)
                print("Episode reward: {}".format(current_reward))
                print("Eps: {}".format(current_eps))


if __name__ == '__main__':
    agent = Agent("Freeway-v0")
    #agent.train(1000)

    agent.load_agent("saved_model/agent_episode_{}.pth".format(0))
    agent.play(1)
    #agent.env.close()
