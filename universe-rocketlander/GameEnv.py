import gym
from Model import Model
from Memory import ReplayMemory
import copy
import random
import math
from itertools import count
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image
import torch

# if gpu is to be used
use_cuda = torch.cuda.is_available()
torch.cuda.set_device(1)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

episode_scores = []

import matplotlib
import matplotlib.pyplot as plt

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_scores)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig("score.jpg")
    plt.pause(0.001)  # pause a bit so that plots are updated

class GameEnv:

    def __init__(self, config=None):
        self.env = gym.make(config["game"])
        self.Q = Model(self.env.action_space.n).cuda(device=config["cuda_device"])
        self.Q_hat = copy.deepcopy(self.Q)
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=config["learning_rate"], momentum=config["grad_momentum"])
        self.memory = ReplayMemory(config["memory_size"], config["history_length"])
        self.learning_starts = config["replay_start"]
        self.m = 4#config["m"]
        self.eps = config["init_eps"]
        self.eps_start = config["init_eps"]
        self.eps_end = config["final_eps"]

        # After how many frames do we take ask for an action?
        self.k_th_frame = config["action_repeat"]
        self.update_freq = config["update_freq"]
        self.gamma = config["discount_factor"]
        self.decay = 200
        self.batch_size = config["batch"]
        self.C = config["target_update_freq"]

        self.am_param_updates = 0
        self.episodes = config["episodes"]
        self.final_expl_frame = config["final_expl_frame"]

    def select_action(self, state,loop_count):
        sample = random.random()
        # while we have less than learning_start frames perform random actions
        if self.learning_starts > loop_count:
            return LongTensor([[random.randrange(self.env.action_space.n)]])
        elif self.learning_starts == loop_count:
            print("STARTING TO LEARN NOW!!")

        eps_threshold = self.eps_start - min(float(loop_count)/self.final_expl_frame, 1) * (self.eps_end - self.eps_start)

        if sample > eps_threshold:
            return self.Q(
                Variable(torch.from_numpy(state), volatile=True).type(FloatTensor).unsqueeze(0)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.env.action_space.n)]])

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        #transitions = self.memory.sample(self.batch_size)
        #batch = Transition(*zip(*transitions))

        current_states, actions, rewards, next_states, done = self.memory.sample(self.batch_size)


        # Compute a mask of non-final states and concatenate the batch elements
        #non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # create our batches

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states_batch = Variable(torch.from_numpy(next_states), volatile=True).type(Tensor).cuda()
        state_batch = Variable(torch.from_numpy(current_states)).type(Tensor).cuda()
        action_batch = Variable(torch.from_numpy(actions)).type(LongTensor)
        reward_batch = Variable(torch.from_numpy(rewards)).type(Tensor)

        non_final_mask = Variable(torch.from_numpy(1 - done)).type(Tensor)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        current_state_action_values = self.Q(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        #next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        next_state_values = non_final_mask * self.Q_hat(non_final_next_states_batch).detach().max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute Huber loss
        loss = torch.nn.MSELoss()
        output = loss(current_state_action_values, expected_state_action_values.view(-1,1))
        output.clamp(-1,1)

        # Clear previous gradients before backward pass
        # run backward pass

        # Optimize the model
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()
        self.am_param_updates += 1

        if self.am_param_updates % self.C == 0:
            print("updating q hat")
            self.Q_hat.load_state_dict(self.Q.state_dict())

    def run(self):
        loops_count = 0
        actions_selected = 0
        for i in range(self.episodes):
            #Initialize sequence s 1 ~ f x 1 g and preprocessed sequence w 1 ~w ð s 1 Þ
            obs = self.env.reset()

            cum_reward = 0.0
            for t in count():
                # get current screen and store in memory
                last_screen_idx = self.memory.store_frame(obs)
                if t % self.k_th_frame == 0: 
                        # every k'th frame take new action
                        last_states = self.memory.get_recent_history(last_screen_idx)
                        action = self.select_action(last_states, loops_count)
                        actions_selected += 1

                obs, reward, done, _ = self.env.step(action[0,0])

                reward = torch.clamp(Tensor([reward]),-1,1)
                cum_reward += reward[0]
                self.memory.store_transition(last_screen_idx, action[0,0], reward[0],done)

                #self.memory.push(state.cpu(), action.cpu(), next_state.cpu() if next_state is not None else next_state, reward.cpu())
                if loops_count > self.learning_starts and actions_selected % self.update_freq == 0:
                    self.train_model()


                # for obj in gc.get_objects():
                #     if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #         print(type(obj), obj.size())
                loops_count+=1
                if done:
                    episode_scores.append(cum_reward)
                    plot_durations()
                    break
