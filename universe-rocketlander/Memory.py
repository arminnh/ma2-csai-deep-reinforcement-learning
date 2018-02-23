from collections import namedtuple
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from PIL import Image
import torch

use_cuda = torch.cuda.is_available()
torch.cuda.set_device(1)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory(object):

    def __init__(self, capacity, frame_hist_length):
        self.capacity = capacity
        #self.memory = []
        self.frame_hist_length = frame_hist_length
        self.position = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.resize = T.Compose([T.ToPILImage(),
                    T.Resize((84,84), interpolation=Image.CUBIC),
                    T.Grayscale(),
                    T.ToTensor()])
        self.current_frames = 0

    def _modify_screen(self,frame):
        screen = frame.transpose((2, 0, 1))

        # convert to grayscale

        def rgb2gray(rgb):
            #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            return 0.2989 * r + 0.5870 * g + 0.1140 * b

        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize
        return self.resize(screen)

    def store_frame(self, frame):
        frame = self._modify_screen(frame)

        if len(self.states) == 0:
            self.states = np.zeros([self.capacity] + list(frame.shape))
            self.actions = np.zeros([self.capacity])
            self.rewards = np.zeros([self.capacity])
            self.done = np.zeros([self.capacity])

        self.states[self.position] = frame
        cur_pos = self.position
        self.current_frames = min(self.current_frames + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity
        return cur_pos

    def store_transition(self, cur_frame_idx, action, reward, done):
        self.actions[cur_frame_idx] = action
        self.done[cur_frame_idx] = 1 if done else 0
        self.rewards[cur_frame_idx] = reward

    def sample(self, batch_size):
        selected_idxs = random.sample(range(self.current_frames), batch_size)

        current_states = []
        actions = self.actions[selected_idxs]
        rewards = self.rewards[selected_idxs]
        done = self.done[selected_idxs]
        next_states = []
        for ids in selected_idxs:
            current_states.append(np.expand_dims(self.get_recent_history(ids),0))
            next_states.append(np.expand_dims(self.get_recent_history(ids+1),0))

        current_states = np.concatenate(current_states, 0)
        next_states = np.concatenate(next_states, 0)

        return current_states, actions, rewards, next_states, done

    def __len__(self):
        return self.current_frames

    def get_recent_history(self, position=None):
        # get the last frame hist length frames
        # to give it to the network
        if position == None:
            position = self.position

        start_hist = position - self.frame_hist_length

        if len(self) == self.capacity and start_hist < 0:
            # + since start_hist is already neg
            start_hist = self.capacity + start_hist
        elif len(self) < self.capacity and start_hist < 0:
            # not enough frames
            start_hist = 0

        # Mhh what about frames where we are at the end of the episode??
        # we should skip those!?
        new_start = start_hist
        for j in range(self.frame_hist_length):

            if self.done[(start_hist + j) % self.capacity]:
                new_start += 1

        start_hist = (new_start % self.capacity)
        # How many frames have we left?
        missing = self.frame_hist_length - (position - start_hist)


        frames = []

        # First append the missing ones
        for j in range(missing):
            frames.append(np.zeros_like(self.states[0]))

        for j in range(self.frame_hist_length - missing):
            frames.append(self.states[(start_hist + j) % self.capacity])

        return np.concatenate(frames, 0)



