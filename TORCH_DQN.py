# based on https://github.com/keon/deep-q-learning/blob/master/dqn.py
from collections import deque
import numpy as np
import random
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

class DQNNetwork(nn.Module):
    def __init__(self, actions_size):
        super(DQNNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, actions_size)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

class DQN:

    def __init__(self, action_size, GPU=torch.cuda.is_available()):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gpu = GPU

        # Torch
        self.model = DQNNetwork(action_size)
        if GPU:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def fit(self, state, target):
        if self.gpu:
            state_var = Variable(state).cuda()
            target_var = Variable(target).cuda()
        else:
            state_var = Variable(state)
            target_var = Variable(target)

        prediction = self.model(state_var)
        loss = F.smooth_l1_loss(prediction, target_var)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Do random action
            return random.randrange(self.action_size)

        outputs = self.model(Variable(state))
        #_, predicted = torch.max(outputs.data, 1)

        #act_values = self.model.predict(state)

        return np.argmax(outputs.data[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch = []
        target_batch = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model(next_state)[0]))
            target_f = self.model(state)
            target_f[0][action] = target
            state_batch.append(state)
            target_batch.append(target_f)

        # Send it all like a full batch
        self.fit(torch.stack(state_batch), torch.stack(target_batch))

            #self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load checkpoint and map everything to cpu in case we trained on gpu
        checkpoint = torch.load(name, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save(self, name):
        state = {
            'state_dict': self.model.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, name)
