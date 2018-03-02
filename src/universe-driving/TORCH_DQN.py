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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(9152, actions_size)

    def forward(self, x):
        if x.dim() == 3:
            # Make x 4d
            #x =xx  x.resize(1,128,200)
            x = x[None, :, :, :]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQN:

    def __init__(self, action_size, GPU=torch.cuda.is_available(), GPU_id=0):
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gpu = GPU
        self.gpu_id = GPU_id
        # Torch
        self.model = DQNNetwork(action_size).cuda(GPU_id)
        #if GPU:
        #    self.model = self.model.cuda(device=GPU_id)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((torch.from_numpy(state), action, reward, torch.from_numpy(next_state), done))

    def fit(self, state, target):
        if self.gpu:
            state_var = Variable(state).cuda(device=self.gpu_id)
            target_var = Variable(target).cuda(device=self.gpu_id)
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

        outputs = self.model(Variable(torch.from_numpy(state)).cuda(self.gpu_id))
        #_, predicted = torch.max(outputs.data, 1)

        #act_values = self.model.predict(state)

        return np.argmax(outputs.data[0].cpu().numpy())  # returns action

    def replay(self, batch_size):
        # If we don't have enough things in memory just return
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        state_batch = []
        target_batch = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                if self.gpu:
                    next_state = Variable(next_state).cuda()
                else:
                    next_state = Variable(next_state)

                #print(self.model(next_state))
                target = (reward + self.gamma *
                          np.amax(self.model(next_state)[0].data.cpu().numpy()))

            if self.gpu:
                state = Variable(state).cuda()
            else:
                state = Variable(state)

            target_f = self.model(state).data.cpu()
            target_f[0][action] = target
            state_batch.append(state.data.cpu())
            target_batch.append(target_f)

        # Send it all like a full batch
        state_batch = torch.stack(state_batch)
        target_batch = torch.stack(target_batch)
        #print("state")
        #print(state_batch)
        #print("target")
        #print(target_batch)
        self.fit(state_batch, target_batch)

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
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, name)
