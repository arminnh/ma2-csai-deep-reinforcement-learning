#https://github.com/openai/universe-starter-agent/blob/master/envs.py

import gym
import universe
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from TORCH_DQN import DQN
from enum import Enum
import torchvision.transforms as T
import ast
import torch
from env import create_flash_env

class Moves(Enum):
    LEFT = 0
    RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3
    TURBO = 4

    def __str__(self):
        if self == Moves.ACCELERATE:
            return "up"
        elif self == Moves.BRAKE:
            return "down"
        elif self == Moves.LEFT:
            return "left"
        elif self == Moves.RIGHT:
            return "right"
        elif self == Moves.TURBO:
            return "x"

class SelfDrivingAgent:

    def __init__(self):
        #4 moves
        self.DQN = DQN(len(Moves))

        self.state = None
        self.lastScreen = None

def main():
    # Create env
    env, w, h  = create_flash_env('flashgames.DuskDrive-v0')
    _ = env.reset()

    agent = SelfDrivingAgent()
    #print(observation_n)
    agent.state = torch.zeros((1,128,200)).numpy()
    agent.lastScreen = torch.zeros((1,128,200)).numpy()

    next_state = torch.zeros((1,128,200)).numpy()
    count = 1
    while True:
        action = agent.DQN.act(agent.state)

        observation_n, reward_n, done_n, info = env.step(action)
        if "global/episode_reward" in info:
            count += 1
            # we have finished an episode
            if count in [100,200,300,400,500,600,700,800,900] or count % 1000 == 0:
                #save
                agent.DQN.save("agent_ep{}".format(count))

        #print("learning")
        agent.DQN.remember(agent.state, action, reward_n, next_state, False)
        #print(observation_n)
        next_state = observation_n - agent.lastScreen
        agent.lastScreen = observation_n

        agent.state = next_state
        agent.DQN.replay(128)

        env.render()

main()