# code based on https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym
import gym
import numpy as np

# Environment for a simple text game
env = gym.make("Taxi-v2")
# initialize environment
env.reset()
env.render()

print("observation space", env.observation_space.n)
print("action space", env.action_space.n)

# Random agent
# state = env.reset()
# counter = 0
# reward = None
# while reward != 20:
#     state, reward, done, info = env.step(env.action_space.sample())
#     counter += 1
# print("Random agent reward after", counter, "steps")

# Q table as a numpy array. A value for each pair of (state, action),
# so sizo of Q table size = |states|*|actions|
# If we were to create a Q table for atari games and use images as state, we would need
# to represent images of resolution 160 by 192 with let's say 32 different grayscale values
# => not feasible
Q = np.zeros([env.observation_space.n, env.action_space.n])

# learning rate
alpha = 0.7
episodes = 1000

# basic Q learning algorithm
for episode in range(episodes):
    done = False
    # sum of rewards for an episode
    R = 0
    reward = 0
    # reset the environment for the new episode
    state = env.reset()
    while not done:
        # Select the action that currently has the best Q value
        action = np.argmax(Q[state])
        # Take the action and observe the results
        state2, reward, done, info = env.step(action)
        # Update the (state, action) pairin the Q table (Bellman equation)
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state,action])

        R += reward
        state = state2

    if (episode+1) % 25 == 0:
        print("Episode:", episode+1, "\tReward:", R)

env.render()
