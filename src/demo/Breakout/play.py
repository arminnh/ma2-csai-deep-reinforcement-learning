from main import *

if __name__ == '__main__':
    agent = Agent("Breakout-v4")

    episode = input("Which episode should be used for the network? ")
    agent.load_agent("agent_episode_{}.pth".format(int(episode)))
    agent.play(1)
    agent.env.close()
