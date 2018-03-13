from main import *

if __name__ == '__main__':
    agent = Agent("PongDeterministic-v4")

    input("Which episode should be used for the network?")
    agent.load_agent("agent_episode_{}.pth".format(1))
    agent.play(1)
    agent.env.close()
