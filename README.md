#  Human-level control through deep reinforcement learning
> Project for the course [Capita Selecta Computer Science: Artificial Intelligence](https://onderwijsaanbod.kuleuven.be/syllabi/e/H05N0AE.htm#activetab=doelstellingen_idm1514848) at KU Leuven

## Components of the project

### Main goal
A presentation on the topic "Human-level control through deep reinforcement learning". For our demo, we implemented an agent that is able to learn to play a variety of (simple) games using deep reinforcement learning.

### Presentation (&plusmn;30 min):
1. Intro to the problem this project tries to solve
2. A refresher on reinforcement learning
3. An introduction to convolutional neural networks
4. Deep reinforcement learning
5. Demo!
6. Curious Reinforcment Learning, a short section on what may come after deep reinforcement learning.

### Implementation
The implementation of an agent that can successfully learn to play games can be found in [src/main.py](src/main.py).

#### Neural network
Our implementation closely follows Deepmind's Deep Q-Network.  
The deep neural network takes as input 4 video frames (grayscale with resolution 84x84) and returns probabilities for next actions to take. It consists of 3 convolutional layers and 2 fully connected layers with ReLUs in between. The network uses Experience Replay and a target network as described in Deepmind's DQN paper.

#### Virtual environment
For the agent to learn to play games, we needed a virtual environment.

##### Requirements
* Can host a variety of games
* Get observations from environment
* Make actions in environment
* Receive reward for action from environment
* Tracks to drive on

The environment we ended up using OpenAI's Gym.  
We originally started off with Udacity's Behavioral cloning project to create an agent for self driving cars. That path ended up being unfruitful, as we were not able to synchronise the data of the simulator with the separate neural network program correctly.  
We then opted for a simpler approach in order to stay within the allocated time for this project. OpenAI's Gym library allowed us to write a working program in much less time. It also gave us the possibility to try out our network on a variety of different games.  

## Dependencies
This project relies on the following python dependencies:

    numpy pytorch gym gym[atari]

## Usage
The agent can be trained as follows
```
python3 main.py
```
The agent will start learning and will output the achieved score and save its network's weights every 10 episodes.

The agent can afterwards be loaded and be used to play the game with
```
python3 play.py
```

## Built with
* [OpenAI Gym](https://github.com/openai/gym) - Enviroments to interact with games
* [OpenAI Universe](https://github.com/openai/universe) - Training and evaluating AI agents
* [PyTorch](https://github.com/pytorch/pytorch) - Deep neural networks engine with GPU acceleration
* [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) - Udacity's car simulation environment which allows for neural networks to learn to drive cars autonomously around tracks.

## Authors
* **Thiery Deruyterre** - [ThierryDeruyttere](https://github.com/ThierryDeruyttere)
* **Armin Halilovic** – [arminnh](http://github.com/arminnh/)

## License
Distributed under the MIT license. See ``LICENSE`` for more information.

## References

### Papers
- [Human-level control through deep reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)
- [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf)
- [A REINFORCEMENT LEARNING ALGORITHM FOR NEURAL NETWORKS WITH INCREMENTAL LEARNING ABILITY](http://www2.kobe-u.ac.jp/~ozawasei/pub/iconip02a.pdf)
- [DEEP REINFORCEMENT LEARNING: AN OVERVIEW](https://arxiv.org/pdf/1701.07274.pdf)
- [Reinforcement Learning for Robots Using Neural Networks](https://pdfs.semanticscholar.org/54c4/cf3a8168c1b70f91cf78a3dc98b671935492.pdf)

### Useful links
- [Human-level control through deep reinforcement learning slides](http://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_dqn.pdf)
- [NEURAL NETWORKS AND REINFORCEMENT LEARNING](http://web.mst.edu/~gosavia/neural_networks_RL.pdf)
- [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/deeptraffic/)
- [OpenAI Baselines: DQN](https://blog.openai.com/openai-baselines-dqn/)
- [Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning](http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/)
- [Curiosity Driven Exploration by Self-Supervised Prediction](https://www.youtube.com/watch?v=J3FHOyhUn3A)
- [Clever Machines Learn How to Be Curious](https://www.quantamagazine.org/clever-machines-learn-how-to-be-curious-20170919)
- [A solution for Udacity's Self-Driving Car project](https://github.com/alexhagiopol/end-to-end-deep-learning)
- [Self-Driving Truck in Euro Truck Simulator 2, trained via Reinforcement Learning](https://github.com/aleju/self-driving-truck)
- [Competitive Self-Play](https://blog.openai.com/competitive-self-play/)
- [Stock market environment using OpenGym with Deep Q-learning and Policy Gradient](https://github.com/kh-kim/stock_market_reinforcement_learning)
- [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [ConvNetJS Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [ConvNetJS: Deep Learning in your browser](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
- [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
- [Vanilla DQN, Double DQN, and Dueling DQN in PyTorch](https://github.com/dxyang/DQN_pytorch)
