#  Human-level control through deep reinforcement learning
> Project for the course [Capita Selecta Computer Science: Artificial Intelligence](https://onderwijsaanbod.kuleuven.be/syllabi/e/H05N0AE.htm#activetab=doelstellingen_idm1514848) at KU Leuven

## Components of the project

### Main goal
Implementation of an agent, using deep reinforcement learning, that is able to learn to play a variety of (simple) games.

### Presentation (30 min):
1. Intro to the problem this project tries to solve
2. An refresher on reinforcement learning (Q-learning)
3. An introduction to convolutional neural networks.
4. Explanation on deep reinforcement learning
5. Demo!
6. Curious Reinforcment Learning, a short section on what may come after deep reinforcement learning.

### Implementation

#### Neural network
TODO: describe NN design (input 4 video frames -> convolutional layers -> fully connected layers -> output action)

#### Virtual environment

##### Requirements
* Can host a variety of games
* Get observations from environment
* Make actions in environment
* Receive reward for action from environment
* Tracks to drive on

The environment we ended up using OpenAI's Gym.  
We originally started off with Udacity's Behavioral cloning project, but that path was not fruitful in the end, as we were not able to synchronise the data of the simulator with the separate neural network program correctly.  
We then opted for a simpler approach in order to stay within the allocated time for this project. OpenAI's Gym library allowed us to write a working program in much less time. It also gave us the possibility to try out our network on a variety of different games.  
TODO: expand more on this?

### Curiosity learning
TODO

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
TODO: What things you need to install the software and how to install them

    dep 1, dep 2, ... TODO

### Installing
TODO: A step by step series of examples that tell you have to get a development env running

Install the Python dependencies
```sh
pip3 install -r requirements.txt
```

## Usage
TODO: A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

```sh
python3 main.py
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
