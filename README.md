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
6. [Curious Reinforcment Learning](https://www.quantamagazine.org/clever-machines-learn-how-to-be-curious-20170919/), a short section on what may come after deep reinforcement learning. 

### Implementation

#### Neural network (TODO: update this)
* Convolutional neural network
* Inputs = image of view in front of car + velocity of car + acceleration of car + (maybe) angle of steering wheel + (maybe) g forces on car
* Outputs = move left, move right, accelerate, brake, do nothing
* Heuristic for scoring:
* * Reward:
* * * Higher velocity
* * * Lower times to reach destination (so the car goes in the right direction with the higher velocity)
* * * Staying near middle (or one side) of track (maybe)
* * Punish:
* * * Too fast
* * * Not on track
* * * Collisions
* * * Distance from destination

#### Virtual environment requirements
* Tracks to drive on
* Car + methods to drive the car
* Camera in the virtual world (from the perspective of a driver) that allow for image streams to be used as input to the NN.
* A way to retrieve the following data of the car: velocity, acceleration, angle of steering wheel, (maybe) g forces on car
* A way to know if the car is still on the track
* A way to link the NN to the car camera and car controls.

#### Chosen virtual environment
The environment we ended up using OpenAI's Gym.
We originally started off with Udacity's Behavioral cloning project, but that path was not fruitful in the end, as we were not able to synchronise the data of the simulator with the separate neural network program correctly.
We then opted for a simpler approach in order to stay within the allocated time for this project. OpenAI's Gym library allowed us to write a working program in much less time. It also gave us the possibility to try out our network on a variety of different games.
TODO: expand more on this?

### Curiosity learning
https://www.youtube.com/watch?v=J3FHOyhUn3A  
https://www.quantamagazine.org/clever-machines-learn-how-to-be-curious-20170919/  
Paper: https://arxiv.org/pdf/1705.05363.pdf


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

### Data
Load the data into a running MongoDB instance
```sh
mongorestore --drop -d cern_ldb --archive=data-cern/cern_ldb.mongodb.20170309.mki_v0_5.gz --gzip
```

The MongoDB dump archive was created as follows:
```sh
mongodump -d cern_ldb --archive=X.gz --gzip
```

## Usage
TODO: A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

```sh
command
```

## Built with
* [Behavioral Cloning Project](https://github.com/udacity/CarND-Behavioral-Cloning-P3) - Udacity's car simulation environment which allows for neural networks to learn to drive cars autonomously around tracks.
* [OpenAI Gym](https://github.com/openai/gym) - 
* [OpenAI Universe](https://github.com/openai/gym) - 
* [PyTorch](https://github.com/pytorch/pytorch) -

## Authors
* **Thiery Deruyterre** - [ThierryDeruyttere](https://github.com/ThierryDeruyttere)
* **Armin Halilovic** – [arminnh](http://github.com/arminnh/)

## License
Distributed under the MIT license. See ``LICENSE`` for more information.

## References
- [Human-level control through deep reinforcement learning slides](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)
- [Human-level control through deep reinforcement learning slides](http://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_dqn.pdf)
- [OpenAI Baselines: DQN](https://blog.openai.com/openai-baselines-dqn/)
- [Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning](http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/)
- [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/deeptraffic/)
- [A solution for Udacity's Self-Driving Car project](https://github.com/alexhagiopol/end-to-end-deep-learning)
- [Self-Driving Truck in Euro Truck Simulator 2, trained via Reinforcement Learning](https://github.com/aleju/self-driving-truck)
- [Competitive Self-Play](https://blog.openai.com/competitive-self-play/)
- [Stock market environment using OpenGym with Deep Q-learning and Policy Gradient](https://github.com/kh-kim/stock_market_reinforcement_learning)
- [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [ConvNetJS Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [ConvNetJS: Deep Learning in your browser](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
- [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

