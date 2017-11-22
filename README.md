#  Human-level control through deep reinforcement learning
> Project for the course [Capita Selecta Computer Science: Artificial Intelligence](https://onderwijsaanbod.kuleuven.be/syllabi/e/H05N0AE.htm#activetab=doelstellingen_idm1514848) at KU Leuven

### Plan for presentation (30 min)
1. Short intro to reinforcement learning works (2/3 min)
2. Short intro to neural networks and in particular Convolutional neural networks (2/3 min)
3. Explanation of deep reinforcement learning. Discuss how the two previous topics can be combined into one. (12 min)
4. Demo (10 min)
5. What comes after deep reinforcement learning? Small talk about curiosity learning. [Curious Reinforcment Learning](https://www.quantamagazine.org/clever-machines-learn-how-to-be-curious-20170919/) (2 min)

### Demo based on project
Application of deep reinforcement learning for self driving cars in a virtual world.

#### Planned neural network
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

#### Requirements in virtual environment
* Tracks to drive on
* Car + methods to drive the car
* Camera in the virtual world (from the perspective of a driver) that allow for image streams to be used as input to the NN.
* A way to retrieve the following data of the car: velocity, acceleration, angle of steering wheel, (maybe) g forces on car
* A way to know if the car is still on the track
* A way to link the NN to the car camera and car controls.

#### Curiosity learning
https://www.youtube.com/watch?v=J3FHOyhUn3A  
https://www.quantamagazine.org/clever-machines-learn-how-to-be-curious-20170919/  
Paper: https://arxiv.org/pdf/1705.05363.pdf


### Links
- [Human-level control through deep reinforcement learning slides](http://www.teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_dqn.pdf)
- [OpenAI Baselines: DQN](https://blog.openai.com/openai-baselines-dqn/)
- [Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning](http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/)
- [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/deeptraffic/)
- [Self-Driving Truck in Euro Truck Simulator 2, trained via Reinforcement Learning](https://github.com/aleju/self-driving-truck)
- [Competitive Self-Play](https://blog.openai.com/competitive-self-play/)
- [The Self Learning Quant](https://hackernoon.com/the-self-learning-quant-d3329fcc9915)
- [stock market environment using OpenGym with Deep Q-learning and Policy Gradient](https://github.com/kh-kim/stock_market_reinforcement_learning)
- [github Autonomous driving in the Unity engine.](https://github.com/alexhagiopol/end-to-end-deep-learning)
- [pdf End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)
- [ConvNetJS Deep Q Learning Demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [ConvNetJS: Deep Learning in your browser](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
- [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
<!-- ![](header.png) -->

<!--
## Installation

OS X & Linux:

```sh
command
```
-->

### Usage example

TODO: A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

```sh
command
```  

### Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### Prerequisites
TODO: What things you need to install the software and how to install them

    dep 1, dep 2, ... TODO

#### Installing
TODO: A step by step series of examples that tell you have to get a development env running

Install the Python dependencies
```sh
pip3 install -r requirements.txt
```

#### Data
Load the data into a running MongoDB instance
```sh
mongorestore --drop -d cern_ldb --archive=data-cern/cern_ldb.mongodb.20170309.mki_v0_5.gz --gzip
```

The MongoDB dump archive was created as follows:
```sh
mongodump -d cern_ldb --archive=X.gz --gzip
```

### Built with
* [MongoDB](http://mongodb.com) - The database used
* [scikit-learn](http://scikit-learn.org) - Machine learning in Python
* ...

### Authors

* **Thiery Deruyterre** - [ThierryDeruyttere](https://github.com/ThierryDeruyttere)
* **Armin Halilovic** – [arminnh](http://github.com/arminnh/)


### License
Distributed under the MIT license. See ``LICENSE`` for more information.
