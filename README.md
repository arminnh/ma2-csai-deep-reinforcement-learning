<<<<<<< HEAD
# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

=======
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
>>>>>>> 47c74d05b81bd2f83a7fa29155cc21d2cf59902d
