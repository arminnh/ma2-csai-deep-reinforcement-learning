import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from TORCH_DQN import DQN
from enum import Enum
import torchvision.transforms as T
import ast

parser = argparse.ArgumentParser(description='Remote Driving')
# parser.add_argument(
#     'model',
#     type=str,
#     help='Path to model h5 file. Model should be on the same path.'
# )
parser.add_argument(
    'gpu',
    type=int,
    default=None,
    help='The gpu this will run on'
)
args = parser.parse_args()


class Moves(Enum):
    LEFT = 0
    RIGHT = 1
    ACCELERATE = 2
    BRAKE = 3
    NOTHING = 4

class Steering:
    # Based on code found in steering.cs

    def __init__(self):
        self.H = 0.
        self.V = 0.

    def update(self, move):
        if move == Moves.LEFT:
            if self.H > -1.0:
                self.H -= 0.05
        elif move == Moves.RIGHT:
            if self.H < 1.0:
                self.H += 0.05
        elif move == Moves.ACCELERATE:
            if self.V < 1.0:
                self.V += 0.05
        elif move == Moves.BRAKE:
            if self.V > 0:
                self.V -= 0.05
        elif move == Moves.NOTHING:
            pass

    def reset(self):
        self.H = 0
        self.V = 0

    def getAngle(self):
        return self.H

    def getAcceleration(self):
        return self.V

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral



class SelfDrivingAgent:

    def __init__(self, set_speed = 9):
        self.controller = SimplePIController(0.1, 0.002)
        self.controller.set_desired(set_speed)
        if args.gpu is not None:
            self.DQN = DQN(len(Moves))
        else:
            self.DQN = DQN(len(Moves), True, args.gpu)


        self.state = None
        self.lastScreen = None
        self.resize = T.Compose([T.Scale(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
        self.steer = Steering()
        self.noSpeed = 0
        self.oldReward = 0
        self.currentReward = 0

    def rewardFunction(self, speed, touches_track, time_alive, steering):
        touches = int(touches_track)
        if speed == 0:
            self.noSpeed += 1
        else:
            self.noSpeed = 0

        self.oldReward = self.currentReward
        val = (touches * int(speed>1) - (1-touches) * 10 - self.noSpeed - int(abs(steering) > 0.5) * 5) * time_alive / 1000
        self.currentReward = self.oldReward * 0.95 + val
        return self.currentReward


agent = SelfDrivingAgent()
sio = socketio.Server(async_handlers=True)
app = Flask(__name__)


@sio.on("reset")
def resetAgent(sid, data):
    agent.steer.reset()
    sio.emit('manual', data={}, skip_sid=True)

@sio.on("manual")
def manual(sid, data):
    # The current steering angle of the car
    old_steering_angle = ast.literal_eval(data["steering_angle"])
    # The current throttle of the car
    throttle = ast.literal_eval(data["throttle"])
    # The current speed of the car
    speed = ast.literal_eval(data["speed"])
    time_alive = ast.literal_eval(data["time_alive"])

    touches_track = ast.literal_eval(data["touches_track"])

    # Change args to do something
    reward = agent.rewardFunction(speed, touches_track, time_alive, old_steering_angle)
    print("getting values")
    send_reward(reward)


def send_reward(reward):
    sio.emit(
        "reward",
        data={
            'reward': reward.__str__(),
        },
        skip_sid=True)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        old_steering_angle = ast.literal_eval(data["steering_angle"])
        # The current throttle of the car
        throttle = ast.literal_eval(data["throttle"])
        # The current speed of the car
        speed = ast.literal_eval(data["speed"])
        time_alive = ast.literal_eval(data["time_alive"])

        touches_track = ast.literal_eval(data["touches_track"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        current_screen = agent.resize(image)

        if agent.state is None:
            agent.state = current_screen - current_screen
            agent.lastScreen = current_screen

        next_state = current_screen - agent.lastScreen

        #steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        #throttle = controller.update(float(speed))

        action = agent.DQN.act(agent.state)

        # Update controller
        agent.steer.update(Moves(action))

        steering_angle = agent.steer.getAngle()
        throttle = agent.steer.getAcceleration()

        # Change args to do something
        reward = agent.rewardFunction(speed, touches_track, time_alive, steering_angle)
        if reward < -100:
            sio.emit("reset",  data={}, skip_sid=True)
            agent.oldReward = 0
            agent.currentReward = 0
            return

        agent.DQN.remember(agent.state, action, reward, next_state, False)
        print("Time alive: {}, Score: {}, Speed: {}".format(data["time_alive"], reward, speed, old_steering_angle))

        agent.lastScreen = current_screen
        agent.state = next_state
        agent.DQN.replay(128)

        # # save frame
        # if args.image_folder != '':
        #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        #     image_filename = os.path.join(args.image_folder, timestamp)
        #     image.save('{}.jpg'.format(image_filename))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)
        send_reward(reward)


    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    # check that model Keras version is same as local Keras version
    # Swap out for pytorch
    #f = h5py.File(args.model, mode='r')
    # model_version = f.attrs.get('keras_version')
    # keras_version = str(keras_version).encode('utf8')
    #
    # if model_version != keras_version:
    #     print('You are using Keras version ', keras_version,
    #           ', but the model was built using ', model_version)
    #
    # model = load_model(args.model)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('localhost', 4567)), app)
