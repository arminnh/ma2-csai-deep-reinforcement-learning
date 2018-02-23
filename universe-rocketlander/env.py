import cv2
from gym.spaces.box import Box
import gym

from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger


def _process_frame84(frame):
    return cv2.resize(frame, (84, 84))

class Rescale84x84(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(Rescale84x84, self).__init__(env)
        self.observation_space = Box(0, 255, [84, 84, 1])

    def _observation(self, observation_n):
        return [_process_frame84(observation) for observation in observation_n]


def create_env(env_id, remotes=1, **_):
    env = gym.make(env_id)
    # Get the vision out of dict that is being sent back when you reset or take a step
    env = Vision(env)

    env = Logger(env)
    env = BlockingReset(env)

    env = Rescale84x84(env)
    env = EpisodeID(env)
    env = Unvectorize(env)

    #fps = env.metadata['video.frames_per_second']
    #env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env