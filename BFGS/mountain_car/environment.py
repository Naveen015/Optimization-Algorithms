import sys
import glob
import os
import time
import numpy as np
import cv2
import math
import itertools
import random
import gym

# HYPER PARAMETERS

# Environment Code
class gym_env:
    def __init__(self):
        # establish connection to the server
        self.env = gym.make("MountainCar-v0")
        self.sim = self.env
        self.actions = list(range(self.env.action_space.n))

    def visually_simulate(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        # convert neural network max output neuron index to environment action
        # print(action)
        action = np.array(self.actions[action])

        # control the vehicle by giving inputs
        state, reward, done, extra = self.env.step(action)

        return state, reward, done, None

    def destroy_env(self):
        self.env.close()
