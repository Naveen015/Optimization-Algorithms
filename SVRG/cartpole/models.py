import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers

# We need to define 2 networks
# 1. Actor network - input: state, output: action, probability
# 2. critique network - input: state, output: Value

# Maintain consistency
tf.random.set_seed(6)
# tf.get_logger().setLevel("WARNING")

# DQN model
class DQN_Model(keras.Model):
    def __init__(self, n_actions):
        super(DQN_Model, self).__init__()
        self.hidden1 = layers.Dense(128, activation="relu")
        self.hidden2 = layers.Dense(256, activation="relu")
        self.outputs = layers.Dense(n_actions, activation="softmax")

    def call(self, state):
        # print(state.shape)
        x = self.hidden1(state)
        x = self.hidden2(x)
        x = self.outputs(x)
        return x
