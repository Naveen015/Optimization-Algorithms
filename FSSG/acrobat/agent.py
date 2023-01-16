import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import DQN_Model
from buffer import ppoMemory
import main
import time
import datetime
import os

# Hyper parameters
MODEL_NAME = main.MODEL_NAME


class Agent:
    def __init__(
        self,
        n_actions,
        update_target_every_n_episode=5,
        replay_memory_size=5000,
        min_replay_memory_size=1000,
        prediction_batch_size=1,
        mini_batch_size=8,
        discount=0.99,
        epsilon_decay_rate=0.95,
        min_epsilon=0.001,
        learning_rate=0.001,
        model_dir="models",
    ):
        self.n_actions = n_actions
        self.update_target_every_n_episode = update_target_every_n_episode
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.prediction_batch_size = prediction_batch_size
        self.mini_batch_size = mini_batch_size
        self.discount = discount
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.training_batch_size = self.mini_batch_size // 4

        # NN Model initializations
        self.dqn_model = DQN_Model(self.n_actions,)
        self.target_dqn_model = DQN_Model(self.n_actions,)
        self.target_dqn_model.set_weights(self.dqn_model.get_weights())
        self.target_update_counter = 0
        self.training_initialized = False
        self.training_terminate = False
        self.update_target_every_n_episode = update_target_every_n_episode
        self.learned_steps = 0

        # initialize buffer memory
        self.ppo_buffer = ppoMemory(
            replay_memory_size=self.replay_memory_size,
            mini_batch_size=self.mini_batch_size,
        )

        # for plotting graphs and logging metrics
        self.log_folder = "log_dir"
        self.tensorboard_writer = tf.summary.create_file_writer(
            self.log_folder
            + "/"
            + MODEL_NAME
            + "_"
            + datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            + "/"
        )
        self.tensorboard_step = 1
        self.last_logged_episode = 0

    def store_transition(self, current_state, action, reward, new_state, done):
        # transitions = (state, action, action_probabs, value, reward, done)
        self.ppo_buffer.save_transition(current_state, action, reward, new_state, done)

    def get_Q_values(self, state):
        state = np.array(state).reshape(-1, *state.shape)
        q_values = tf.squeeze(self.dqn_model(state))

    def get_action(self, state):
        action = np.argmax(self.get_Q_values(state))
        return action

    def train(self):
        if len(self.ppo_buffer.replay_memory) < self.min_replay_memory_size:
            return
        # print("learning")

        (
            mini_batch,
            current_states,
            new_current_states,
        ) = self.ppo_buffer.return_mini_buffer()
        current_q_values = self.dqn_model(current_states).numpy()
        future_q_values = self.target_dqn_model(new_current_states).numpy()

        input_x = []
        true_output_y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(
            mini_batch
        ):
            if not done:
                max_future_q_value = np.max(future_q_values[index])
                new_q_value = reward + self.discount * max_future_q_value
            else:
                new_q_value = reward

            current_q_value = current_q_values[index]
            current_q_value[action] = new_q_value

            input_x.append(current_state)
            true_output_y.append(current_q_value)

        def model_function(fixed_step_data_x):
            input_x = fixed_step_data_x.reshape(-1, *fixed_step_data_x.shape)
            predicted_y = self.dqn_model(input_x)
            return predicted_y

        def gradient(model_function, X, fixed_step_data_x, fixed_step_data_y):
            with tf.GradientTape(persistent=True) as tape:
                Y = model_function(fixed_step_data_x)
                loss = tf.reduce_mean(tf.square(fixed_step_data_y - Y))
            gradient = tape.gradient(loss, X)
            return gradient

        def question1(X_0, model_function, data_x, data_y, lr=0.001, sgd_iters=10):
            # X_0 - shape - n,m. n - datapoints, m - features
            # target_functio - function that takes two arguments and returns real number
            n = len(data_x)
            X = X_0
            for j in range(sgd_iters):
                k = np.random.randint(0, 8)
                g = gradient(model_function, X, data_x[k], data_y[k])
                for i, x in enumerate(X):
                    X[i].assign_sub(lr * g[i])
                # X.assign_sub(lr * g)
            return X

        X = question1(
            self.dqn_model.trainable_variables,
            model_function,
            np.array(input_x),
            np.array(true_output_y),
            lr=self.learning_rate,
            sgd_iters=100,
        )
        for i, x in enumerate(X):
            self.dqn_model.trainable_variables[i].assign_sub(X[i])

        # hist = self.dqn_model.fit(
        #     np.array(input_x),
        #     np.array(true_output_y),
        #     batch_size=self.training_batch_size,
        #     verbose=None,
        #     shuffle=False,
        #     # callbacks=[self.tensorboard] if log_this_step else None,
        # )
        self.learned_steps += 1

        if self.tensorboard_step > self.last_logged_episode:
            # with self.tensorboard_writer.as_default():
            #     tf.summary.scalar(
            #         "accuracy",
            #         data=hist.history["accuracy"][0],
            #         step=self.tensorboard_step,
            #     )
            #     tf.summary.scalar(
            #         "loss", data=hist.history["loss"][0], step=self.tensorboard_step
            #     )
            self.last_logged_episode = self.tensorboard_step
            self.target_update_counter += 1

        if self.target_update_counter > self.update_target_every_n_episode:
            # print("network updated")
            self.target_dqn_model.set_weights(self.dqn_model.get_weights())
            self.target_update_counter = 0

    def train_in_loop(self):
        # # warmup
        # X = np.random.uniform(size=(1, *self.observation_dimensions)).astype(np.float32)
        # y = np.random.uniform(size=(1, self.n_actions)).astype(np.float32)
        # self.dqn_model.fit(X, y, verbose=None, batch_size=1)

        self.training_initialized = True
        while True:
            if self.training_terminate:
                return
            self.train()
            time.sleep(0.03)

    def decay_epsilon(self, epsilon):
        epsilon = epsilon * self.epsilon_decay_rate
        epsilon = max(self.min_epsilon, epsilon)
        return epsilon

    def save_models(self, mean_return):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
            time.sleep(2)
        time_str = str(int(time.time()))
        self.target_dqn_model.save(
            f"{self.model_dir}/actor-{MODEL_NAME}__{mean_return:_>7.2f}mean_return__{datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')}",
        )

