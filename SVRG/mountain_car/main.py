# tensorboard --logdir=log_dir --host=127.0.0.1

# Hyper parameters
# random seed
RANDOM_SEED = 6
# neural network input dimensions
SENSOR_INPUT_HEIGHT = 480
SENSOR_INPUT_WIDTH = 640
# training parameters
EPISODES = 10000
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 200
EPISODE_TERMINATION_TIME = 30  # seconds - 65
EPSILON = 1
EPSILON_DECAY = 0.9995  # 0.9975
MIN_EPSILON = 0.001
FPS = 3
BATCH_SIZE = 100
AGGREGATE_STATS_EVERY = 5
# action discretization
DISCRETE_STATE_NUMS = 5
# name the actor and critique model
MODEL_NAME = "sifour"


if __name__ == "__main__":

    # essential library
    import numpy as np
    import tensorflow as tf
    import environment
    from environment import gym_env
    from agent import Agent
    from tqdm import tqdm
    import time
    import os
    import datetime
    from threading import Thread

    # Maintain consistency
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # essential code for setting memory growth on gpu to effeciently use gpu memory
    physical_devices = tf.config.list_physical_devices("GPU")
    print("Available GPUs =", physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # initialize the environment
    env = gym_env()
    current_state = env.reset()
    print("image_shape", current_state.shape)
    # env.destroy_actors()

    # initialize the agent
    agent = Agent(
        env.n_actions,
        replay_memory_size=REPLAY_MEMORY_SIZE,
        min_replay_memory_size=MIN_REPLAY_MEMORY_SIZE,
        epsilon_decay_rate=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )

    # # safety code for warming up the neural networks
    # agent.get_action(np.ones((SENSOR_INPUT_HEIGHT, SENSOR_INPUT_WIDTH, 4)))
    # agent.get_Q_values(np.ones((SENSOR_INPUT_HEIGHT, SENSOR_INPUT_WIDTH, 4)))

    # start parallel training thread
    training_thread = Thread(target=agent.train_in_loop, daemon=True)
    training_thread.start()
    while not agent.training_initialized:
        time.sleep(0.02)

    # initialize metrics
    episode_rewards = []
    epsilon = EPSILON

    # score threshold to save trained model
    best_episode_average_reward = float("-inf")

    # iterate over each season
    for episode in tqdm(
        range(1, EPISODES + 1), ascii=False, unit="episode", desc="EPISODES"
    ):
        agent.tensorboard_step = episode
        current_episode_reward = 0
        current_episode_length = 0
        episode_done = False
        current_state = env.reset()

        # iterate over total number of time steps in eash season
        while not episode_done:
            if np.random.random() > epsilon:
                action = agent.get_action(current_state)
            else:
                action = np.random.randint(0, env.n_actions)
            # action, action_probabs = agent.get_sample_action(current_state)
            # print(action, action_probabs)
            new_state, reward, episode_done, _ = env.step(action)
            current_episode_reward += reward
            # value = agent.get_value(current_state)
            agent.store_transition(
                current_state, action, reward, new_state, episode_done
            )

            # update the state
            current_state = new_state

            # update the current episode length
            current_episode_length += 1

        # restart a new episode
        # env.destroy_actors()

        # Log the metrics for plotting
        episode_rewards.append(current_episode_reward)
        with agent.tensorboard_writer.as_default():
            tf.summary.scalar(
                "Episode Reward",
                data=current_episode_reward,
                step=agent.tensorboard_step,
            )
            tf.summary.scalar(
                "Episode Length",
                data=current_episode_length,
                step=agent.tensorboard_step,
            )

        if episode % AGGREGATE_STATS_EVERY == 0:
            last_n_episode_rewards = episode_rewards[-AGGREGATE_STATS_EVERY:]
            average_reward = sum(last_n_episode_rewards) / AGGREGATE_STATS_EVERY
            min_reward = min(last_n_episode_rewards)
            max_reward = max(last_n_episode_rewards)
            with agent.tensorboard_writer.as_default():
                tf.summary.scalar(
                    "reward_average", data=average_reward, step=agent.tensorboard_step
                )
                tf.summary.scalar(
                    "reward_minimum", data=min_reward, step=agent.tensorboard_step
                )
                tf.summary.scalar(
                    "reward_maximum", data=max_reward, step=agent.tensorboard_step
                )
                tf.summary.scalar("epsilon", data=epsilon, step=agent.tensorboard_step)

            # # save the better season model
            # if average_reward > best_episode_average_reward:
            #     agent.save_models(average_reward)
            #     best_episode_average_reward = average_reward

        epsilon = agent.decay_epsilon(epsilon)

    # end things
    agent.training_terminate = True
    training_thread.join()

    # save the final most model after learning through all the seasons irrespective of the average rewards
    # agent.save_models(best_episode_average_reward)

    # # destroy spawned actors / clean the environment
    # env.destroy_actors()

    print("Total number of mini batches learned = ", agent.learned_steps)
