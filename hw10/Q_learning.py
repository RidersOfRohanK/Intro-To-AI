import gym
import random
import numpy as np
import time
from collections import deque
import pickle

from collections import defaultdict

EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)

    numStates = env.observation_space.n
    numActions = env.action_space.n

    # You will need to update the Q_table in your iteration
    Q_table = np.zeros([numStates, numActions])
    # Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            rnd = np.random.uniform(0, 1)

            # If random number < epsilon, take a random action
            if rnd < EPSILON:
                action = env.action_space.sample()
            # Else, take the action with the highest value in the current state
            else:
                action = np.argmax(Q_table[obs, :])

            # Implement this action and move the agent in the desired direction
            next_state, reward, done, info = env.step(action)
            nextMax = np.max(Q_table[next_state, :])
            # We update our Q-table using the Q-learning iteration
            newval = (1 - LEARNING_RATE) * Q_table[obs, action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * nextMax)
            Q_table[obs, action] = newval
            episode_reward += reward
            # If the episode is finished, we leave the for loop
            if done:
                episode_reward_record.append(episode_reward)
                val = (1 - LEARNING_RATE) * Q_table[obs, action] + LEARNING_RATE * reward
                Q_table[obs, action] = val

                break
            obs = next_state
        EPSILON = EPSILON * EPSILON_DECAY
        if EPSILON < 0.01:
            EPSILON = 0.01

        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record)) / 100))
            print("EPSILON: " + str(EPSILON))

    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    model_file.close()
    #########################