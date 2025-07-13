import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import random

env = gym.make("CartPole-v1")
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

def build_q_network():
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=(state_shape,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = deque(maxlen=2000)
episodes = 500
target_update_freq = 10

main_model = build_q_network()
target_model = build_q_network()
target_model.set_weights(main_model.get_weights())

def get_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.randint(num_actions)
    q_values = main_model.predict(state[np.newaxis], verbose=0)[0]
    return np.argmax(q_values)

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = get_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            states = np.array([exp[0] for exp in minibatch])
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch])
            next_states = np.array([exp[3] for exp in minibatch])
            dones = np.array([exp[4] for exp in minibatch])

            target_qs = target_model.predict(next_states, verbose=0)
            target_values = rewards + (1 - dones) * gamma * np.amax(target_qs, axis=1)

            q_values = main_model.predict(states, verbose=0)
            for i, action in enumerate(actions):
                q_values[i][action] = target_values[i]

            main_model.fit(states, q_values, epochs=1, verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update_freq == 0:
        target_model.set_weights(main_model.get_weights())

    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")
