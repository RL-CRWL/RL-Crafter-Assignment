'''
Components of a PPO Algorithm:
Actor: This is the policy network responsible for choosing actions to take in a given state. 
Critic: This is the value function approximator that estimates the value of states or state-action pairs, 
providing feedback to the actor. 
Generalized Advantage Estimation (GAE): A technique used by PPO to compute advantage values, 
which helps reduce variance in policy gradient estimates. 
Clipped Surrogate Objective: The core mechanism of PPO that limits the magnitude of policy updates during training, ensuring stability. 
'''

import gymnasium as gym
import stable_baselines3
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#code: adapted from geeksForgeeks - https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/
class ActorCriticAgent:
    def __init__(self, env, actor_lr=0.001, critic_lr=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma

        #the networks using tensorflow
        #this was adapated to handling images, by using Flatten, Dense and a 2D convolution network
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.env.action_space.n, activation='softmax')
        ])

        self.critic = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        #using adam optimizer as a default but it should 
        #adaptively adjust the learning rate for each parameter based on the history of its gradients
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def select_action(self, state):
        #feeding raw images from the crafter information
        state = state.astype(np.float32) / 255.0  # normalize image
        state = np.expand_dims(state, axis=0)    
        probs = self.actor(state).numpy().flatten() #flattening the pixels so can be worked with my numpy
        probs = probs / np.sum(probs)
        action = np.random.choice(self.env.action_space.n, p=probs)
        return action, probs

    #the advantage function measures how much better an action taken 
    #in a given state is compared to the average action in that state
    def compute_advantage(self, state, next_state, reward, done):
        """Compute advantage using CNN-based critic."""
        state = state.astype(np.float32) / 255.0
        next_state = next_state.astype(np.float32) / 255.0
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        state_value = self.critic(state)[0, 0]
        next_state_value = self.critic(next_state)[0, 0]
        target = reward + (0 if done else self.gamma * next_state_value)
        advantage = target - state_value
        return advantage, state_value

    #so the actor should be updated with the best action 
    #and the critic with the best action-value
    def update_networks(self, state, action, reward, next_state, done):
        state = state.astype(np.float32) / 255.0
        next_state = next_state.astype(np.float32) / 255.0
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        #this does the differenation for the objective and loss functions
        with tf.GradientTape(persistent=True) as tape:
            
            probs = self.actor(state)
            value = self.critic(state)[0, 0]
            next_value = self.critic(next_state)[0, 0]

            
            target = reward + (0 if done else self.gamma * next_value)
            advantage = target - value

            log_prob = tf.math.log(probs[0, action] + 1e-10)
            actor_loss = -log_prob * tf.stop_gradient(advantage)
            critic_loss = tf.square(target - value)

        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape

    def plot_rewards(self, rewards):
        episodes = np.arange(len(rewards))
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, label="Episode Reward", alpha=0.7)
        
        
        if len(rewards) >= 50:
            moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(np.arange(49, len(rewards)), moving_avg, label="Moving Average (50)", linewidth=2)
        
        plt.title("Training Reward Trend")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    
    
    def train(self, num_episodes=1000, max_steps=10000):
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action, _ = self.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                self.update_networks(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)  # <-- collect total reward

            if episode % 10 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

        self.env.close()
        self.plot_rewards(episode_rewards)  # <-- visualize trend


if __name__ == "__main__":
    raw_env = crafter.Env()
    env = GymV21CompatibilityV0(env=raw_env)
    agent = ActorCriticAgent(env)
    agent.train(num_episodes=10)
    env = crafter.Recorder(
        env, './path/to/logdir',
        save_stats=True,
        save_video=False,
        save_episode=False,
    )




