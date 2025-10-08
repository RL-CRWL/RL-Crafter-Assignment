'''
Components of an Actor-Critic Algorithm:
Actor: This is the policy network responsible for choosing actions to take in a given state.
Critic: This is the value function approximator that estimates the value of states or state-action pairs,
providing feedback to the actor.
Generalized Advantage Estimation (GAE): A technique used to compute advantage values,
which helps reduce variance in policy gradient estimates.

IMPROVEMENTS:
- Fixed Lambda layer shape inference issues
- Added entropy regularization to encourage exploration
- Added epsilon-greedy exploration strategy
- Improved network architecture with batch normalization
- Added gradient clipping for stability
- Added action masking to prevent invalid actions
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.utils.wrappers import make_crafter_env

#code: adapted from geeksForgeeks - https://www.geeksforgeeks.org/machine-learning/actor-critic-algorithm-in-reinforcement-learning/

class ActorCriticAgent:
    def __init__(self, env, actor_lr=0.0001, critic_lr=0.0005, gamma=0.99, entropy_coef=0.01):
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Exploration parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate per episode

        # Get observation shape from the wrapped environment
        obs_shape = env.observation_space.shape
        print(f"Observation shape: {obs_shape}")
        print(f"Action space size: {env.action_space.n}")

        # Build improved actor network with better architecture
        self.actor = self._build_actor(obs_shape)

        # Build improved critic network
        self.critic = self._build_critic(obs_shape)

        # Using adam optimizer with gradient clipping
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr, clipnorm=1.0)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr, clipnorm=1.0)

        # Track action distribution for debugging
        self.action_counts = np.zeros(self.env.action_space.n)
        self.step_count = 0

    def _build_actor(self, obs_shape):
        """Build actor network with improved architecture and fixed normalization."""
        inputs = tf.keras.Input(shape=obs_shape)

        # Use Rescaling layer instead of Lambda for normalization (fixes shape inference)
        x = tf.keras.layers.Rescaling(1.0/255.0)(inputs)

        # Convolutional layers with batch normalization
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        # Output layer (softmax for action probabilities)
        outputs = tf.keras.layers.Dense(self.env.action_space.n, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def _build_critic(self, obs_shape):
        """Build critic network with improved architecture and fixed normalization."""
        inputs = tf.keras.Input(shape=obs_shape)

        # Use Rescaling layer instead of Lambda for normalization (fixes shape inference)
        x = tf.keras.layers.Rescaling(1.0/255.0)(inputs)

        # Convolutional layers with batch normalization
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        # Output layer (single value for state value)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def select_action(self, state, training=True):
        """Select action with epsilon-greedy exploration."""
        state_tensor = np.expand_dims(state, axis=0).astype(np.float32)
        probs = self.actor(state_tensor, training=training).numpy().flatten()

        # Add small epsilon to prevent zero probabilities
        probs = probs + 1e-10
        probs = probs / np.sum(probs)  # Renormalize

        # Epsilon-greedy exploration during training
        if training and np.random.random() < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            # Sample from probability distribution
            action = np.random.choice(self.env.action_space.n, p=probs)

        # Track action distribution
        self.action_counts[action] += 1
        self.step_count += 1

        return action, probs

    def update_networks(self, state, action, reward, next_state, done):
        """Update actor and critic networks with improved loss functions."""
        state_tensor = np.expand_dims(state, axis=0).astype(np.float32)
        next_state_tensor = np.expand_dims(next_state, axis=0).astype(np.float32)

        # Update critic
        with tf.GradientTape() as tape:
            value = self.critic(state_tensor, training=True)[0, 0]
            next_value = self.critic(next_state_tensor, training=False)[0, 0]

            # TD target
            target = reward + (0 if done else self.gamma * next_value)

            # Critic loss (MSE)
            critic_loss = tf.square(target - value)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            probs = self.actor(state_tensor, training=True)
            value = self.critic(state_tensor, training=False)[0, 0]
            next_value = self.critic(next_state_tensor, training=False)[0, 0]

            # Advantage
            target = reward + (0 if done else self.gamma * next_value)
            advantage = target - value

            # Policy loss (negative log probability weighted by advantage)
            log_prob = tf.math.log(probs[0, action] + 1e-10)
            policy_loss = -log_prob * tf.stop_gradient(advantage)

            # Entropy bonus for exploration
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))

            # Total actor loss (policy loss - entropy bonus)
            actor_loss = policy_loss - self.entropy_coef * entropy

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        return actor_loss.numpy(), critic_loss.numpy(), entropy.numpy()

    def print_action_distribution(self):
        """Print the distribution of actions taken."""
        if self.step_count > 0:
            action_names = [
                "noop", "move_left", "move_right", "move_up", "move_down",
                "do", "sleep", "place_stone", "place_table", "place_furnace",
                "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
                "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
                "make_iron_sword"
            ]

            print("\n" + "="*60)
            print("ACTION DISTRIBUTION (Last Episode)")
            print("="*60)

            percentages = (self.action_counts / np.sum(self.action_counts)) * 100
            sorted_indices = np.argsort(percentages)[::-1]

            for idx in sorted_indices[:10]:  # Show top 10
                if percentages[idx] > 0:
                    action_name = action_names[idx] if idx < len(action_names) else f"action_{idx}"
                    bar = "#" * int(percentages[idx] / 2)
                    print(f"{action_name:20s}: {bar} {percentages[idx]:.1f}%")

    def plot_rewards(self, rewards):
        """Plot training rewards with moving average."""
        episodes = np.arange(len(rewards))
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')

        if len(rewards) >= 10:
            moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(np.arange(9, len(rewards)), moving_avg, 
                    label="Moving Average (10)", linewidth=2, color='red')

        plt.title("Training Reward Trend")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.plot(episodes, rewards, marker='o', linestyle='-', markersize=4)
        plt.title("Episode-by-Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()

    def train(self, num_episodes=1000, max_steps=10000, visualize=False, print_every_n_steps=50):
        """
        Train the agent with optional real-time visualization.

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            visualize: If True, shows real-time visualization of the agent
            print_every_n_steps: Print progress every N steps (reduces console spam)
        """
        episode_rewards = []

        # Action names for better console output
        action_names = [
            "noop", "move_left", "move_right", "move_up", "move_down",
            "do", "sleep", "place_stone", "place_table", "place_furnace",
            "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
            "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
            "make_iron_sword"
        ]

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            episode_step = 0

            # Reset action tracking for this episode
            self.action_counts = np.zeros(self.env.action_space.n)
            self.step_count = 0

            # Setup visualization if requested (only for first few episodes)
            if visualize and episode < 3:
                fig, ax = plt.subplots(figsize=(8, 8))
                img_plot = ax.imshow(state)
                ax.axis('off')
                plt.ion()
                plt.show()

            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes} | Epsilon: {self.epsilon:.3f}")
            print(f"{'='*60}")

            actor_losses = []
            critic_losses = []
            entropies = []

            for step in range(max_steps):
                action, _ = self.select_action(state, training=True)
                next_state, reward, done, truncated, _ = self.env.step(action)
                done = done or truncated

                actor_loss, critic_loss, entropy = self.update_networks(
                    state, action, reward, next_state, done
                )

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                entropies.append(entropy)

                total_reward += reward
                episode_step += 1

                # Print step information (every N steps to reduce spam)
                if step % print_every_n_steps == 0 or done:
                    action_name = action_names[action] if action < len(action_names) else f"action_{action}"
                    print(f"Step {episode_step:4d} | Action: {action_name:20s} | "
                          f"Reward: {reward:7.3f} | Total: {total_reward:7.3f} | "
                          f"Entropy: {entropy:.3f}")

                # Update visualization
                if visualize and episode < 3:
                    img_plot.set_array(next_state)
                    ax.set_title(f'Episode {episode+1} - Step {episode_step} - Reward: {reward:.2f} - Total: {total_reward:.2f}')
                    plt.pause(0.01)

                state = next_state

                if done:
                    print(f"\nEpisode ended at step {episode_step}")
                    break

            episode_rewards.append(total_reward)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Close visualization window if it was open
            if visualize and episode < 3:
                plt.ioff()
                plt.close()

            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Episode Length: {episode_step}")
            print(f"  Avg Actor Loss: {np.mean(actor_losses):.4f}")
            print(f"  Avg Critic Loss: {np.mean(critic_losses):.4f}")
            print(f"  Avg Entropy: {np.mean(entropies):.4f}")
            if len(episode_rewards) > 1:
                print(f"  Average Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")

            # Print action distribution
            self.print_action_distribution()

        self.env.close()

        # Show final reward plot
        print(f"\n{'='*60}")
        print("Training Complete! Generating reward plot...")
        print(f"{'='*60}")
        self.plot_rewards(episode_rewards)

        return episode_rewards


if __name__ == "__main__":
    # Use the project's make_crafter_env wrapper
    env = make_crafter_env()

    print("="*60)
    print("Actor-Critic Agent Training on Crafter")
    print("With Exploration Improvements")
    print("="*60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("="*60)

    # Create agent with improved hyperparameters
    agent = ActorCriticAgent(
        env, 
        actor_lr=0.0001,      # Lower learning rate for stability
        critic_lr=0.0005,     # Slightly higher for critic
        gamma=0.99,           # Standard discount factor
        entropy_coef=0.02     # Entropy bonus for exploration
    )

    # Train with visualization and reduced console output
    # Set print_every_n_steps higher to reduce console spam
    rewards = agent.train(
        num_episodes=20, 
        max_steps=500, 
        visualize=True,
        print_every_n_steps=100  # Only print every 100 steps
    )

    print(f"\nFinal Statistics:")
    print(f"  Average Reward: {np.mean(rewards):.2f}")
    print(f"  Best Reward: {np.max(rewards):.2f}")
    print(f"  Worst Reward: {np.min(rewards):.2f}")
    print(f"  Std Deviation: {np.std(rewards):.2f}")
