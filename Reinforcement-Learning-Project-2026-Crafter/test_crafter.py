import crafter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create environment
env = crafter.Env(size=(256, 256))  # Larger size for better visibility

obs = env.reset()
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
print("Number of actions:", env.action_space.n)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
img_plot = ax.imshow(obs)
ax.axis('off')
plt.title('Crafter Environment (Random Actions)')

done = False
steps = 0
max_steps = 500

def update_frame(frame):
    global obs, done, steps
    
    if done or steps >= max_steps:
        plt.close()
        return [img_plot]
    
    # Take random action
    action = np.random.randint(0, env.action_space.n)
    obs, reward, done, info = env.step(action)
    steps += 1
    
    # Update display
    img_plot.set_array(obs)
    plt.title(f'Crafter Environment - Step {steps} - Reward: {reward:.1f}')
    
    return [img_plot]

# Create animation (updates every 50ms)
anim = FuncAnimation(fig, update_frame, interval=50, blit=True, cache_frame_data=False)

plt.tight_layout()
plt.show()

print(f"✅ Environment demo finished, ran {steps} steps")