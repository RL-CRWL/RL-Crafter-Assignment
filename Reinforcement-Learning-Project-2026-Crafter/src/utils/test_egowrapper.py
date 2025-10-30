"""
Test the Crafter egocentric wrapper
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ego_wrapper import make_crafter_egocentric_env

def test_wrapper():
    print("\n" + "="*70)
    print("TESTING CRAFTER EGOCENTRIC WRAPPER")
    print("="*70 + "\n")
    
    # Test 1: Full egocentric (64x64)
    print("Test 1: Full egocentric (no crop)")
    env = make_crafter_egocentric_env(seed=42, crop_size=None, debug=True)
    obs, info = env.reset()
    print(f"  Output shape: {obs.shape}")
    assert obs.shape == (3, 64, 64), f"Expected (3, 64, 64), got {obs.shape}"
    
    # Take some steps and visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Egocentric View After Different Actions', fontsize=16)
    
    actions = [0, 3, 2, 4, 1, 0]  # noop, up, right, down, left, noop
    action_names = ['noop', 'up', 'right', 'down', 'left', 'noop']
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        obs, reward, done, truncated, info = env.step(action)
        
        ax = axes[i // 3, i % 3]
        ax.imshow(obs.transpose(1, 2, 0))
        ax.set_title(f'After {name} (action {action})')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('egocentric_test_full.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization: egocentric_test_full.png\n")
    env.close()
    
    # Test 2: Cropped egocentric (32x32)
    print("Test 2: Cropped egocentric (32x32)")
    env = make_crafter_egocentric_env(seed=42, crop_size=32, debug=False)
    obs, info = env.reset()
    print(f"  Output shape: {obs.shape}")
    assert obs.shape == (3, 32, 32), f"Expected (3, 32, 32), got {obs.shape}"
    
    # Visualize cropped view
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cropped Egocentric View (32x32)', fontsize=16)
    
    for i, (action, name) in enumerate(zip(actions, action_names)):
        obs, reward, done, truncated, info = env.step(action)
        
        ax = axes[i // 3, i % 3]
        ax.imshow(obs.transpose(1, 2, 0))
        ax.set_title(f'After {name} (action {action})')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('egocentric_test_crop32.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization: egocentric_test_crop32.png\n")
    env.close()
    
    # Test 3: Different crop size (48x48)
    print("Test 3: Cropped egocentric (48x48)")
    env = make_crafter_egocentric_env(seed=42, crop_size=48, debug=False)
    obs, info = env.reset()
    print(f"  Output shape: {obs.shape}")
    assert obs.shape == (3, 48, 48), f"Expected (3, 48, 48), got {obs.shape}"
    print("  ✓ Test passed\n")
    env.close()
    
    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_wrapper()