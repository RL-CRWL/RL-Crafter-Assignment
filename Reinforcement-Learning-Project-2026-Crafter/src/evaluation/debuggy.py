"""
Debug script to check what's actually being used in training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.configs import get_config

# Load the config you used
config = get_config('DQN_IMPROVEMENT_1')

print("="*70)
print("DEBUGGING DQN IMPROVEMENT 1")
print("="*70)

print("\n📋 Config loaded:")
for key, value in config.items():
    print(f"  {key:30s}: {value}")

print("\n🔍 Key checks:")
print(f"  use_custom_cnn:      {config.get('use_custom_cnn', False)}")
print(f"  preprocess_type:     {config.get('preprocess_type', 'none')}")
print(f"  use_frame_stacking:  {config.get('use_frame_stacking', False)}")

# Check if DQN_improvement1.py exists
improvement_file = "src/agents/DQN_improv1.py"
if os.path.exists(improvement_file):
    print(f"\n✓ {improvement_file} exists")
else:
    print(f"\n✗ {improvement_file} NOT FOUND!")
    print("  This means train_with_config.py is using DQN_baseline.py!")
    print("  Your 'improvement' was actually just baseline with different hyperparameters!")

# Check what agent would be created
print("\n🤖 Agent that would be created:")

is_improvement1 = 'improvement1' in config['name'].lower() or config.get('use_custom_cnn', False)

if is_improvement1:
    print("  Type: DQNImprovement1")
    print("  Features:")
    print("    - Custom CNN with batch norm")
    print("    - Better feature extraction")
    if config.get('use_frame_stacking', False):
        print("    - Frame stacking enabled")
    else:
        print("    - Frame stacking disabled")
else:
    print("  Type: DQNAgent (BASELINE)")
    print("  ⚠ WARNING: This is NOT using improvements!")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if not os.path.exists(improvement_file):
    print("❌ PROBLEM FOUND!")
    print("\nYour training used DQN_baseline.py, NOT DQN_improvement1.py!")
    print("That's why you saw no improvement - you just changed hyperparameters")
    print("on the same baseline agent.")
    print("\n📝 What happened:")
    print("  1. train_with_config.py tried to import DQN_improvement1")
    print("  2. File doesn't exist → import failed")
    print("  3. Fell back to DQN_baseline.py")
    print("  4. Only normalization was applied (via wrapper)")
    print("  5. No custom CNN, no architectural improvements")
    print("\n✅ SOLUTION:")
    print("  1. Save DQN_improvement1.py to src/agents/")
    print("  2. Re-run training")
    print("  3. Should see much better results!")
else:
    print("✓ DQN_improvement1.py exists")
    print("\nIf results are still poor, the issue might be:")
    print("  1. Not enough training time (500K might be too little)")
    print("  2. Seed variation (try different seed)")
    print("  3. Need more improvements (reward shaping, curriculum learning)")

print("="*70)