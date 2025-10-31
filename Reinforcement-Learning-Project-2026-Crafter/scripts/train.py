"""
FIXED Training script for all DQN agents

Usage:
    python train.py --config DQN_CONFIG              # Baseline
    python train.py --config DQN_IMPROVEMENT_1       # Improvement 1
    python train.py --config DQN_IMPROVEMENT_2       # Improvement 2
    python train.py --config QUICK_TEST_CONFIG       # Quick test
"""

import argparse
import os
import sys
import torch
import json
from datetime import datetime
import numpy as np 

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import configs
from configs.configs import get_config, print_config, estimate_memory_usage

# Import all agents
print("🔄 Importing agents...")
try:
    from src.agents.DQN_baseline import DQNAgent
    print("  ✓ DQNAgent (baseline)")
    HAS_BASELINE = True
except ImportError as e:
    print(f"  ✗ DQNAgent failed: {e}")
    HAS_BASELINE = False

try:
    # FIXED: Import from DQN_improv1.py (the file that actually exists)
    from src.agents.DQN_improv1 import DQNImprovement1Simplified
    print("  ✓ DQNImprovement1Simplified")
    HAS_IMPROVEMENT1 = True
except ImportError as e:
    print(f"  ✗ DQNImprovement1Simplified failed: {e}")
    HAS_IMPROVEMENT1 = False

try:
    from src.agents.DQN_FRAMESTACK import DQNImprovement2FrameStack
    print("  ✓ DQNImprovement2FrameStack")
    HAS_IMPROVEMENT2 = True
except ImportError as e:
    print(f"  ✗ DQNImprovement2FrameStack failed: {e}")
    HAS_IMPROVEMENT2 = False

try:
    from src.agents.DQN_DUEL import DQNImprovement3DuelCNN
    print("  ✓ DQNImprovement3Duel")
    HAS_IMPROVEMENT3 = True
except ImportError as e:
    print(f"  ✗ DQNImprovement3Duel failed: {e}")
    HAS_IMPROVEMENT3 = False


def create_agent(config):
    """Create appropriate agent based on config"""
    config_name = config['name'].lower()
    
    if 'improvement3' in config_name:
        if not HAS_IMPROVEMENT3:
            raise ImportError("❌ DQN_improv2.py not found!")
        
        print("🎬 Creating DQN Improvement 2 (Frame Stacking) agent...")
        
        # Import the frame stacking agent
        from src.agents.DQN_DUEL import DQNImprovement3DuelCNN
        
        return DQNImprovement3DuelCNN(
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            target_update_interval=config.get('target_update_interval', 10000),
            exploration_fraction=config['exploration_fraction'],
            exploration_initial_eps=config['exploration_initial_eps'],
            exploration_final_eps=config['exploration_final_eps'],
            train_freq=config.get('train_freq', 4),
            gradient_steps=config.get('gradient_steps', 1),
            n_stack=config.get('n_stack', 4),
            device=config['device'],
            seed=config['seed']
        )
    
    # Determine which agent to create
    elif 'improvement2' in config_name or config.get('use_frame_stacking', False):
        if not HAS_IMPROVEMENT2:
            raise ImportError("❌ DQN_improv2.py not found!")
        
        print("🎬 Creating DQN Improvement 2 (Frame Stacking) agent...")
        
        # Import the frame stacking agent
        from src.agents.DQN_FRAMESTACK import DQNImprovement2FrameStack
        
        return DQNImprovement2FrameStack(
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            target_update_interval=config.get('target_update_interval', 10000),
            exploration_fraction=config['exploration_fraction'],
            exploration_initial_eps=config['exploration_initial_eps'],
            exploration_final_eps=config['exploration_final_eps'],
            train_freq=config.get('train_freq', 4),
            gradient_steps=config.get('gradient_steps', 1),
            n_stack=config.get('n_stack', 4),
            device=config['device'],
            seed=config['seed']
        )
    
    elif 'improvement1' in config_name or config.get('use_custom_cnn', False):
        if not HAS_IMPROVEMENT1:
            raise ImportError("❌ DQN_improv1.py not found! Please add it to src/agents/")
        
        print("🔧 Creating DQN Improvement 1 (Simplified) agent...")
        return DQNImprovement1Simplified(
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            target_update_interval=config.get('target_update_interval', 5000),
            exploration_fraction=config['exploration_fraction'],
            exploration_initial_eps=config['exploration_initial_eps'],
            exploration_final_eps=config['exploration_final_eps'],
            train_freq=config.get('train_freq', 4),
            gradient_steps=config.get('gradient_steps', 1),
            device=config['device'],
            seed=config['seed']
        )
    
    else:  # Baseline
        if not HAS_BASELINE:
            raise ImportError("❌ DQN_baseline.py not found! Please add it to src/agents/")
        
        print("📦 Creating DQN Baseline agent...")
        return DQNAgent(
            learning_rate=config['learning_rate'],
            buffer_size=config['buffer_size'],
            learning_starts=config['learning_starts'],
            batch_size=config['batch_size'],
            gamma=config['gamma'],
            target_update_interval=config.get('target_update_interval', 5000),
            exploration_fraction=config['exploration_fraction'],
            exploration_initial_eps=config['exploration_initial_eps'],
            exploration_final_eps=config['exploration_final_eps'],
            train_freq=config.get('train_freq', 4),
            gradient_steps=config.get('gradient_steps', 1),
            device=config['device'],
            seed=config['seed']
        )


def monitor_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"\n💾 GPU Memory Status:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Usage:     {(allocated/total)*100:.1f}%")
        
        if allocated / total > 0.9:
            print("  ⚠️  WARNING: High memory usage! Consider reducing batch_size or buffer_size")
        
        return allocated, total
    return None, None


def save_config_to_file(config, save_dir):
    """Save configuration to JSON file"""
    config_path = os.path.join(save_dir, 'config.json')
    
    # Make serializable (convert any non-JSON types)
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool, list, dict, type(None))):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"💾 Configuration saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train DQN agents on Crafter environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline
  python train.py --config DQN_CONFIG
  
  # Train improvement 1
  python train.py --config DQN_IMPROVEMENT_1
  
  # Train improvement 2
  python train.py --config DQN_IMPROVEMENT_2
  
  # Quick test (10k steps)
  python train.py --config QUICK_TEST_CONFIG
  
  # Force CPU training
  python train.py --config DQN_CONFIG --force_cpu
  
  # Custom save directory
  python train.py --config DQN_CONFIG --save_dir my_results
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='DQN_CONFIG',
        choices=['DQN_CONFIG', 'DQN_IMPROVEMENT_1', 'DQN_IMPROVEMENT_2','DQN_IMPROVEMENT_3', 'QUICK_TEST_CONFIG'],
        help='Configuration to use'
    )
    
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default=None,
        help='Directory to save results (default: results/<config_name>_<timestamp>)'
    )
    
    parser.add_argument(
        '--force_cpu', 
        action='store_true',
        help='Force CPU training even if CUDA is available'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override total timesteps (default: use config value)'
    )
    
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=None,
        help='Override evaluation frequency (default: use config value)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎮 CRAFTER DQN TRAINING SCRIPT")
    print("="*70)
    
    # Load configuration
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nAvailable configs: DQN_CONFIG, DQN_IMPROVEMENT_1, DQN_IMPROVEMENT_2, QUICK_TEST_CONFIG")
        return
    
    # Override config values if specified
    if args.timesteps is not None:
        config['total_timesteps'] = args.timesteps
        print(f"⚙️  Overriding timesteps: {args.timesteps:,}")
    
    if args.eval_freq is not None:
        config['eval_freq'] = args.eval_freq
        print(f"⚙️  Overriding eval_freq: {args.eval_freq:,}")
    
    # Override device if forcing CPU
    if args.force_cpu:
        config['device'] = 'cpu'
        print("\n⚠️  Forcing CPU training (this will be slower)")
    
    # Set save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_name_clean = config['name'].replace(' ', '_').replace('/', '_')
        args.save_dir = f"results/{config_name_clean}_{timestamp}"
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print_config(config)
    
    # Estimate memory
    est_memory = estimate_memory_usage(config)
    print(f"📊 Estimated GPU Memory: {est_memory:.2f} GB")
    if est_memory > 1.8:
        print("⚠️  This configuration may push memory limits on GTX 750 Ti (2GB)")
        if not args.force_cpu:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborting...")
                return
    
    # Check GPU availability
    print("\n" + "="*70)
    print("🖥️  HARDWARE CHECK")
    print("="*70)
    if torch.cuda.is_available():
        print(f"✅ CUDA Available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available - training on CPU")
        if config['device'] == 'cuda':
            print("  Switching to CPU...")
            config['device'] = 'cpu'
    
    # Create agent
    print("\n" + "="*70)
    print("🤖 CREATING AGENT")
    print("="*70)
    
    try:
        agent = create_agent(config)
    except ImportError as e:
        print(f"\n❌ Error creating agent: {e}")
        print("\nMake sure the agent file exists in src/agents/")
        print("Expected files:")
        print("  - src/agents/DQN_baseline.py")
        print("  - src/agents/DQN_improv1.py")
        print("  - src/agents/DQN_improv2.py")
        return
    except Exception as e:
        print(f"\n❌ Error creating agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Monitor memory before training
    if config['device'] == 'cuda':
        monitor_gpu_memory()
    
    # Save configuration
    save_config_to_file(config, args.save_dir)
    
    # Train
    print("\n" + "="*70)
    print("🏋️  STARTING TRAINING")
    print("="*70)
    print(f"📁 Save directory: {args.save_dir}")
    print(f"⏱️  Total timesteps: {config['total_timesteps']:,}")
    print(f"📊 Evaluation frequency: {config['eval_freq']:,}")
    print(f"🎲 Random seed: {config['seed']}")
    print("="*70 + "\n")
    
    try:
        # Train the agent
        agent.train(
            total_timesteps=config['total_timesteps'],
            eval_freq=config['eval_freq'],
            save_dir=args.save_dir
        )
        
        # Monitor memory after training
        if config['device'] == 'cuda':
            print("\n📊 Post-training GPU status:")
            monitor_gpu_memory()
        
        # Final evaluation
        print("\n" + "="*70)
        print("🎯 FINAL EVALUATION")
        print("="*70)
        
        metrics = agent.evaluate(n_episodes=10)
        
        # Save final metrics
        metrics_path = os.path.join(args.save_dir, 'final_metrics.json')
        
        # Make metrics serializable
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_metrics[k] = v
            elif isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = str(v)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n💾 Final metrics saved to: {metrics_path}")
        
        # Print success summary
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"📁 All results saved to: {args.save_dir}")
        print("\n📦 Files created:")
        print(f"  • Model: {args.save_dir}/model.zip")
        print(f"  • Metrics: {args.save_dir}/final_metrics.json")
        print(f"  • Config: {args.save_dir}/config.json")
        print(f"  • Plots: {args.save_dir}/*.png")
        print("="*70 + "\n")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*70)
            print("❌ ERROR: OUT OF MEMORY")
            print("="*70)
            print("Your GPU ran out of memory. Try these fixes:")
            print(f"1. Reduce batch_size (currently: {config['batch_size']})")
            print(f"2. Reduce buffer_size (currently: {config['buffer_size']})")
            print("3. Use CPU instead: python train.py --config <CONFIG> --force_cpu")
            print("\nExample with smaller memory:")
            print(f"  python train.py --config QUICK_TEST_CONFIG")
            print("="*70)
            
            if config['device'] == 'cuda':
                monitor_gpu_memory()
        else:
            print(f"\n❌ Runtime Error: {e}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        print(f"Partial results saved to: {args.save_dir}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()