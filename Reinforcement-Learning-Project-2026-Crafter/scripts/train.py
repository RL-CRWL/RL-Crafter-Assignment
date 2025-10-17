"""
Training script using configuration files optimized for GTX 750 Ti

Usage:
    python scripts/train.py --config DQN_CONFIG
    python scripts/train.py --config DQN_IMPROVEMENT_1
    python scripts/train.py --config QUICK_TEST_CONFIG
"""

import argparse
import os
import sys
import torch
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.DQN_baseline import DQNAgent
from configs.configs import get_config, print_config, estimate_memory_usage


def monitor_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"\nGPU Memory Status:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Usage:     {(allocated/total)*100:.1f}%")
        
        if allocated / total > 0.9:
            print("  ⚠ WARNING: High memory usage! Consider reducing batch_size or buffer_size")


def main():
    parser = argparse.ArgumentParser(description='Train DQN on Crafter with predefined configs')
    parser.add_argument('--config', type=str, default='DQN_CONFIG',
                       choices=['DQN_CONFIG', 'DQN_IMPROVEMENT_1', 'DQN_IMPROVEMENT_2', 
                               'QUICK_TEST_CONFIG'],
                       help='Configuration to use')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (default: results/<config_name>)')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU training even if CUDA is available')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override device if forcing CPU
    if args.force_cpu:
        config['device'] = 'cpu'
        print("\n⚠ Forcing CPU training (this will be slower)")
    
    # Set save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.save_dir = f"results/{config['name']}_{timestamp}"
    
    # Print configuration
    print_config(config)
    
    # Estimate memory
    est_memory = estimate_memory_usage(config)
    print(f"Estimated GPU Memory: {est_memory:.2f} GB")
    if est_memory > 1.8:
        print("⚠ This configuration may push memory limits on GTX 750 Ti")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting...")
            return
    
    # Check GPU availability
    print("\n" + "="*70)
    print("HARDWARE CHECK")
    print("="*70)
    if torch.cuda.is_available():
        print(f"✓ CUDA Available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available - training on CPU")
        if config['device'] == 'cuda':
            print("  Switching to CPU...")
            config['device'] = 'cpu'
    
    # Create agent
    print("\n" + "="*70)
    print("CREATING AGENT")
    print("="*70)
    
    agent = DQNAgent(
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
        device=config['device'],
        seed=config['seed']
    )
    
    # Monitor memory before training
    if config['device'] == 'cuda':
        monitor_gpu_memory()
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Save directory: {args.save_dir}")
    print(f"Total timesteps: {config['total_timesteps']:,}")
    print(f"Evaluation frequency: {config['eval_freq']:,}")
    print("="*70 + "\n")
    
    try:
        agent.train(
            total_timesteps=config['total_timesteps'],
            eval_freq=config['eval_freq'],
            save_dir=args.save_dir
        )
        
        # Monitor memory after training
        if config['device'] == 'cuda':
            monitor_gpu_memory()
        
        # Final evaluation
        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)
        metrics = agent.evaluate(n_episodes=10)
        
        # Save metrics
        import json
        metrics_path = os.path.join(args.save_dir, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert any non-serializable values
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool, list, dict)):
                    serializable_metrics[k] = v
                else:
                    serializable_metrics[k] = str(v)
            json.dump(serializable_metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*70)
            print("ERROR: OUT OF MEMORY")
            print("="*70)
            print("Your GPU ran out of memory. Try these fixes:")
            print("1. Reduce batch_size (currently: {})".format(config['batch_size']))
            print("2. Reduce buffer_size (currently: {})".format(config['buffer_size']))
            print("3. Use CPU instead: --force_cpu")
            print("="*70)
            
            if config['device'] == 'cuda':
                monitor_gpu_memory()
        else:
            raise
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.save_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()