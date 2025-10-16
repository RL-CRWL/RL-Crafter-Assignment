"""
Check GPU availability and specifications
"""
import torch
import sys

print("="*60)
print("GPU AVAILABILITY CHECK")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi Processor Count: {props.multi_processor_count}")
    
    # Current device
    print(f"\nCurrent Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Test tensor operations
    print("\nTesting GPU tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU tensor operations working!")
        
        # Memory usage
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
    
    # Recommendation
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if props.total_memory < 2e9:  # Less than 2GB
        print("⚠ Your GPU has limited memory (<2GB).")
        print("  Consider:")
        print("  - Using smaller batch sizes")
        print("  - Using CPU for training if GPU is too slow")
        print("  - Reducing buffer size")
    elif props.total_memory < 4e9:  # Less than 4GB
        print("⚠ Your GPU has moderate memory (<4GB).")
        print("  Should work, but use conservative hyperparameters:")
        print("  - batch_size=32 or lower")
        print("  - buffer_size=50000-100000")
    else:
        print("✓ Your GPU should handle training well!")
        print("  You can use standard hyperparameters.")
    print("="*60)
else:
    print("\n⚠ No CUDA-capable GPU detected.")
    print("Training will use CPU, which will be significantly slower.")
    print("\nOptions:")
    print("1. Continue with CPU (expect slower training)")
    print("2. Use Google Colab for free GPU access")
    print("3. Use cloud computing services (AWS, Azure, GCP)")
    print("="*60)

# Python and PyTorch versions
print(f"\nPython Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda if cuda_available else 'N/A'}")
print("="*60)