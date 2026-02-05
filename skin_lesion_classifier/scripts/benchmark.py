"""
Performance Benchmarking Script for Skin Lesion Classifier.

This script helps profile training performance and identify bottlenecks.
It measures:
- Data loading speed
- Forward pass speed
- Backward pass speed
- Memory usage
- Overall training throughput
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_and_split_data, create_dataloaders
from src.models.efficientnet import create_model


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    return f"{seconds:.3f}s"


def format_memory(bytes: int) -> str:
    """Format memory in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024
    return f"{bytes:.2f}TB"


def benchmark_data_loading(
    loader: torch.utils.data.DataLoader,
    num_batches: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Benchmark data loading speed."""
    print(f"\n📊 Benchmarking data loading ({num_batches} batches)...")
    
    times = []
    total_samples = 0
    
    # Warmup
    for i, (images, targets) in enumerate(loader):
        if i >= 5:
            break
    
    # Actual benchmark
    for i, (images, targets) in enumerate(tqdm(loader, total=num_batches, desc="Loading")):
        start = time.perf_counter()
        images = images.to(device)
        targets = targets.to(device)
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        
        times.append(end - start)
        total_samples += len(images)
        
        if i >= num_batches - 1:
            break
    
    avg_time = sum(times) / len(times)
    throughput = total_samples / sum(times)
    
    print(f"  ✓ Avg batch time: {format_time(avg_time)}")
    print(f"  ✓ Throughput: {throughput:.1f} samples/sec")
    
    return {
        "avg_batch_time": avg_time,
        "throughput": throughput,
    }


def benchmark_forward_pass(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 50,
) -> Dict[str, float]:
    """Benchmark forward pass speed."""
    print(f"\n🚀 Benchmarking forward pass ({num_batches} batches)...")
    
    model.eval()
    times = []
    total_samples = 0
    
    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            _ = model(images)
            if i >= 5:
                break
    
    # Actual benchmark
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, total=num_batches, desc="Forward")):
            images = images.to(device)
            
            start = time.perf_counter()
            _ = model(images)
            torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter()
            
            times.append(end - start)
            total_samples += len(images)
            
            if i >= num_batches - 1:
                break
    
    avg_time = sum(times) / len(times)
    throughput = total_samples / sum(times)
    
    print(f"  ✓ Avg forward time: {format_time(avg_time)}")
    print(f"  ✓ Throughput: {throughput:.1f} samples/sec")
    
    return {
        "avg_forward_time": avg_time,
        "throughput": throughput,
    }


def benchmark_training_step(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_batches: int = 50,
) -> Dict[str, float]:
    """Benchmark full training step (forward + backward + optimizer)."""
    print(f"\n🔥 Benchmarking training step ({num_batches} batches)...")
    
    model.train()
    times = {
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }
    total_samples = 0
    
    # Warmup
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i >= 5:
            break
    
    # Actual benchmark
    for i, (images, targets) in enumerate(tqdm(loader, total=num_batches, desc="Training")):
        images = images.to(device)
        targets = targets.to(device)
        
        step_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Forward pass
        forward_start = time.perf_counter()
        outputs = model(images)
        loss = criterion(outputs, targets)
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
        forward_end = time.perf_counter()
        
        # Backward pass
        backward_start = time.perf_counter()
        loss.backward()
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
        backward_end = time.perf_counter()
        
        # Optimizer step
        opt_start = time.perf_counter()
        optimizer.step()
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None
        opt_end = time.perf_counter()
        
        step_end = time.perf_counter()
        
        times["forward"].append(forward_end - forward_start)
        times["backward"].append(backward_end - backward_start)
        times["optimizer"].append(opt_end - opt_start)
        times["total"].append(step_end - step_start)
        total_samples += len(images)
        
        if i >= num_batches - 1:
            break
    
    avg_forward = sum(times["forward"]) / len(times["forward"])
    avg_backward = sum(times["backward"]) / len(times["backward"])
    avg_optimizer = sum(times["optimizer"]) / len(times["optimizer"])
    avg_total = sum(times["total"]) / len(times["total"])
    throughput = total_samples / sum(times["total"])
    
    print(f"  ✓ Avg forward time: {format_time(avg_forward)}")
    print(f"  ✓ Avg backward time: {format_time(avg_backward)}")
    print(f"  ✓ Avg optimizer time: {format_time(avg_optimizer)}")
    print(f"  ✓ Avg total time: {format_time(avg_total)}")
    print(f"  ✓ Throughput: {throughput:.1f} samples/sec")
    
    return {
        "avg_forward_time": avg_forward,
        "avg_backward_time": avg_backward,
        "avg_optimizer_time": avg_optimizer,
        "avg_total_time": avg_total,
        "throughput": throughput,
    }


def get_memory_usage(device: torch.device) -> Dict[str, int]:
    """Get current memory usage."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        return {
            "allocated": allocated,
            "reserved": reserved,
        }
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory()
        reserved = torch.mps.driver_allocated_memory()
        return {
            "allocated": allocated,
            "reserved": reserved,
        }
    return {"allocated": 0, "reserved": 0}


def main():
    parser = argparse.ArgumentParser(description="Benchmark training performance")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches to benchmark",
    )
    args = parser.parse_args()
    
    # Load config
    print("📋 Loading configuration...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎮 Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple MPS (M4)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    # Load data
    print("\n📁 Loading data...")
    data_config = config.get("data", {})
    train_config = config.get("training", {})
    
    train_df, val_df, test_df = load_and_split_data(
        labels_csv=Path(data_config.get("labels_csv")),
        images_dir=Path(data_config.get("images_dir")),
        val_size=data_config.get("val_size", 0.15),
        test_size=data_config.get("test_size", 0.15),
        random_state=42,
        lesion_aware=data_config.get("lesion_aware", True),
    )
    
    print(f"  ✓ Train samples: {len(train_df)}")
    
    # Create dataloaders
    print("\n🔄 Creating dataloaders...")
    batch_size = train_config.get("batch_size", 32)
    num_workers = train_config.get("num_workers", 4)
    
    train_loader, _, _ = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        images_dir=Path(data_config.get("images_dir")),
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=config.get("model", {}).get("image_size", 224),
        augmentation_strength=train_config.get("augmentation", "medium"),
        use_weighted_sampling=train_config.get("use_weighted_sampling", True),
        pin_memory=device.type == "cuda",
        prefetch_factor=train_config.get("prefetch_factor", 2),
        persistent_workers=train_config.get("persistent_workers", True) and num_workers > 0,
    )
    
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ Num workers: {num_workers}")
    print(f"  ✓ Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\n🤖 Creating model...")
    model_config = config.get("model", {})
    model = create_model(
        num_classes=model_config.get("num_classes", 7),
        model_size=model_config.get("size", "small"),
        pretrained=model_config.get("pretrained", True),
        dropout_rate=model_config.get("dropout_rate", 0.3),
        freeze_backbone=model_config.get("freeze_backbone", False),
        head_type=model_config.get("head_type", "simple"),
    )
    model = model.to(device)
    
    # Try torch.compile if available (skip for MPS due to backward pass issues)
    if train_config.get("use_torch_compile", True) and hasattr(torch, "compile") and device.type != "mps":
        print("  ⚡ Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode="default")
            print("  ✓ Model compiled successfully")
        except Exception as e:
            print(f"  ⚠️  Compilation failed: {e}")
    elif device.type == "mps":
        print("  ⚠️  Skipping torch.compile for MPS (known backward pass issues in benchmark)")
    
    print(f"  ✓ Total params: {model.get_total_params():,}")
    print(f"  ✓ Trainable params: {model.get_trainable_params():,}")
    
    # Create optimizer and criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Initial memory
    mem_before = get_memory_usage(device)
    print(f"\n💾 Initial memory:")
    print(f"  Allocated: {format_memory(mem_before['allocated'])}")
    print(f"  Reserved: {format_memory(mem_before['reserved'])}")
    
    # Run benchmarks
    print("\n" + "="*60)
    print("🔬 PERFORMANCE BENCHMARKS")
    print("="*60)
    
    data_results = benchmark_data_loading(train_loader, args.num_batches, device)
    forward_results = benchmark_forward_pass(model, train_loader, device, args.num_batches)
    training_results = benchmark_training_step(
        model, train_loader, criterion, optimizer, device, args.num_batches
    )
    
    # Final memory
    mem_after = get_memory_usage(device)
    print(f"\n💾 Peak memory:")
    print(f"  Allocated: {format_memory(mem_after['allocated'])}")
    print(f"  Reserved: {format_memory(mem_after['reserved'])}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Data loading: {format_time(data_results['avg_batch_time'])} per batch")
    print(f"Forward pass: {format_time(forward_results['avg_forward_time'])} per batch")
    print(f"Training step: {format_time(training_results['avg_total_time'])} per batch")
    print(f"  └─ Forward: {format_time(training_results['avg_forward_time'])}")
    print(f"  └─ Backward: {format_time(training_results['avg_backward_time'])}")
    print(f"  └─ Optimizer: {format_time(training_results['avg_optimizer_time'])}")
    print(f"\nTraining throughput: {training_results['throughput']:.1f} samples/sec")
    
    # Estimate epoch time
    batches_per_epoch = len(train_loader)
    epoch_time = training_results['avg_total_time'] * batches_per_epoch
    print(f"Estimated epoch time: {epoch_time/60:.1f} minutes")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
