# examples/step4_compression.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from model_compressor import ModelCompressor, CompressionConfig
from architecture_generator import ArchitectureGenerator
from task_analyzer import TaskAnalyzer
import torch

def main():
    # Build a model first
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y)

    arch_gen = ArchitectureGenerator()
    architectures = arch_gen.generate_candidates(task_chars, num_candidates=2)
    model = arch_gen.build_model(architectures[0])

    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")

    # Compress the model
    compressor = ModelCompressor()
    compression_config = CompressionConfig(
        target_compression_ratio=0.5,
        enable_pruning=True,
        enable_quantization=False,
        fine_tune_epochs=2
    )

    compressed_model = compressor.compress(model, X, y, compression_config)
    print(f"Compressed model parameters: {sum(p.numel() for p in compressed_model.parameters())}")

    # Get compression stats
    stats = compressor.get_compression_stats()
    print(f"Compression ratio achieved: {stats['compression_ratio']:.2f}")
    print(f"Size reduction: {stats['size_reduction_percent']:.1f}%")
    print(f"Original accuracy: {stats['original_accuracy']:.4f}")
    print(f"Compressed accuracy: {stats['compressed_accuracy']:.4f}")

if __name__ == "__main__":
    main()
