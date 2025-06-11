# examples/step2_architecture_gen.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from architecture_generator import ArchitectureGenerator
from task_analyzer import TaskAnalyzer
import torch

def main():
    # Use results from Step 1
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y)

    # Generate architectures
    arch_gen = ArchitectureGenerator()
    architectures = arch_gen.generate_candidates(task_chars, num_candidates=2)

    print(f"Generated {len(architectures)} architecture candidates:")
    for i, arch in enumerate(architectures):
        print(f"Architecture {i + 1}: type={arch.architecture_type}, layers={len(arch.layers)}")
        model = arch_gen.build_model(arch)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

if __name__ == "__main__":
    main()
