# examples/step5_automl.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from automl_optimizer import AutoMLOptimizer, OptimizationConfig
from architecture_generator import ArchitectureGenerator
from task_analyzer import TaskAnalyzer
import torch

def main():
    # Prepare data and models
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y)

    arch_gen = ArchitectureGenerator()
    architectures = arch_gen.generate_candidates(task_chars, num_candidates=2)
    models = [arch_gen.build_model(arch) for arch in architectures]

    # AutoML optimization
    opt_config = OptimizationConfig(
        max_trials=10,
        timeout_seconds=300,
        primary_metric="accuracy"
    )
    automl = AutoMLOptimizer(opt_config)
    print("Running AutoML optimization...")

    best_model, best_config = automl.optimize(models, X, y, task_chars)
    print(f"Best model found with config: {best_config}")

if __name__ == "__main__":
    main()
