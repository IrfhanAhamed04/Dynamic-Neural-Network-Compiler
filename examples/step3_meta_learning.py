# examples/step3_meta_learning.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from meta_learner import MetaLearner
from architecture_generator import ArchitectureGenerator
from task_analyzer import TaskAnalyzer
import torch

def main():
    # Previous steps
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y)

    arch_gen = ArchitectureGenerator()
    architectures = arch_gen.generate_candidates(task_chars, num_candidates=2)

    # Meta-learning initialization
    meta_learner = MetaLearner()

    print("Applying meta-learning to architectures...")
    for i, arch in enumerate(architectures):
        model = arch_gen.build_model(arch)
        meta_model = meta_learner.initialize_from_similar_tasks(model, task_chars.__dict__)
        print(f"Architecture {i + 1} initialized with meta-learning")

if __name__ == "__main__":
    main()
