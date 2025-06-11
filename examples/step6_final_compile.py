# examples/step6_final_compile.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from neural_compiler import NeuralCompiler
from architecture_generator import ArchitectureGenerator
from task_analyzer import TaskAnalyzer
import torch

def main():
    # Build and optimize a model (simplified)
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y)

    arch_gen = ArchitectureGenerator()
    architectures = arch_gen.generate_candidates(task_chars, num_candidates=2)
    model = arch_gen.build_model(architectures[0])

    # Final compilation
    compiler = NeuralCompiler()
    performance_requirements = {
        "accuracy_threshold": 0.85,
        "inference_time_ms": 10
    }

    compiled_model = compiler.compile(model, task_chars, performance_requirements)
    print("Model compilation completed!")

    # Test the compiled model
    compiled_model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 10)
        output = compiled_model(test_input)
        print(f"Test output shape: {output.shape}")
        
if __name__ == "__main__":
    main()