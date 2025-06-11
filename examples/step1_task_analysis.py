# examples/step1_task_analysis.py

import sys
import os

# Ensure components directory is on the path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "components"))
if components_dir not in sys.path:
    sys.path.append(components_dir)

from task_analyzer import TaskAnalyzer
import torch

def main():
    # Create sample data
    X = torch.randn(500, 10)  # 500 samples, 10 features
    y = torch.randint(0, 3, (500,))  # 3-class classification

    # Initialize analyzer
    analyzer = TaskAnalyzer()

    # Analyze the task
    task_chars = analyzer.analyze_task(
        X, y,
        "Multi-class classification with 20 input features"
    )

    print("Task Analysis Results:")
    print(f"Task Type: {task_chars.task_type.value}")
    print(f"Input Shape: {task_chars.input_shape}")
    print(f"Dataset Size: {task_chars.dataset_size}")
    print(f"Num Features: {task_chars.num_features}")
    print(f"Num Classes: {task_chars.num_classes}")
    print(f"Data Complexity: {task_chars.data_complexity.value}")
    print(f"Suggested Depth: {task_chars.suggested_architecture_depth}")
    print(f"Suggested Width: {task_chars.suggested_width}")
    print(f"Recommended Activation: {task_chars.activation_recommendation}")
    print(f"Recommended Loss: {task_chars.loss_function_recommendation}")

if __name__ == "__main__":
    main()
