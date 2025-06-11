#!/usr/bin/env python3
"""
--- Updated main.py ---
Main execution script for the Dynamic Neural Network Compiler
"""

import torch
import sys
import os

# Ensure components directory is on the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
components_dir = os.path.join(current_dir, "components")
if components_dir not in sys.path:
    sys.path.append(components_dir)

from task_analyzer import TaskAnalyzer
from architecture_generator import ArchitectureGenerator
from meta_learner import MetaLearner
from model_compressor import ModelCompressor, CompressionConfig
from automl_optimizer import AutoMLOptimizer, OptimizationConfig
from neural_compiler import NeuralCompiler


def create_sample_data() -> dict:
    """Create sample datasets for testing."""
    datasets = {
        "classification": {
            "X": torch.randn(500, 10),
            "y": torch.randint(0, 3, (500,)),
            "description": "Multi-class classification with 10 features",
        },
        "regression": {
            "X": torch.randn(800, 15),
            "y": torch.randn(800, 1),
            "description": "Regression task with 15 features",
        },
    }
    return datasets


def run_pipeline(
    X: torch.Tensor,
    y: torch.Tensor,
    description: str = "",
    step_by_step: bool = True,
) -> torch.nn.Module:
    """Run the complete pipeline end-to-end."""

    print(f"\n{'=' * 60}")
    print(f"RUNNING PIPELINE: {description}")
    print(f"{'=' * 60}")

    # Step 1: Task Analysis
    if step_by_step:
        input("Press Enter to run Step 1: Task Analysis...")
    print("\n🔍 Step 1: Analyzing Task...")
    analyzer = TaskAnalyzer()
    task_chars = analyzer.analyze_task(X, y, description)
    print(f"✅ Task Type: {task_chars.task_type.value}")
    print(
        f"✅ Suggested Architecture: depth={task_chars.suggested_architecture_depth}, width={task_chars.suggested_width}"
    )

    # Step 2: Architecture Generation
    if step_by_step:
        input("Press Enter to run Step 2: Architecture Generation...")
    print("\n🏗️ Step 2: Generating Architectures...")
    arch_gen = ArchitectureGenerator()
    candidates = arch_gen.generate_candidates(task_chars, num_candidates=2)
    models = [arch_gen.build_model(template) for template in candidates[:3]]
    print(f"✅ Generated {len(models)} architecture candidates")

    # Step 3: Meta-Learning
    if step_by_step:
        input("Press Enter to run Step 3: Meta-Learning...")
    print("\n🧠 Step 3: Applying Meta-Learning...")
    meta_learner = MetaLearner()
    meta_models = []
    for model in models:
        meta_model = meta_learner.initialize_from_similar_tasks(model, task_chars.__dict__)
        meta_models.append(meta_model)
    print(f"✅ Applied meta-learning to {len(meta_models)} models")

    # Step 4: AutoML Optimization
    if step_by_step:
        input("Press Enter to run Step 4: AutoML Optimization...")
    print("\n⚡ Step 4: AutoML Optimization...")
    opt_config = OptimizationConfig(max_trials=5, timeout_seconds=300, primary_metric="accuracy")
    automl = AutoMLOptimizer(opt_config)
    best_model, best_config = automl.optimize(meta_models, X, y, task_chars)
    print(f"✅ Found best model with config: {best_config}")

    # Step 5: Model Compression
    if step_by_step:
        input("Press Enter to run Step 5: Model Compression...")
    print("\n📦 Step 5: Model Compression...")
    compressor = ModelCompressor()
    compression_cfg = CompressionConfig(
        target_compression_ratio=0.5, enable_pruning=True, enable_quantization=False, fine_tune_epochs=2
    )
    compressed_model = compressor.compress(best_model, X, y, compression_cfg)
    stats = compressor.get_compression_stats()
    print(f"✅ Compression ratio: {stats['compression_ratio']:.2f}")

    # Step 6: Final Compilation
    if step_by_step:
        input("Press Enter to run Step 6: Final Compilation...")
    print("\n🎯 Step 6: Final Compilation...")
    compiler = NeuralCompiler()
    performance_req = {"accuracy_threshold": 0.8, "inference_time_ms": 10}
    final_model = compiler.compile(compressed_model, task_chars, performance_req)
    print("✅ Final model compilation completed!")

    # Test the compiled model
    print("\n🧪 Testing compiled model...")
    final_model.eval()
    with torch.no_grad():
        test_input = X[:5]
        output = final_model(test_input)
        print(f"✅ Model works! Output shape: {output.shape}")

    return final_model, task_chars


def main() -> None:
    """Main execution function."""
    print("🚀 Dynamic Neural Network Compiler")
    print("Choose execution mode:")
    print("1. Step-by-step (interactive)")
    print("2. Automatic (run all steps)")
    print("3. Run single component")

    choice = input("Enter choice (1-3): ").strip()
    datasets = create_sample_data()

    if choice == "1":
        print("\nAvailable datasets:")
        for i, (name, data) in enumerate(datasets.items(), 1):
            print(f"{i}. {name}: {data['description']}")
        ds_choice = input("Choose dataset (1-2): ").strip()
        ds_key = list(datasets.keys())[int(ds_choice) - 1]
        ds = datasets[ds_key]
        run_pipeline(ds["X"], ds["y"], ds["description"], step_by_step=True)

    elif choice == "2":
        for name, ds in datasets.items():
            run_pipeline(ds["X"], ds["y"], ds["description"], step_by_step=False)

    elif choice == "3":
        print("\nAvailable components:")
        components = [
            "Task Analyzer",
            "Architecture Generator",
            "Meta Learner",
            "Model Compressor",
            "AutoML Optimizer",
            "Neural Compiler",
        ]
        for i, comp in enumerate(components, 1):
            print(f"{i}. {comp}")
        comp_choice = input("Choose component (1-6): ").strip()
        comp = components[int(comp_choice) - 1]
        print(f"\nRunning {comp}...")

        if comp == "Task Analyzer":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            res = analyzer.analyze_task(X, y, "Demo classification")
            print(f"Task Type: {res.task_type.value}")
            print(f"Suggested Depth: {res.suggested_architecture_depth}")
            print(f"Suggested Width: {res.suggested_width}")

        elif comp == "Architecture Generator":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            task_chars = analyzer.analyze_task(X, y)
            arch_gen = ArchitectureGenerator()
            cand = arch_gen.generate_candidates(task_chars, num_candidates=2)
            print(f"Generated {len(cand)} templates")

        elif comp == "Meta Learner":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            task_chars = analyzer.analyze_task(X, y)
            arch_gen = ArchitectureGenerator()
            templates = arch_gen.generate_candidates(task_chars, num_candidates=1)
            model = arch_gen.build_model_from_template(templates[0])
            meta = MetaLearner()
            _ = meta.initialize_from_similar_tasks(model, task_chars.__dict__)
            print("Meta-learning initialization done")

        elif comp == "Model Compressor":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            task_chars = analyzer.analyze_task(X, y)
            arch_gen = ArchitectureGenerator()
            tmpl = arch_gen.generate_candidates(task_chars, num_candidates=1)[0]
            model = arch_gen.build_model_from_template(tmpl)
            print(f"Original params: {sum(p.numel() for p in model.parameters())}")
            compressor = ModelCompressor()
            cfg = CompressionConfig(target_compression_ratio=0.5, enable_pruning=True, enable_quantization=False, fine_tune_epochs=2)
            compressed = compressor.compress(model, X, y, cfg)
            print(f"Compressed params: {sum(p.numel() for p in compressed.parameters())}")

        elif comp == "AutoML Optimizer":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            task_chars = analyzer.analyze_task(X, y)
            arch_gen = ArchitectureGenerator()
            templates = arch_gen.generate_candidates(task_chars, num_candidates=2)
            models = [arch_gen.build_model_from_template(t) for t in templates]
            optimizer = AutoMLOptimizer(OptimizationConfig(max_trials=5, timeout_seconds=60))
            best_model, best_cfg = optimizer.optimize(models, X, y, task_chars)
            print(f"Best config: {best_cfg}")

        elif comp == "Neural Compiler":
            X = torch.randn(500, 10)
            y = torch.randint(0, 3, (500,))
            analyzer = TaskAnalyzer()
            task_chars = analyzer.analyze_task(X, y)
            arch_gen = ArchitectureGenerator()
            template = arch_gen.generate_candidates(task_chars, num_candidates=1)[0]
            model = arch_gen.build_model_from_template(template)
            compiler = NeuralCompiler()
            perf_req = {"inference_time_ms": 20}
            compiled = compiler.compile(model, task_chars, perf_req)
            print("Compiled model ready")

        else:
            print("Invalid component choice!")

    else:
        print("Invalid choice! Exiting.")
        return

    print("\n🎉 Execution completed!")


if __name__ == "__main__":
    main()
