"""
--- Updated model_compressor.py ---
Model Compressor Component
Implements various neural network compression techniques including pruning,
quantization, and knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import math


@dataclass
class CompressionConfig:
    """Configuration for model compression parameters."""
    target_compression_ratio: float = 0.5
    enable_pruning: bool = True
    enable_quantization: bool = False
    enable_knowledge_distillation: bool = True
    pruning_method: str = "magnitude"  # "magnitude", "structured", "gradual"
    quantization_bits: int = 8
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.7
    fine_tune_epochs: int = 2


class CompressionStats:
    """Statistics tracking for compression process."""

    def __init__(self):
        self.original_size: float = 0.0
        self.compressed_size: float = 0.0
        self.compression_ratio: float = 0.0
        self.original_accuracy: float = 0.0
        self.compressed_accuracy: float = 0.0
        self.accuracy_drop: float = 0.0
        self.inference_speedup: float = 0.0
        self.methods_applied: List[str] = []


class BasePruner(ABC):
    """Abstract base class for pruning methods."""

    @abstractmethod
    def compute_importance_scores(self, model: nn.Module, data: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute importance scores for model parameters."""
        pass

    @abstractmethod
    def apply_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """Apply pruning to the model."""
        pass


class MagnitudePruner(BasePruner):
    """Prunes parameters based on magnitude (L1/L2 norm)."""

    def __init__(self, norm_type: str = "l1"):
        self.norm_type = norm_type

    def compute_importance_scores(self, model: nn.Module, data: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute importance scores based on parameter magnitudes."""
        importance_scores: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim > 1:
                if self.norm_type == "l1":
                    scores = torch.abs(param)
                elif self.norm_type == "l2":
                    scores = param.pow(2)
                else:
                    scores = torch.abs(param)
                importance_scores[name] = scores
        return importance_scores

    def apply_pruning(self, model: nn.Module, sparsity_ratio: float) -> nn.Module:
        """Apply magnitude-based pruning."""
        importance_scores = self.compute_importance_scores(model, None)
        if not importance_scores:
            return model

        # Flatten all scores to find global threshold
        all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
        threshold = torch.quantile(all_scores, sparsity_ratio)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in importance_scores:
                    mask = importance_scores[name] > threshold
                    param.data.mul_(mask.float())

        return model


class GradualPruner(BasePruner):
    """Implements gradual pruning during training."""

    def __init__(self, initial_sparsity: float = 0.0, final_sparsity: float = 0.9,
                 pruning_frequency: int = 100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.pruning_frequency = pruning_frequency
        self.current_step = 0

    def compute_importance_scores(self, model: nn.Module, data: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute importance scores (same as magnitude pruner for simplicity)."""
        importance_scores: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.ndim > 1:
                importance_scores[name] = torch.abs(param)
        return importance_scores

    def apply_pruning(self, model: nn.Module, sparsity_ratio: float = None) -> nn.Module:
        """Apply gradual pruning based on current step."""
        if self.current_step % self.pruning_frequency != 0:
            self.current_step += 1
            return model

        progress = min(1.0, self.current_step / (10 * self.pruning_frequency))
        current_sparsity = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        pruner = MagnitudePruner()
        model = pruner.apply_pruning(model, current_sparsity)
        self.current_step += 1
        return model


class QuantizationHandler:
    """Handles model quantization operations."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.scale_factors: Dict[str, float] = {}
        self.zero_points: Dict[str, int] = {}

    def calibrate(self, model: nn.Module, calibration_data: torch.Tensor):
        """Calibrate quantization parameters using calibration data."""
        model.eval()
        activation_stats: Dict[str, Dict[str, float]] = {}
        hooks: List[Any] = []

        def activation_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = {
                        'min': float(output.min().item()),
                        'max': float(output.max().item()),
                        'mean': float(output.mean().item()),
                        'std': float(output.std().item())
                    }
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(activation_hook(name)))

        with torch.no_grad():
            for i in range(0, len(calibration_data), 32):
                batch = calibration_data[i:i + 32]
                model(batch)

        for hook in hooks:
            hook.remove()

        for name, stats in activation_stats.items():
            qmin, qmax = 0, (2 ** self.bits) - 1
            min_val, max_val = stats['min'], stats['max']
            scale = (max_val - min_val + 1e-8) / (qmax - qmin)
            zero_point = qmin - round(min_val / (scale + 1e-8))
            zero_point = int(max(qmin, min(qmax, zero_point)))
            self.scale_factors[name] = scale
            self.zero_points[name] = zero_point

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model parameters."""
        quantized_model = copy.deepcopy(model)
        for name, param in quantized_model.named_parameters():
            if param.requires_grad:
                qmin, qmax = -(2 ** (self.bits - 1)), (2 ** (self.bits - 1)) - 1
                min_val, max_val = float(param.min().item()), float(param.max().item())
                scale = (max_val - min_val + 1e-8) / (qmax - qmin)
                zero_point = qmin - round(min_val / (scale + 1e-8))
                zero_point = int(max(qmin, min(qmax, zero_point)))
                quantized = torch.clamp(torch.round(param / (scale + 1e-8) + zero_point), qmin, qmax)
                dequantized = scale * (quantized - zero_point)
                param.data.copy_(dequantized)
        return quantized_model


class KnowledgeDistiller:
    """Implements knowledge distillation for model compression."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha

    def create_student_model(self, teacher_model: nn.Module, compression_ratio: float) -> nn.Module:
        """
        Create a smaller student model based on the teacher, preserving input and output sizes.
        We build a simple 2-layer network: input → hidden → output.
        """
        # Find teacher’s first and last Linear layers to infer sizes
        input_size = None
        output_size = None
        for module in teacher_model.modules():
            if isinstance(module, nn.Linear):
                if input_size is None:
                    input_size = module.in_features
                output_size = module.out_features
        # Fallback defaults if we couldn't find them
        if input_size is None:
            input_size = 20  # Replace with your default input size
        if output_size is None:
            output_size = 3  # Replace with your default output size

        # Hidden layer size scaled by compression ratio
        hidden_size = max(1, int(input_size * compression_ratio))

        # Build a simple student network: Linear → ReLU → Linear
        student_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        return student_model

    def distill(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 10
    ) -> nn.Module:
        """Perform knowledge distillation training."""
        teacher_model.eval()
        student_model.train()
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.no_grad():
                    t_out = teacher_model(batch_x)
                    soft_targets = F.softmax(t_out / self.temperature, dim=1)

                s_out = student_model(batch_x)
                s_log_prob = F.log_softmax(s_out / self.temperature, dim=1)
                distill_loss = F.kl_div(s_log_prob, soft_targets, reduction='batchmean') * (self.temperature ** 2)

                if s_out.shape[1] == 1:
                    # Regression task
                    ce_loss = F.mse_loss(s_out.view(-1), batch_y.view(-1).float())
                else:
                    # Classification task
                    ce_loss = F.cross_entropy(s_out, batch_y.long())

                loss = self.alpha * distill_loss + (1 - self.alpha) * ce_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                print(f"Distillation epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        return student_model


class ModelCompressor:
    """Main model compression orchestrator."""

    def __init__(self):
        self.stats = CompressionStats()
        self.pruners: Dict[str, BasePruner] = {
            'magnitude': MagnitudePruner(),
            'gradual': GradualPruner()
        }
        self.quantizer = QuantizationHandler()
        self.distiller = KnowledgeDistiller()

    def compress(
        self,
        model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor,
        config: CompressionConfig
    ) -> nn.Module:
        """Apply compression techniques to the model."""
        print("🗜️ Starting model compression...")

        # Record original statistics
        self.stats.original_size = self._calculate_model_size(model)
        self.stats.original_accuracy = self._evaluate_model(model, data, labels)

        compressed_model = copy.deepcopy(model)

        # Knowledge Distillation
        if config.enable_knowledge_distillation:
            print("📚 Applying knowledge distillation...")
            student = self.distiller.create_student_model(
                compressed_model, config.target_compression_ratio
            )
            compressed_model = self.distiller.distill(
                compressed_model, student, data, labels, epochs=config.fine_tune_epochs
            )
            self.stats.methods_applied.append("knowledge_distillation")

        # Pruning
        if config.enable_pruning:
            print(f"✂️ Applying {config.pruning_method} pruning...")
            pruner = self.pruners.get(config.pruning_method, self.pruners['magnitude'])
            curr_size = self._calculate_model_size(compressed_model)
            target_size = self.stats.original_size * config.target_compression_ratio
            sparsity = 1.0 - (target_size / (curr_size + 1e-8))
            sparsity = float(max(0.0, min(0.95, sparsity)))
            compressed_model = pruner.apply_pruning(compressed_model, sparsity)
            self.stats.methods_applied.append(f"{config.pruning_method}_pruning")

        # Quantization
        if config.enable_quantization:
            print(f"🔢 Applying {config.quantization_bits}-bit quantization...")
            self.quantizer.bits = config.quantization_bits
            self.quantizer.calibrate(compressed_model, data)
            compressed_model = self.quantizer.quantize_model(compressed_model)
            self.stats.methods_applied.append(f"{config.quantization_bits}bit_quantization")

        # Fine-tuning
        if config.fine_tune_epochs > 0:
            print(f"🎯 Fine-tuning for {config.fine_tune_epochs} epochs...")
            compressed_model = self._fine_tune(compressed_model, data, labels, config.fine_tune_epochs)

        # Final stats
        self.stats.compressed_size = self._calculate_model_size(compressed_model)
        self.stats.compressed_accuracy = self._evaluate_model(compressed_model, data, labels)
        self.stats.compression_ratio = self.stats.compressed_size / (self.stats.original_size + 1e-8)
        self.stats.accuracy_drop = self.stats.original_accuracy - self.stats.compressed_accuracy

        print("✅ Model compression completed!")
        self._print_compression_summary()
        return compressed_model

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total = 0
        for param in model.parameters():
            total += param.numel() * param.element_size()
        return total / (1024 ** 2)

    def _evaluate_model(self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> float:
        """Evaluate model performance returning accuracy or inverse MSE."""
        model.eval()
        with torch.no_grad():
            outputs = model(data)
            if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                if outputs.dim() > 1 and outputs.shape[1] > 1:
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == labels.flatten()).float().mean().item()
                else:
                    preds = (outputs > 0.5).float().flatten()
                    acc = (preds == labels.flatten()).float().mean().item()
                return acc
            else:
                mse = F.mse_loss(outputs, labels).item()
                return 1.0 / (1.0 + mse)

    def _fine_tune(
        self, model: nn.Module, data: torch.Tensor, labels: torch.Tensor, epochs: int
    ) -> nn.Module:
        """Fine-tune the compressed model."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                if batch_y.ndim == 1 or (batch_y.ndim == 2 and batch_y.shape[1] == 1):
                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        loss = F.cross_entropy(outputs, batch_y.long())
                    else:
                        loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), batch_y.view(-1).float())
                else:
                    loss = F.mse_loss(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Fine-tuning epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        return model

    def _print_compression_summary(self) -> None:
        """Print compression results summary."""
        print("\n" + "=" * 50)
        print("🗜️ COMPRESSION SUMMARY")
        print("=" * 50)
        print(f"Original Size: {self.stats.original_size:.2f} MB")
        print(f"Compressed Size: {self.stats.compressed_size:.2f} MB")
        print(f"Compression Ratio: {self.stats.compression_ratio:.2f}x")
        print(f"Size Reduction: {(1 - self.stats.compression_ratio) * 100:.1f}%")
        print(f"Original Accuracy: {self.stats.original_accuracy:.4f}")
        print(f"Compressed Accuracy: {self.stats.compressed_accuracy:.4f}")
        print(f"Accuracy Drop: {self.stats.accuracy_drop:.4f}")
        print(f"Methods Applied: {', '.join(self.stats.methods_applied)}")
        print("=" * 50)

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics as dictionary."""
        return {
            'original_size_mb': self.stats.original_size,
            'compressed_size_mb': self.stats.compressed_size,
            'compression_ratio': self.stats.compression_ratio,
            'size_reduction_percent': (1 - self.stats.compression_ratio) * 100,
            'original_accuracy': self.stats.original_accuracy,
            'compressed_accuracy': self.stats.compressed_accuracy,
            'accuracy_drop': self.stats.accuracy_drop,
            'methods_applied': self.stats.methods_applied
        }

    def benchmark_inference_speed(
        self, model: nn.Module, sample_input: torch.Tensor, num_runs: int = 100
    ) -> float:
        """Benchmark model inference speed (ms per run)."""
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input[:1])

        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input[:1])
        end = time.time()
        return ((end - start) / num_runs) * 1000  # ms per inference
