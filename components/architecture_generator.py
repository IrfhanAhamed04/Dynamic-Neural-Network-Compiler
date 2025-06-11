"""
Architecture Generator Component
Generates neural network architectures based on task characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from task_analyzer import DataComplexity
import random
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: str
    input_size: int
    output_size: int
    activation: str
    dropout_rate: float = 0.0
    batch_norm: bool = False
    kwargs: Dict[str, Any] = None


@dataclass
class NetworkTemplate:
    """Template for a complete neural network architecture."""
    layers: List[LayerConfig]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    architecture_type: str
    estimated_params: int
    estimated_flops: int
    confidence_score: float


class BaseArchitecture(nn.Module, ABC):
    """Base class for generated architectures."""

    def __init__(self, template: NetworkTemplate):
        super().__init__()
        self.template = template
        self.layers = nn.ModuleList()
        self._build_from_template()

    @abstractmethod
    def _build_from_template(self):
        """Build the network from the template."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass


class DenseNetwork(BaseArchitecture):
    """Fully connected dense network."""

    def _build_from_template(self):
        """Build dense network from template."""
        for layer_config in self.template.layers:
            if layer_config.layer_type == "linear":
                layer = nn.Linear(layer_config.input_size, layer_config.output_size)
                self.layers.append(layer)
                if layer_config.batch_norm:
                    self.layers.append(nn.BatchNorm1d(layer_config.output_size))
                if layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout(layer_config.dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dense network."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        idx = 0
        for i, layer_config in enumerate(self.template.layers):
            # Linear
            x = self.layers[idx](x)
            idx += 1

            # Batch norm if specified
            if layer_config.batch_norm:
                x = self.layers[idx](x)
                idx += 1

            # Activation except for last layer
            if i < len(self.template.layers) - 1:
                x = self._get_activation(layer_config.activation)(x)

            # Dropout if specified
            if layer_config.dropout_rate > 0:
                x = self.layers[idx](x)
                idx += 1

        return x

    def _get_activation(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'swish': lambda x: x * torch.sigmoid(x),
            'gelu': F.gelu
        }
        return activations.get(activation_name, F.relu)


class ConvolutionalNetwork(BaseArchitecture):
    """Convolutional neural network for image-like data."""

    def _build_from_template(self):
        """Build convolutional network from template."""
        for layer_config in self.template.layers:
            if layer_config.layer_type == "conv2d":
                kwargs = layer_config.kwargs or {}
                layer = nn.Conv2d(
                    layer_config.input_size,
                    layer_config.output_size,
                    **kwargs
                )
                self.layers.append(layer)
                if layer_config.batch_norm:
                    self.layers.append(nn.BatchNorm2d(layer_config.output_size))
                if layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout2d(layer_config.dropout_rate))

            elif layer_config.layer_type == "linear":
                layer = nn.Linear(layer_config.input_size, layer_config.output_size)
                self.layers.append(layer)
                if layer_config.batch_norm:
                    self.layers.append(nn.BatchNorm1d(layer_config.output_size))
                if layer_config.dropout_rate > 0:
                    self.layers.append(nn.Dropout(layer_config.dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional network."""
        idx = 0
        flatten_next = False
        for i, layer_config in enumerate(self.template.layers):
            if flatten_next and x.dim() > 2:
                x = x.view(x.size(0), -1)
                flatten_next = False

            # Main layer
            x = self.layers[idx](x)
            idx += 1

            # Batch norm if specified
            if layer_config.batch_norm:
                x = self.layers[idx](x)
                idx += 1

            # Activation except last layer
            if i < len(self.template.layers) - 1:
                x = self._get_activation(layer_config.activation)(x)
                if layer_config.layer_type == "conv2d":
                    x = F.max_pool2d(x, kernel_size=2, stride=2)

            # Dropout if specified
            if layer_config.dropout_rate > 0:
                x = self.layers[idx](x)
                idx += 1

            # Flatten if next is linear
            if (i < len(self.template.layers) - 1 and
                    layer_config.layer_type == "conv2d" and
                    self.template.layers[i + 1].layer_type == "linear"):
                flatten_next = True

        return x

    def _get_activation(self, activation_name: str):
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'swish': lambda x: x * torch.sigmoid(x),
            'gelu': F.gelu
        }
        return activations.get(activation_name, F.relu)


class ResidualNetwork(BaseArchitecture):
    """Residual network with skip connections."""

    def _build_from_template(self):
        """Build residual network from template."""
        self.residual_blocks = nn.ModuleList()
        self.projections = nn.ModuleList()
        for layer_config in self.template.layers:
            if layer_config.layer_type == "residual_block":
                block_layers = []
                # First layer
                block_layers.append(nn.Linear(layer_config.input_size, layer_config.output_size))
                if layer_config.batch_norm:
                    block_layers.append(nn.BatchNorm1d(layer_config.output_size))
                block_layers.append(self._get_activation_module(layer_config.activation))
                # Second layer
                block_layers.append(nn.Linear(layer_config.output_size, layer_config.output_size))
                if layer_config.batch_norm:
                    block_layers.append(nn.BatchNorm1d(layer_config.output_size))
                if layer_config.dropout_rate > 0:
                    block_layers.append(nn.Dropout(layer_config.dropout_rate))
                self.residual_blocks.append(nn.Sequential(*block_layers))
                # Projection if sizes differ
                if layer_config.input_size != layer_config.output_size:
                    self.projections.append(nn.Linear(layer_config.input_size, layer_config.output_size))
                else:
                    self.projections.append(nn.Identity())

            elif layer_config.layer_type == "linear":
                self.layers.append(nn.Linear(layer_config.input_size, layer_config.output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual network."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        for block, projection in zip(self.residual_blocks, self.projections):
            identity = projection(x)
            x = block(x) + identity
            x = F.relu(x)
        if self.layers:
            x = self.layers[0](x)
        return x

    def _get_activation_module(self, activation_name: str):
        """Get activation function module by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU()
        }
        return activations.get(activation_name, nn.ReLU())


class ArchitectureGenerator:
    """
    Generates neural network architectures based on task characteristics.
    """

    def __init__(self):
        self.architecture_patterns = {
            'dense': self._generate_dense_architectures,
            'convolutional': self._generate_conv_architectures,
            'residual': self._generate_residual_architectures,
            'hybrid': self._generate_hybrid_architectures
        }

    def generate_candidates(
        self,
        task_chars,
        num_candidates: int = 10
    ) -> List[NetworkTemplate]:
        """
        Generate architecture candidates based on task characteristics.

        Args:
            task_chars: TaskCharacteristics object
            num_candidates: Number of candidate architectures to generate

        Returns:
            List of NetworkTemplate objects
        """
        candidates: List[NetworkTemplate] = []

        # Determine suitable architecture types
        suitable_types = self._determine_suitable_architectures(task_chars)

        # Generate candidates per type
        candidates_per_type = max(1, num_candidates // len(suitable_types))
        for arch_type in suitable_types:
            if arch_type in self.architecture_patterns:
                type_candidates = self.architecture_patterns[arch_type](
                    task_chars, candidates_per_type
                )
                candidates.extend(type_candidates)

        # If not enough, fill with dense
        while len(candidates) < num_candidates:
            candidates.extend(self._generate_dense_architectures(task_chars, 1))

        # Score and sort top num_candidates
        scored_candidates: List[NetworkTemplate] = []
        for template in candidates[:num_candidates]:
            score = self._score_architecture(template, task_chars)
            template.confidence_score = score
            scored_candidates.append(template)

        scored_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return scored_candidates

    def _determine_suitable_architectures(self, task_chars) -> List[str]:
        """Determine which architecture types are suitable for the task."""
        suitable = ["dense"]
        if len(task_chars.input_shape) >= 3:
            suitable.append("convolutional")
        if task_chars.data_complexity in {DataComplexity.HIGH, DataComplexity.VERY_HIGH} or \
           task_chars.dataset_size > 10000:
            suitable.append("residual")
        if task_chars.data_complexity == DataComplexity.VERY_HIGH and task_chars.dataset_size > 50000:
            suitable.append("hybrid")
        return suitable

    def _generate_dense_architectures(
        self,
        task_chars,
        num_candidates: int
    ) -> List[NetworkTemplate]:
        """Generate dense (fully connected) architecture candidates."""
        candidates: List[NetworkTemplate] = []
        input_size = task_chars.num_features
        output_size = task_chars.num_classes if task_chars.num_classes > 1 else 1

        for _ in range(num_candidates):
            layers: List[LayerConfig] = []
            num_layers = random.randint(2, task_chars.suggested_architecture_depth + 1)
            base_width = task_chars.suggested_width
            current_size = input_size

            for i in range(num_layers - 1):
                width_multiplier = random.uniform(0.5, 2.0)
                layer_width = max(16, int(base_width * width_multiplier))
                decay = (num_layers - i - 1) / num_layers
                layer_width = max(16, int(layer_width * (0.5 + 0.5 * decay)))

                layers.append(LayerConfig(
                    layer_type="linear",
                    input_size=current_size,
                    output_size=layer_width,
                    activation=task_chars.activation_recommendation,
                    dropout_rate=random.uniform(0.0, 0.3),
                    batch_norm=random.choice([True, False])
                ))
                current_size = layer_width

            layers.append(LayerConfig(
                layer_type="linear",
                input_size=current_size,
                output_size=output_size,
                activation="linear",
                dropout_rate=0.0,
                batch_norm=False
            ))

            estimated_params = self._estimate_parameters(layers)
            template = NetworkTemplate(
                layers=layers,
                input_shape=task_chars.input_shape,
                output_shape=task_chars.output_shape,
                architecture_type="dense",
                estimated_params=estimated_params,
                estimated_flops=estimated_params * 2,
                confidence_score=0.0
            )
            candidates.append(template)

        return candidates

    def _generate_conv_architectures(
        self,
        task_chars,
        num_candidates: int
    ) -> List[NetworkTemplate]:
        """Generate convolutional architecture candidates."""
        candidates: List[NetworkTemplate] = []
        if len(task_chars.input_shape) < 3:
            return candidates

        input_channels = task_chars.input_shape[0]
        output_size = task_chars.num_classes if task_chars.num_classes > 1 else 1

        for _ in range(num_candidates):
            layers: List[LayerConfig] = []
            num_conv = random.randint(2, 4)
            current_channels = input_channels

            for _ in range(num_conv):
                out_channels = random.choice([32, 64, 128, 256])
                kernel_size = random.choice([3, 5])
                layers.append(LayerConfig(
                    layer_type="conv2d",
                    input_size=current_channels,
                    output_size=out_channels,
                    activation=task_chars.activation_recommendation,
                    dropout_rate=random.uniform(0.0, 0.2),
                    batch_norm=random.choice([True, False]),
                    kwargs={"kernel_size": kernel_size, "padding": kernel_size // 2}
                ))
                current_channels = out_channels

            spatial = task_chars.input_shape[-1]
            for _ in range(num_conv):
                spatial = spatial // 2

            flattened = current_channels * spatial * spatial
            num_dense = random.randint(1, 2)
            current_size = flattened

            for i in range(num_dense - 1):
                layer_width = random.choice([128, 256, 512])
                layers.append(LayerConfig(
                    layer_type="linear",
                    input_size=current_size,
                    output_size=layer_width,
                    activation=task_chars.activation_recommendation,
                    dropout_rate=random.uniform(0.2, 0.5),
                    batch_norm=random.choice([True, False])
                ))
                current_size = layer_width

            layers.append(LayerConfig(
                layer_type="linear",
                input_size=current_size,
                output_size=output_size,
                activation="linear",
                dropout_rate=0.0,
                batch_norm=False
            ))

            estimated_params = self._estimate_parameters(layers)
            template = NetworkTemplate(
                layers=layers,
                input_shape=task_chars.input_shape,
                output_shape=task_chars.output_shape,
                architecture_type="convolutional",
                estimated_params=estimated_params,
                estimated_flops=estimated_params * 4,
                confidence_score=0.0
            )
            candidates.append(template)

        return candidates

    def _generate_residual_architectures(
        self,
        task_chars,
        num_candidates: int
    ) -> List[NetworkTemplate]:
        """Generate residual architecture candidates."""
        candidates: List[NetworkTemplate] = []
        input_size = task_chars.num_features
        output_size = task_chars.num_classes if task_chars.num_classes > 1 else 1

        for _ in range(num_candidates):
            layers: List[LayerConfig] = []
            num_blocks = random.randint(2, 4)
            base_width = task_chars.suggested_width
            current_size = input_size

            for _ in range(num_blocks):
                width_multiplier = random.uniform(0.8, 1.5)
                block_width = max(32, int(base_width * width_multiplier))
                layers.append(LayerConfig(
                    layer_type="residual_block",
                    input_size=current_size,
                    output_size=block_width,
                    activation=task_chars.activation_recommendation,
                    dropout_rate=random.uniform(0.0, 0.2),
                    batch_norm=True
                ))
                current_size = block_width

            layers.append(LayerConfig(
                layer_type="linear",
                input_size=current_size,
                output_size=output_size,
                activation="linear",
                dropout_rate=0.0,
                batch_norm=False
            ))

            estimated_params = self._estimate_parameters(layers) * 2
            template = NetworkTemplate(
                layers=layers,
                input_shape=task_chars.input_shape,
                output_shape=task_chars.output_shape,
                architecture_type="residual",
                estimated_params=estimated_params,
                estimated_flops=estimated_params * 3,
                confidence_score=0.0
            )
            candidates.append(template)

        return candidates

    def _generate_hybrid_architectures(
        self,
        task_chars,
        num_candidates: int
    ) -> List[NetworkTemplate]:
        """Generate hybrid architecture candidates combining different types."""
        dense_c = self._generate_dense_architectures(task_chars, num_candidates // 2)
        residual_c = self._generate_residual_architectures(task_chars, num_candidates // 2)
        for template in dense_c + residual_c:
            template.architecture_type = "hybrid"
        return dense_c + residual_c

    def _estimate_parameters(self, layers: List[LayerConfig]) -> int:
        """Estimate the number of parameters in the architecture."""
        total = 0
        for layer in layers:
            if layer.layer_type in ["linear", "conv2d"]:
                if layer.layer_type == "linear":
                    total += (layer.input_size + 1) * layer.output_size
                else:
                    kernel = layer.kwargs.get("kernel_size", 3) if layer.kwargs else 3
                    total += (layer.input_size * kernel * kernel + 1) * layer.output_size
                if layer.batch_norm:
                    total += layer.output_size * 2
            elif layer.layer_type == "residual_block":
                total += (layer.input_size + 1) * layer.output_size
                total += (layer.output_size + 1) * layer.output_size
                if layer.batch_norm:
                    total += layer.output_size * 4
        return total

    def _score_architecture(self, template: NetworkTemplate, task_chars) -> float:
        """Score an architecture based on task characteristics."""
        score = 0.0

        # Parameter efficiency
        param_ratio = template.estimated_params / (task_chars.dataset_size + 1)
        if param_ratio < 0.1:
            score += 0.3
        elif param_ratio < 1.0:
            score += 0.2
        else:
            score += 0.1

        # Architecture-type match
        if task_chars.data_complexity in {DataComplexity.HIGH, DataComplexity.VERY_HIGH}:
            if template.architecture_type in ["residual", "hybrid"]:
                score += 0.3
            elif template.architecture_type == "convolutional":
                score += 0.2
            else:
                score += 0.1
        else:
            if template.architecture_type == "dense":
                score += 0.3
            else:
                score += 0.2

        # Depth appropriateness
        num_layers = len(template.layers)
        if task_chars.data_complexity == DataComplexity.LOW and num_layers <= 3:
            score += 0.2
        elif task_chars.data_complexity in {DataComplexity.HIGH, DataComplexity.VERY_HIGH} and num_layers >= 3:
            score += 0.2
        else:
            score += 0.1

        # Width relative to dataset size
        widths = [layer.output_size for layer in template.layers[:-1] if layer.layer_type == "linear"]
        avg_width = np.mean(widths) if widths else 0
        if task_chars.dataset_size < 1000 and avg_width <= 128:
            score += 0.2
        elif task_chars.dataset_size > 10000 and avg_width >= 128:
            score += 0.2
        else:
            score += 0.1

        return min(score, 1.0)

    def build_model(self, template: NetworkTemplate) -> nn.Module:
        """
        Build a PyTorch model from a template.

        Args:
            template: NetworkTemplate to build

        Returns:
            Instantiated nn.Module
        """
        if template.architecture_type == "dense":
            return DenseNetwork(template)
        elif template.architecture_type == "convolutional":
            return ConvolutionalNetwork(template)
        elif template.architecture_type == "residual":
            return ResidualNetwork(template)
        elif template.architecture_type == "hybrid":
            if len(template.input_shape) >= 3:
                return ConvolutionalNetwork(template)
            else:
                return ResidualNetwork(template)
        else:
            return DenseNetwork(template)
