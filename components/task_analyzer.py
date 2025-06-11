"""
--- Updated task_analyzer.py ---
Task Analyzer Component
Analyzes input data and task descriptions to determine optimal neural network characteristics.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass
import re
import warnings

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class TaskType(Enum):
    """Enumeration of supported task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


class DataComplexity(Enum):
    """Data complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TaskCharacteristics:
    """Container for task characteristics and requirements."""
    task_type: TaskType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    dataset_size: int
    num_features: int
    num_classes: int
    data_complexity: DataComplexity
    feature_importance: List[float]
    class_balance: Dict[int, float]
    noise_level: float
    correlation_matrix: np.ndarray
    suggested_architecture_depth: int
    suggested_width: int
    activation_recommendation: str
    loss_function_recommendation: str


class TaskAnalyzer:
    """
    Analyzes tasks to extract characteristics that inform architecture decisions.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.analysis_cache: Dict[str, TaskCharacteristics] = {}

    def analyze_task(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        task_description: str = ""
    ) -> TaskCharacteristics:
        """
        Analyze the given task and return characteristics.

        Args:
            data: Input data tensor
            labels: Target labels tensor
            task_description: Optional natural language description

        Returns:
            TaskCharacteristics object with analysis results
        """
        # Generate a simple cache key based on shapes and a hash of the first few elements
        data_key = f"{tuple(data.shape)}_{tuple(labels.shape)}_{hash(data.detach().cpu().numpy().tobytes()[:1000])}"
        if data_key in self.analysis_cache:
            return self.analysis_cache[data_key]

        # Convert to numpy for analysis
        X = data.detach().cpu().numpy()
        y = labels.detach().cpu().numpy()

        # Determine task type
        task_type = self._infer_task_type(X, y, task_description)

        # Basic shape analysis
        input_shape = tuple(data.shape[1:])
        output_shape = tuple(labels.shape[1:]) if labels.ndim > 1 else (1,)
        dataset_size = data.shape[0]
        num_features = int(np.prod(input_shape))

        # Determine number of classes/outputs
        if task_type == TaskType.CLASSIFICATION:
            num_classes = int(len(np.unique(y)))
        elif task_type == TaskType.MULTI_LABEL:
            num_classes = int(y.shape[1]) if y.ndim > 1 else 1
        else:
            num_classes = 1

        # Analyze data complexity
        data_complexity = self._assess_data_complexity(X, y)

        # Feature importance analysis
        feature_importance = self._compute_feature_importance(X, y, task_type)

        # Class balance analysis
        class_balance = self._analyze_class_balance(y, task_type)

        # Noise level estimation
        noise_level = self._estimate_noise_level(X, y)

        # Correlation analysis
        correlation_matrix = np.corrcoef(X.T) if X.shape[1] > 1 else np.array([[1.0]])

        # Architecture recommendations
        arch_depth, arch_width = self._suggest_architecture_size(
            dataset_size, num_features, num_classes, data_complexity
        )

        # Activation function recommendation
        activation_rec = self._recommend_activation(task_type, data_complexity)

        # Loss function recommendation
        loss_rec = self._recommend_loss_function(task_type, class_balance)

        result = TaskCharacteristics(
            task_type=task_type,
            input_shape=input_shape,
            output_shape=output_shape,
            dataset_size=dataset_size,
            num_features=num_features,
            num_classes=num_classes,
            data_complexity=data_complexity,
            feature_importance=feature_importance,
            class_balance=class_balance,
            noise_level=noise_level,
            correlation_matrix=correlation_matrix,
            suggested_architecture_depth=arch_depth,
            suggested_width=arch_width,
            activation_recommendation=activation_rec,
            loss_function_recommendation=loss_rec
        )

        # Cache the results
        self.analysis_cache[data_key] = result
        return result

    def _infer_task_type(
        self,
        X: np.ndarray,
        y: np.ndarray,
        description: str
    ) -> TaskType:
        """Infer the task type from data dimensions, labels, and description."""

        desc_lower = description.lower()

        # Check description keywords first
        if any(word in desc_lower for word in ['time series', 'temporal', 'sequence']):
            return TaskType.TIME_SERIES
        if any(word in desc_lower for word in ['anomaly', 'outlier', 'detection']):
            return TaskType.ANOMALY_DETECTION
        if any(word in desc_lower for word in ['cluster', 'group', 'segment']):
            return TaskType.CLUSTERING
        if any(word in desc_lower for word in ['regression', 'predict', 'continuous']):
            return TaskType.REGRESSION
        if any(word in desc_lower for word in ['classify', 'classification', 'category']):
            return TaskType.CLASSIFICATION

        # Fallback based on labels
        if y.ndim == 1:
            unique_vals = np.unique(y)
            # If integer labels with few unique values, classify
            if np.issubdtype(y.dtype, np.integer) and len(unique_vals) <= 20:
                return TaskType.CLASSIFICATION
            else:
                return TaskType.REGRESSION
        else:
            # Multi-dimensional output: check if binary indicators => multi-label
            if np.issubdtype(y.dtype, np.integer) or np.all(np.isin(y, [0, 1])):
                return TaskType.MULTI_LABEL
            else:
                return TaskType.REGRESSION

    def _assess_data_complexity(self, X: np.ndarray, y: np.ndarray) -> DataComplexity:
        """Assess the complexity of the data."""
        n_samples, n_features = X.shape

        # Feature-to-sample ratio
        feature_ratio = n_features / (n_samples + 1e-8)

        # Dimensionality bucket
        if n_features < 10:
            dim_complexity = 0
        elif n_features < 100:
            dim_complexity = 1
        elif n_features < 1000:
            dim_complexity = 2
        else:
            dim_complexity = 3

        # Non-linearity estimation using variance of pairwise distances
        if n_samples > 100:
            sample_idx = np.random.choice(n_samples, min(100, n_samples), replace=False)
            X_sample = X[sample_idx]
        else:
            X_sample = X

        distances = []
        m = min(50, len(X_sample))
        for i in range(m):
            for j in range(i + 1, m):
                distances.append(np.linalg.norm(X_sample[i] - X_sample[j]))

        if distances:
            distance_var = np.var(distances)
            nonlinearity_score = min(distance_var / (np.mean(distances) + 1e-8), 3)
        else:
            nonlinearity_score = 0

        complexity_score = dim_complexity + feature_ratio + (nonlinearity_score / 3)

        if complexity_score < 1.0:
            return DataComplexity.LOW
        elif complexity_score < 2.0:
            return DataComplexity.MEDIUM
        elif complexity_score < 3.0:
            return DataComplexity.HIGH
        else:
            return DataComplexity.VERY_HIGH

    def _compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType
    ) -> List[float]:
        """Compute feature importance scores using mutual information or variance fallback."""
        try:
            if task_type == TaskType.CLASSIFICATION or task_type == TaskType.MULTI_LABEL:
                if len(np.unique(y)) > 1:
                    importance = mutual_info_classif(X, y.ravel())
                else:
                    importance = np.ones(X.shape[1])
            else:
                importance = mutual_info_regression(X, y.ravel())

            # Normalize to [0, 1]
            max_imp = np.max(importance) if importance.size else 0
            if max_imp > 0:
                importance = importance / max_imp

            return importance.tolist()

        except Exception as e:
            warnings.warn(f"Feature importance calculation failed: {e}. Using variance fallback.")
            variances = np.var(X, axis=0)
            max_var = np.max(variances) if variances.size else 1.0
            return (variances / (max_var + 1e-8)).tolist()

    def _analyze_class_balance(
        self,
        y: np.ndarray,
        task_type: TaskType
    ) -> Dict[int, float]:
        """Analyze class balance for classification tasks."""
        if task_type != TaskType.CLASSIFICATION:
            return {}

        y_flat = y.ravel()
        unique, counts = np.unique(y_flat, return_counts=True)
        total = len(y_flat)
        return {int(cls): float(count) / total for cls, count in zip(unique, counts)}

    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """Estimate noise level in the data using local variance via k-NN."""
        try:
            if X.shape[0] > 10:
                indices = np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)
                X_subset = X[indices]

                nn = NearestNeighbors(n_neighbors=min(5, len(X_subset) - 1))
                nn.fit(X_subset)
                distances, _ = nn.kneighbors(X_subset)
                # Exclude self-distance
                avg_distance = np.mean(distances[:, 1:], axis=1).mean()
                noise_estimate = min(avg_distance / (np.std(X_subset) + 1e-8), 1.0)
                return float(noise_estimate)
            else:
                return 0.1
        except Exception:
            return 0.1

    def _suggest_architecture_size(
        self,
        dataset_size: int,
        num_features: int,
        num_classes: int,
        complexity: DataComplexity
    ) -> Tuple[int, int]:
        """Suggest appropriate architecture depth and width."""
        complexity_multiplier = {
            DataComplexity.LOW: 0.5,
            DataComplexity.MEDIUM: 1.0,
            DataComplexity.HIGH: 1.5,
            DataComplexity.VERY_HIGH: 2.0
        }[complexity]

        # Depth suggestion
        if dataset_size < 1000:
            base_depth = 2
        elif dataset_size < 10000:
            base_depth = 3
        else:
            base_depth = 4

        suggested_depth = max(2, int(base_depth * complexity_multiplier))

        # Width suggestion
        base_width = max(32, min(512, num_features * 2))
        suggested_width = max(16, int(base_width * complexity_multiplier))

        return suggested_depth, suggested_width

    def _recommend_activation(
        self,
        task_type: TaskType,
        complexity: DataComplexity
    ) -> str:
        """Recommend activation function based on task characteristics."""
        if complexity in [DataComplexity.HIGH, DataComplexity.VERY_HIGH]:
            return "swish"
        elif task_type == TaskType.CLASSIFICATION:
            return "relu"
        else:
            return "leaky_relu"

    def _recommend_loss_function(
        self,
        task_type: TaskType,
        class_balance: Dict[int, float]
    ) -> str:
        """Recommend loss function based on task type and class balance."""
        if task_type == TaskType.CLASSIFICATION:
            if len(class_balance) == 2:
                min_ratio = min(class_balance.values()) if class_balance else 1.0
                if min_ratio < 0.2:
                    return "focal_loss"
                else:
                    return "binary_crossentropy"
            else:
                min_ratio = min(class_balance.values()) if class_balance else 1.0
                if min_ratio < 0.1:
                    return "weighted_crossentropy"
                else:
                    return "crossentropy"
        elif task_type == TaskType.REGRESSION:
            return "mse"
        elif task_type == TaskType.MULTI_LABEL:
            return "binary_crossentropy"
        else:
            return "mse"

    def compute_task_similarity(
        self,
        task1: TaskCharacteristics,
        task2: TaskCharacteristics
    ) -> float:
        """Compute similarity score between two tasks."""
        score = 0.0

        # Task type similarity (40% weight)
        if task1.task_type == task2.task_type:
            score += 0.4

        # Feature count similarity (20% weight)
        feat_ratio = min(task1.num_features, task2.num_features) / max(task1.num_features, task2.num_features)
        score += 0.2 * feat_ratio

        # Dataset size similarity (15% weight)
        size_ratio = min(task1.dataset_size, task2.dataset_size) / max(task1.dataset_size, task2.dataset_size)
        score += 0.15 * size_ratio

        # Number of classes similarity (15% weight)
        class_ratio = min(task1.num_classes, task2.num_classes) / max(task1.num_classes, task2.num_classes)
        score += 0.15 * class_ratio

        # Complexity similarity (10% weight)
        if task1.data_complexity == task2.data_complexity:
            score += 0.1

        return min(score, 1.0)

    def get_optimization_hints(
        self,
        task_chars: TaskCharacteristics
    ) -> Dict[str, Any]:
        """Get optimization hints based on task characteristics."""
        hints: Dict[str, Any] = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'regularization': 0.01,
            'dropout_rate': 0.1,
            'early_stopping_patience': 10
        }

        # Adjust based on dataset size
        if task_chars.dataset_size < 1000:
            hints['learning_rate'] = 0.01
            hints['batch_size'] = 16
            hints['regularization'] = 0.1
        elif task_chars.dataset_size > 100000:
            hints['learning_rate'] = 0.0001
            hints['batch_size'] = 128
            hints['regularization'] = 0.001

        # Adjust based on complexity
        if task_chars.data_complexity == DataComplexity.VERY_HIGH:
            hints['dropout_rate'] = 0.3
            hints['early_stopping_patience'] = 20
        elif task_chars.data_complexity == DataComplexity.LOW:
            hints['dropout_rate'] = 0.05
            hints['early_stopping_patience'] = 5

        # Adjust for imbalanced classes
        if task_chars.class_balance:
            min_ratio = min(task_chars.class_balance.values())
            if min_ratio < 0.1:
                hints['class_weights'] = {k: 1.0 / v for k, v in task_chars.class_balance.items()}

        return hints
