"""
--- Updated sample_datasets.py ---
Sample Datasets Module
Generates synthetic datasets for testing the Dynamic Neural Network Compiler.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, Any
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_circles,
    make_moons,
    make_multilabel_classification,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    """Information about a generated dataset."""
    name: str
    task_type: str
    num_samples: int
    num_features: int
    num_classes: int
    difficulty: str
    description: str


class SampleDatasetGenerator:
    """
    Generates various types of synthetic datasets for testing neural network compilation.
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.scaler = StandardScaler()

    def generate_classification_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_classes: int = 3,
        difficulty: str = "medium",
        class_imbalance: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate a classification dataset.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            n_classes: Number of classes.
            difficulty: 'easy', 'medium', or 'hard'.
            class_imbalance: Whether to create imbalanced classes.

        Returns:
            Tuple of (features, labels, dataset_info).
        """
        difficulty_params = {
            "easy": {
                "n_informative": n_features,
                "n_redundant": 0,
                "n_clusters_per_class": 1,
                "class_sep": 2.0,
            },
            "medium": {
                "n_informative": max(1, n_features // 2),
                "n_redundant": n_features // 4,
                "n_clusters_per_class": 1,
                "class_sep": 1.0,
            },
            "hard": {
                "n_informative": max(1, n_features // 3),
                "n_redundant": n_features // 3,
                "n_clusters_per_class": 2,
                "class_sep": 0.5,
            },
        }
        params = difficulty_params.get(difficulty, difficulty_params["medium"])

        weights = None
        if class_imbalance and n_classes > 1:
            weights = [1.0 / (i + 1) for i in range(n_classes)]
            weights = [w / sum(weights) for w in weights]

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=params["n_informative"],
            n_redundant=params["n_redundant"],
            n_clusters_per_class=params["n_clusters_per_class"],
            class_sep=params["class_sep"],
            weights=weights,
            random_state=self.seed,
        )

        X = self.scaler.fit_transform(X)
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        info = DatasetInfo(
            name=f"classification_{difficulty}_{n_classes}class",
            task_type="classification",
            num_samples=n_samples,
            num_features=n_features,
            num_classes=n_classes,
            difficulty=difficulty,
            description=f"{difficulty.capitalize()} {n_classes}-class classification with {n_features} features",
        )
        return X_tensor, y_tensor, info

    def generate_regression_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 15,
        difficulty: str = "medium",
        noise_level: float = 0.1,
        multi_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate a regression dataset.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            difficulty: 'easy', 'medium', or 'hard'.
            noise_level: Amount of noise to add (0.0 to 1.0).
            multi_output: Whether to create a multi-output regression.

        Returns:
            Tuple of (features, targets, dataset_info).
        """
        difficulty_params = {
            "easy": {"n_informative": n_features, "n_targets": 1 if not multi_output else 3, "bias": 0.0},
            "medium": {"n_informative": max(1, n_features // 2), "n_targets": 1 if not multi_output else 3, "bias": 50.0},
            "hard": {"n_informative": max(1, n_features // 3), "n_targets": 1 if not multi_output else 5, "bias": 100.0},
        }
        params = difficulty_params.get(difficulty, difficulty_params["medium"])

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=params["n_informative"],
            n_targets=params["n_targets"],
            noise=noise_level * 100,
            bias=params["bias"],
            random_state=self.seed,
        )

        X = self.scaler.fit_transform(X)
        if y.ndim == 1:
            y = (y - np.mean(y)) / (np.std(y) + 1e-8)
            y = y.reshape(-1, 1)
        else:
            y = StandardScaler().fit_transform(y)

        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        output_dim = y_tensor.shape[1]
        info = DatasetInfo(
            name=f"regression_{difficulty}_{output_dim}d",
            task_type="regression",
            num_samples=n_samples,
            num_features=n_features,
            num_classes=output_dim,
            difficulty=difficulty,
            description=f"{difficulty.capitalize()} regression with {n_features} features and {output_dim} outputs",
        )
        return X_tensor, y_tensor, info

    def generate_time_series_dataset(
        self,
        n_samples: int = 1000,
        sequence_length: int = 50,
        n_features: int = 1,
        pattern_type: str = "sine",
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate a time series dataset.

        Args:
            n_samples: Number of sequences.
            sequence_length: Length of each sequence.
            n_features: Number of features per time step.
            pattern_type: 'sine', 'linear', or 'nonlinear'.

        Returns:
            Tuple of (sequences, targets, dataset_info).
        """
        sequences = []
        targets = []

        for _ in range(n_samples):
            t = np.linspace(0, 10, sequence_length)
            if pattern_type == "sine":
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                signal = np.sin(freq * t + phase) + np.random.normal(0, 0.1, sequence_length)
            elif pattern_type == "linear":
                slope = np.random.uniform(-1, 1)
                intercept = np.random.uniform(-5, 5)
                signal = slope * t + intercept + np.random.normal(0, 0.5, sequence_length)
            else:
                signal = np.sin(t) * np.cos(t / 2) + 0.1 * t**2 + np.random.normal(0, 0.2, sequence_length)

            if n_features > 1:
                features = np.column_stack(
                    [signal] + [signal + np.random.normal(0, 0.1, sequence_length) for _ in range(n_features - 1)]
                )
            else:
                features = signal.reshape(-1, 1)

            sequences.append(features.astype(np.float32))
            targets.append(signal[-1].astype(np.float32))

        X = np.stack(sequences)  # shape: (n_samples, sequence_length, n_features)
        y = np.array(targets).reshape(-1, 1)  # shape: (n_samples, 1)

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        info = DatasetInfo(
            name=f"timeseries_{pattern_type}_{sequence_length}seq",
            task_type="time_series",
            num_samples=n_samples,
            num_features=n_features,
            num_classes=1,
            difficulty="medium",
            description=f"Time series with {pattern_type} pattern, {sequence_length} steps",
        )
        return X_tensor, y_tensor, info

    def generate_anomaly_detection_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        contamination: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate an anomaly detection dataset.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            contamination: Fraction of anomalies (0.0 to 1.0).

        Returns:
            Tuple of (features, labels, dataset_info).
        """
        n_normal = int(n_samples * (1 - contamination))
        n_anomalies = n_samples - n_normal

        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=np.eye(n_features), size=n_normal
        )
        anomaly_data = np.random.multivariate_normal(
            mean=np.ones(n_features) * 3, cov=np.eye(n_features) * 4, size=n_anomalies
        )

        X = np.vstack([normal_data, anomaly_data])
        y = np.array([0] * n_normal + [1] * n_anomalies)

        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        X = StandardScaler().fit_transform(X)
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        info = DatasetInfo(
            name=f"anomaly_{int(contamination * 100)}pct",
            task_type="anomaly_detection",
            num_samples=n_samples,
            num_features=n_features,
            num_classes=2,
            difficulty="medium",
            description=f"Anomaly detection with {contamination * 100:.1f}% anomalies",
        )
        return X_tensor, y_tensor, info

    def generate_multi_label_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        n_labels: int = 5,
        correlation: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate a multi-label classification dataset.

        Args:
            n_samples: Number of samples.
            n_features: Number of features.
            n_labels: Number of labels.
            correlation: Correlation between labels (0.0 to 1.0).

        Returns:
            Tuple of (features, labels, dataset_info).
        """
        X, y = make_multilabel_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_labels,
            n_labels=max(1, n_labels // 2),
            length=50,
            allow_unlabeled=False,
            sparse=False,
            return_indicator="dense",
            random_state=self.seed,
        )

        if correlation > 0:
            for i in range(n_labels - 1):
                mask = np.random.rand(n_samples) < correlation
                y[mask, i + 1] = y[mask, i]

        X = StandardScaler().fit_transform(X)
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        info = DatasetInfo(
            name=f"multilabel_{n_labels}labels",
            task_type="multi_label",
            num_samples=n_samples,
            num_features=n_features,
            num_classes=n_labels,
            difficulty="medium",
            description=f"Multi-label classification with {n_labels} labels",
        )
        return X_tensor, y_tensor, info

    def generate_nonlinear_dataset(
        self,
        n_samples: int = 1000,
        pattern: str = "circles",
    ) -> Tuple[torch.Tensor, torch.Tensor, DatasetInfo]:
        """
        Generate datasets with nonlinear patterns.

        Args:
            n_samples: Number of samples.
            pattern: 'circles', 'moons', or 'blobs'.

        Returns:
            Tuple of (features, labels, dataset_info).
        """
        if pattern == "circles":
            X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=self.seed)
            description = "Concentric circles classification"
        elif pattern == "moons":
            X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=self.seed)
            description = "Interleaving half-circles classification"
        else:
            X, y = make_blobs(
                n_samples=n_samples, centers=4, n_features=2, random_state=self.seed, cluster_std=1.0
            )
            description = "Blob clusters classification"

        X = StandardScaler().fit_transform(X)
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()

        info = DatasetInfo(
            name=f"nonlinear_{pattern}",
            task_type="classification",
            num_samples=n_samples,
            num_features=2,
            num_classes=int(len(np.unique(y))),
            difficulty="hard",
            description=description,
        )
        return X_tensor, y_tensor, info

    def get_all_sample_datasets(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, DatasetInfo]]:
        """
        Generate a comprehensive set of sample datasets for testing.

        Returns:
            Dictionary mapping dataset names to (features, labels, info) tuples.
        """
        datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor, DatasetInfo]] = {}

        datasets["easy_classification"] = self.generate_classification_dataset(
            n_samples=800, n_features=15, n_classes=2, difficulty="easy"
        )
        datasets["hard_classification"] = self.generate_classification_dataset(
            n_samples=1200, n_features=25, n_classes=5, difficulty="hard"
        )
        datasets["imbalanced_classification"] = self.generate_classification_dataset(
            n_samples=1000, n_features=20, n_classes=3, difficulty="medium", class_imbalance=True
        )

        datasets["simple_regression"] = self.generate_regression_dataset(
            n_samples=800, n_features=10, difficulty="easy", noise_level=0.05
        )
        datasets["complex_regression"] = self.generate_regression_dataset(
            n_samples=1200, n_features=20, difficulty="hard", noise_level=0.2
        )
        datasets["multi_output_regression"] = self.generate_regression_dataset(
            n_samples=1000, n_features=15, difficulty="medium", multi_output=True
        )

        datasets["sine_timeseries"] = self.generate_time_series_dataset(
            n_samples=500, sequence_length=30, pattern_type="sine"
        )
        datasets["nonlinear_timeseries"] = self.generate_time_series_dataset(
            n_samples=600, sequence_length=40, pattern_type="nonlinear"
        )

        datasets["anomaly_detection"] = self.generate_anomaly_detection_dataset(
            n_samples=1000, n_features=12, contamination=0.15
        )

        datasets["multi_label"] = self.generate_multi_label_dataset(
            n_samples=800, n_features=18, n_labels=4
        )

        datasets["circles"] = self.generate_nonlinear_dataset(
            n_samples=800, pattern="circles"
        )
        datasets["moons"] = self.generate_nonlinear_dataset(
            n_samples=800, pattern="moons"
        )

        return datasets

    def visualize_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        info: DatasetInfo,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize a dataset (works best for 2D data).

        Args:
            X: Features tensor.
            y: Labels tensor.
            info: Dataset information.
            save_path: Optional path to save the plot.
        """
        X_np = X.cpu().numpy()
        y_np = y.cpu().numpy()

        plt.figure(figsize=(10, 6))

        if X_np.shape[1] == 2:
            if info.task_type in ["classification", "anomaly_detection"]:
                scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="viridis", alpha=0.7)
                plt.colorbar(scatter)
            else:
                scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np.flatten(), cmap="viridis", alpha=0.7)
                plt.colorbar(scatter)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")

        elif info.task_type == "time_series":
            for i in range(min(5, len(X_np))):
                plt.plot(X_np[i, :, 0], alpha=0.7, label=f"Series {i+1}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.legend()

        else:
            n_feat = min(6, X_np.shape[1])
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i in range(n_feat):
                axes[i].hist(X_np[:, i], bins=30, alpha=0.7)
                axes[i].set_title(f"Feature {i+1} Distribution")
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
            for i in range(n_feat, len(axes)):
                axes[i].set_visible(False)

        plt.suptitle(f"{info.name}: {info.description}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def print_dataset_summary(self, datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor, DatasetInfo]]) -> None:
        """Print a summary of all datasets."""
        print("=" * 80)
        print("SAMPLE DATASETS SUMMARY")
        print("=" * 80)
        for name, (X, y, info) in datasets.items():
            print(f"\n📊 {info.name.upper()}")
            print(f"   Type: {info.task_type}")
            print(f"   Samples: {info.num_samples}")
            print(f"   Features: {info.num_features}")
            print(f"   Classes/Outputs: {info.num_classes}")
            print(f"   Difficulty: {info.difficulty}")
            print(f"   Description: {info.description}")
            print(f"   Data Shape: {tuple(X.shape)} -> {tuple(y.shape)}")


def main() -> Dict[str, Tuple[torch.Tensor, torch.Tensor, DatasetInfo]]:
    """Demonstrate the sample dataset generator."""
    generator = SampleDatasetGenerator(seed=42)
    print("🔬 Generating sample datasets...")
    datasets = generator.get_all_sample_datasets()
    generator.print_dataset_summary(datasets)
    print("\n🎨 Visualizing sample datasets...")

    for nm in ["circles", "moons"]:
        if nm in datasets:
            X, y, info = datasets[nm]
            print(f"\nVisualizing {nm}...")
            generator.visualize_dataset(X, y, info)

    print("\n🧪 Testing individual dataset generation...")
    Xc, yc, info_c = generator.generate_classification_dataset(n_samples=500, n_features=8, n_classes=3, difficulty="medium")
    print(f"Generated {info_c.name}: {Xc.shape} -> {yc.shape}")

    Xr, yr, info_r = generator.generate_regression_dataset(n_samples=400, n_features=12, difficulty="hard", multi_output=True)
    print(f"Generated {info_r.name}: {Xr.shape} -> {yr.shape}")

    Xts, yts, info_ts = generator.generate_time_series_dataset(n_samples=200, sequence_length=25, pattern_type="sine")
    print(f"Generated {info_ts.name}: {Xts.shape} -> {yts.shape}")

    print("\n✅ Sample dataset generation completed!")
    return datasets


if __name__ == "__main__":
    _ = main()
