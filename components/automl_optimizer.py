"""
--- Updated automl_optimizer.py ---
AutoML Optimizer Component
Automatically optimizes neural network hyperparameters and architecture choices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import random
import time
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


@dataclass
class OptimizationConfig:
    """Configuration for AutoML optimization process."""
    max_trials: int = 100
    timeout_seconds: int = 3600
    cv_folds: int = 3
    early_stopping_patience: int = 10
    metric_direction: str = "maximize"  # "maximize" or "minimize"
    primary_metric: str = "accuracy"
    enable_pruning: bool = True
    search_space: Optional[Dict[str, Any]] = None


class HyperparameterSearchSpace:
    """Defines the search space for hyperparameters."""

    @staticmethod
    def get_default_search_space() -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameter search space."""
        return {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True,
            },
            "batch_size": {
                "type": "categorical",
                "choices": [16, 32, 64, 128, 256],
            },
            "optimizer": {
                "type": "categorical",
                "choices": ["adam", "sgd", "rmsprop", "adamw"],
            },
            "weight_decay": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-2,
                "log": True,
            },
            "dropout_rate": {
                "type": "float",
                "low": 0.0,
                "high": 0.5,
            },
            "hidden_layers": {
                "type": "int",
                "low": 1,
                "high": 5,
            },
            "hidden_size": {
                "type": "int",
                "low": 32,
                "high": 512,
                "step": 32,
            },
            "activation": {
                "type": "categorical",
                "choices": ["relu", "leaky_relu", "swish", "gelu", "tanh"],
            },
            "scheduler": {
                "type": "categorical",
                "choices": ["none", "cosine", "step", "exponential", "plateau"],
            },
        }


class ModelTrainer:
    """Handles model training with early stopping and validation."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def train_and_evaluate(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        max_epochs: int = 100,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Train model and return evaluation metrics."""

        model = model.to(self.device)

        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)
        criterion = self._create_loss_function(config)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0

        train_losses: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            total_val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()

                    if outputs.dim() > 1 and outputs.shape[1] > 1:
                        # Classification
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == batch_y).sum().item()
                    else:
                        # Regression: use R² as "accuracy"
                        ss_res = ((batch_y - outputs.squeeze()) ** 2).sum()
                        ss_tot = ((batch_y - batch_y.mean()) ** 2).sum()
                        r2 = 1 - (ss_res / (ss_tot + 1e-8))
                        correct += r2.item() * len(batch_y)
                    total += batch_y.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return {
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "epochs_trained": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }

    def _create_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 0.001)
        weight_decay = config.get("weight_decay", 1e-4)

        if name == "adam":
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            momentum = config.get("momentum", 0.9)
            return optim.SGD(
                model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        elif name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _create_scheduler(
        self, optimizer: optim.Optimizer, config: Dict[str, Any]
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = config.get("scheduler", "none").lower()

        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        else:
            return None

    def _create_loss_function(self, config: Dict[str, Any]) -> nn.Module:
        """Create appropriate loss function."""
        loss_name = config.get("loss_function", "crossentropy").lower()

        if loss_name == "crossentropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "binary_crossentropy":
            return nn.BCEWithLogitsLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.SmoothL1Loss()
        else:
            return nn.CrossEntropyLoss()


class ArchitectureSearcher:
    """Searches for optimal neural network architectures."""

    def __init__(self, task_characteristics):
        self.task_chars = task_characteristics

    def create_model_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create a model based on hyperparameter configuration."""
        input_size = self.task_chars.num_features
        output_size = self.task_chars.num_classes if self.task_chars.num_classes > 1 else 1

        hidden_layers = config.get("hidden_layers", 2)
        hidden_size = config.get("hidden_size", 128)
        dropout_rate = config.get("dropout_rate", 0.1)
        activation = config.get("activation", "relu")

        layers: List[nn.Module] = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        if self.task_chars.task_type.value == "classification" and output_size == 1:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "swish": nn.SiLU(),  # SiLU is Swish
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(name, nn.ReLU())


class AutoMLOptimizer:
    """
    Main AutoML optimizer that combines hyperparameter optimization
    with neural architecture search.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.trainer = ModelTrainer()
        self.search_space = HyperparameterSearchSpace.get_default_search_space()
        if self.config.search_space:
            self.search_space.update(self.config.search_space)
        self.study: Optional[optuna.Study] = None
        self.best_model: Optional[nn.Module] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.optimization_history: List[Dict[str, Any]] = []

    def optimize(
        self,
        candidate_models: List[nn.Module],
        data: torch.Tensor,
        labels: torch.Tensor,
        task_characteristics: Any,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Main optimization method that finds the best model and hyperparameters.

        Args:
            candidate_models: List of candidate model architectures (unused in current implementation)
            data: Training data
            labels: Training labels
            task_characteristics: Task analysis results

        Returns:
            Tuple of (best_model, best_config)
        """
        print(f"🔍 Starting AutoML optimization...")

        arch_searcher = ArchitectureSearcher(task_characteristics)
        direction = self.config.metric_direction

        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner() if self.config.enable_pruning else None,
        )

        dataset = TensorDataset(data, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        def objective(trial: optuna.Trial) -> float:
            return self._objective_function(
                trial, arch_searcher, train_dataset, val_dataset, task_characteristics
            )

        start_time = time.time()
        self.study.optimize(
            objective,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout_seconds,
            show_progress_bar=True,
        )
        optimization_time = time.time() - start_time

        self.best_config = self.study.best_params
        self.best_model = arch_searcher.create_model_from_config(self.best_config)

        train_loader = DataLoader(
            train_dataset, batch_size=self.best_config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.best_config["batch_size"]
        )

        final_metrics = self.trainer.train_and_evaluate(
            self.best_model, train_loader, val_loader, self.best_config
        )

        print(f"✅ Optimization completed in {optimization_time:.2f}s")
        print(f"🏆 Best {self.config.primary_metric}: {self.study.best_value:.4f}")
        print(f"🔧 Best config: {self.best_config}")

        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "best_value": self.study.best_value,
                "best_params": self.best_config,
                "n_trials": len(self.study.trials),
                "optimization_time": optimization_time,
                "final_metrics": final_metrics,
            }
        )
        return self.best_model, self.best_config

    def _objective_function(
        self,
        trial: optuna.Trial,
        arch_searcher: ArchitectureSearcher,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        task_characteristics: Any,
    ) -> float:
        """Objective function for Optuna optimization."""
        config: Dict[str, Any] = {}
        for param_name, param_cfg in self.search_space.items():
            if param_cfg["type"] == "float":
                if param_cfg.get("log", False):
                    config[param_name] = trial.suggest_float(
                        param_name, param_cfg["low"], param_cfg["high"], log=True
                    )
                else:
                    config[param_name] = trial.suggest_float(
                        param_name, param_cfg["low"], param_cfg["high"]
                    )
            elif param_cfg["type"] == "int":
                config[param_name] = trial.suggest_int(
                    param_name,
                    param_cfg["low"],
                    param_cfg["high"],
                    step=param_cfg.get("step", 1),
                )
            elif param_cfg["type"] == "categorical":
                config[param_name] = trial.suggest_categorical(
                    param_name, param_cfg["choices"]
                )

        if task_characteristics.task_type.value == "classification":
            config["loss_function"] = (
                "crossentropy"
                if task_characteristics.num_classes > 2
                else "binary_crossentropy"
            )
        else:
            config["loss_function"] = "mse"

        model = arch_searcher.create_model_from_config(config)
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

        try:
            metrics = self.trainer.train_and_evaluate(
                model,
                train_loader,
                val_loader,
                config,
                max_epochs=50,
                patience=self.config.early_stopping_patience,
            )
            if self.config.primary_metric == "accuracy":
                return metrics["best_val_accuracy"]
            elif self.config.primary_metric == "loss":
                return -metrics["best_val_loss"]
            else:
                return metrics.get(self.config.primary_metric, 0.0)
        except Exception as e:
            print(f"Trial failed: {e}")
            return -1.0 if self.config.metric_direction == "maximize" else 1e6

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of the optimization process."""
        if not self.study:
            return {"error": "No optimization has been run yet"}
        completed = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        values = [t.value for t in completed] if completed else []
        summary = {
            "n_trials": len(self.study.trials),
            "best_value": self.study.best_value,
            "best_params": self.study.best_params,
            "optimization_history": self.optimization_history,
        }
        if values:
            summary["trial_statistics"] = {
                "mean_performance": np.mean(values),
                "std_performance": np.std(values),
                "min_performance": np.min(values),
                "max_performance": np.max(values),
            }
        return summary

    def plot_optimization_history(self) -> None:
        """Plot optimization history (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization.matplotlib as opt_viz

            if not self.study:
                print("No optimization history to plot")
                return

            opt_viz.plot_optimization_history(self.study)
            plt.title("AutoML Optimization History")
            plt.show()

            opt_viz.plot_param_importances(self.study)
            plt.title("Hyperparameter Importances")
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting optimization history: {e}")

    def cross_validate_best_model(
        self, data: torch.Tensor, labels: torch.Tensor, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform cross-validation on the best model."""
        if self.best_model is None or self.best_config is None:
            raise ValueError("No best model found. Run optimization first.")

        dataset = TensorDataset(data, labels)
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=self.best_config["batch_size"], shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=self.best_config["batch_size"]
            )
            arch_searcher = ArchitectureSearcher(self.task_chars)
            fold_model = arch_searcher.create_model_from_config(self.best_config)
            metrics = self.trainer.train_and_evaluate(
                fold_model, train_loader, val_loader, self.best_config
            )
            cv_scores.append(metrics["best_val_accuracy"])

        return {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "cv_scores": cv_scores,
            "cv_min": float(np.min(cv_scores)),
            "cv_max": float(np.max(cv_scores)),
        }


# Example usage and testing
def test_automl_optimizer():
    """Test the AutoML optimizer with sample data."""
    X = torch.randn(500, 10)
    y = torch.randint(0, 3, (500,))

    from enum import Enum

    class TaskType(Enum):
        CLASSIFICATION = "classification"

    @dataclass
    class MockTaskChars:
        task_type: TaskType = TaskType.CLASSIFICATION
        num_features: int = 20
        num_classes: int = 3

    task_chars = MockTaskChars()
    config = OptimizationConfig(
        max_trials=20, timeout_seconds=300, primary_metric="accuracy"
    )
    optimizer = AutoMLOptimizer(config)
    best_model, best_config = optimizer.optimize([], X, y, task_chars)
    print("AutoML optimization completed successfully!")
    print(f"Best configuration: {best_config}")
    summary = optimizer.get_optimization_summary()
    print(f"Optimization summary: {summary}")


if __name__ == "__main__":
    test_automl_optimizer()
