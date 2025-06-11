"""
--- Updated meta_learner.py ---
Meta-Learner Component
Implements meta-learning algorithms to quickly adapt to new tasks based on prior experience.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import pickle
from collections import defaultdict, deque
from abc import ABC, abstractmethod


@dataclass
class TaskMetadata:
    """Metadata about a task for meta-learning."""
    task_id: str
    task_characteristics: Dict[str, Any]
    architecture_config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    convergence_time: float
    final_weights: Optional[Dict[str, torch.Tensor]] = None


class MetaLearningAlgorithm(ABC):
    """Abstract base class for meta-learning algorithms."""

    @abstractmethod
    def learn_from_task(self, task_metadata: TaskMetadata) -> None:
        """Learn from a completed task."""
        pass

    @abstractmethod
    def initialize_new_task(
        self, task_chars: Dict[str, Any], architecture: nn.Module
    ) -> nn.Module:
        """Initialize a new task based on meta-knowledge."""
        pass


class MAMLLearner(MetaLearningAlgorithm):
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(self, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.meta_weights: Dict[str, torch.Tensor] = {}
        self.task_gradients: List[Dict[str, torch.Tensor]] = []

    def learn_from_task(self, task_metadata: TaskMetadata) -> None:
        """Store task gradients for meta-update."""
        if task_metadata.final_weights:
            self.task_gradients.append(task_metadata.final_weights)
            if len(self.task_gradients) >= 5:
                self._meta_update()
                self.task_gradients.clear()

    def _meta_update(self) -> None:
        """Update meta-parameters based on accumulated task gradients."""
        if not self.task_gradients:
            return

        # Average gradients across tasks
        meta_grad: Dict[str, torch.Tensor] = {}
        for name in self.task_gradients[0].keys():
            grads = [tg[name] for tg in self.task_gradients]
            meta_grad[name] = torch.stack(grads, dim=0).mean(dim=0)

        # Update meta-weights
        for name, grad in meta_grad.items():
            if name not in self.meta_weights:
                self.meta_weights[name] = torch.zeros_like(grad)
            self.meta_weights[name] -= self.meta_lr * grad

    def initialize_new_task(
        self, task_chars: Dict[str, Any], architecture: nn.Module
    ) -> nn.Module:
        """Initialize architecture with meta-learned weights."""
        if self.meta_weights:
            state = architecture.state_dict()
            for name, param in state.items():
                if name in self.meta_weights:
                    state[name] = self.meta_weights[name].clone()
            architecture.load_state_dict(state)
        return architecture


class ProtoNetLearner(MetaLearningAlgorithm):
    """Prototypical Networks for few-shot learning."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.prototypes: Dict[str, List[np.ndarray]] = {}
        self.task_embeddings: Dict[str, Dict[str, Any]] = {}

    def learn_from_task(self, task_metadata: TaskMetadata) -> None:
        """Learn task prototypes."""
        signature = self._compute_task_signature(task_metadata.task_characteristics)
        task_type = task_metadata.task_characteristics.get("task_type", "unknown")
        if task_type not in self.prototypes:
            self.prototypes[task_type] = []
        self.prototypes[task_type].append(signature)
        if len(self.prototypes[task_type]) > 10:
            self.prototypes[task_type] = self.prototypes[task_type][-10:]

        self.task_embeddings[task_metadata.task_id] = {
            "signature": signature,
            "performance": task_metadata.performance_metrics,
            "architecture": task_metadata.architecture_config,
        }

    def _compute_task_signature(self, task_chars: Dict[str, Any]) -> np.ndarray:
        """Compute a signature vector for the task."""
        signature = np.zeros(self.embedding_dim)
        idx = 0

        # Encode task type (one-hot up to 3 types)
        type_map = {"classification": 0, "regression": 1, "multi_label": 2}
        t_val = type_map.get(task_chars.get("task_type", ""), 0)
        signature[idx : idx + 3] = np.eye(3)[t_val]
        idx += 3

        # Dataset size
        if idx < self.embedding_dim:
            signature[idx] = np.log(task_chars.get("dataset_size", 1000)) / 10
            idx += 1

        # Number of features
        if idx < self.embedding_dim:
            signature[idx] = np.log(task_chars.get("num_features", 10)) / 5
            idx += 1

        # Number of classes
        if idx < self.embedding_dim:
            signature[idx] = task_chars.get("num_classes", 1) / 10
            idx += 1

        # Complexity one-hot (4 categories)
        comp_map = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        c_val = comp_map.get(task_chars.get("data_complexity", "medium"), 1)
        if idx + 4 <= self.embedding_dim:
            signature[idx : idx + 4] = np.eye(4)[c_val]
            idx += 4

        # Fill remaining with small noise
        remaining = self.embedding_dim - idx
        if remaining > 0:
            signature[idx:] = np.random.randn(remaining) * 0.1

        return signature

    def initialize_new_task(
        self, task_chars: Dict[str, Any], architecture: nn.Module
    ) -> nn.Module:
        """Initialize based on closest prototype."""
        signature = self._compute_task_signature(task_chars)
        task_type = task_chars.get("task_type", "")
        if task_type in self.prototypes and self.prototypes[task_type]:
            prototypes = np.stack(self.prototypes[task_type], axis=0)
            distances = np.linalg.norm(prototypes - signature, axis=1)
            closest_idx = int(np.argmin(distances))
            # Conservative initialization: Xavier or zeros
            for param in architecture.parameters():
                if param.ndim > 1:
                    nn.init.xavier_normal_(param, gain=0.8)
                else:
                    nn.init.zeros_(param)
        return architecture


class MetaLearner:
    """
    Main meta-learning system that orchestrates different meta-learning algorithms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.maml_learner = MAMLLearner(
            inner_lr=self.config["inner_lr"], meta_lr=self.config["meta_lr"]
        )
        self.protonet_learner = ProtoNetLearner(
            embedding_dim=self.config["embedding_dim"]
        )
        self.task_memory: deque = deque(maxlen=self.config["max_memory_size"])
        self.task_database: Dict[str, TaskMetadata] = {}
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for meta-learner."""
        return {
            "inner_lr": 0.01,
            "meta_lr": 0.001,
            "embedding_dim": 128,
            "max_memory_size": 1000,
            "similarity_threshold": 0.8,
            "adaptation_steps": 5,
            "enable_maml": True,
            "enable_protonet": True,
        }

    def learn_from_task(
        self,
        task_chars: Dict[str, Any],
        architecture_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        training_history: Optional[List[Dict[str, float]]] = None,
        final_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Learn from a completed task."""
        task_id = f"task_{len(self.task_database)}"
        metadata = TaskMetadata(
            task_id=task_id,
            task_characteristics=task_chars,
            architecture_config=architecture_config,
            performance_metrics=performance_metrics,
            training_history=training_history or [],
            convergence_time=performance_metrics.get("training_time", 0.0),
            final_weights=final_weights,
        )
        self.task_memory.append(metadata)
        self.task_database[task_id] = metadata

        if self.config["enable_maml"]:
            self.maml_learner.learn_from_task(metadata)
        if self.config["enable_protonet"]:
            self.protonet_learner.learn_from_task(metadata)

        print(f"📚 Meta-learner updated with task {task_id}")

    def initialize_from_similar_tasks(
        self, architecture: nn.Module, task_chars: Dict[str, Any]
    ) -> nn.Module:
        """Initialize architecture based on similar tasks."""
        similar = self._find_similar_tasks(task_chars)
        if not similar:
            print("🔄 No similar tasks found; using default initialization")
            return self._default_initialize(architecture)

        best_task = similar[0]
        sim_score = best_task["similarity"]
        print(f"🎯 Found {len(similar)} similar tasks; top similarity = {sim_score:.3f}")

        if sim_score > self.config["similarity_threshold"]:
            if self.config["enable_maml"] and self.maml_learner.meta_weights:
                print("🧠 Using MAML initialization")
                return self.maml_learner.initialize_new_task(task_chars, architecture)
            else:
                print("📋 Transferring weights from similar task")
                return self._transfer_weights(architecture, best_task["metadata"])
        else:
            if self.config["enable_protonet"]:
                print("🎭 Using ProtoNet initialization")
                return self.protonet_learner.initialize_new_task(task_chars, architecture)
        return architecture

    def _find_similar_tasks(
        self, task_chars: Dict[str, Any], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar tasks in memory based on similarity score."""
        similarities: List[Dict[str, Any]] = []
        for metadata in self.task_memory:
            sim = self._compute_task_similarity(task_chars, metadata.task_characteristics)
            similarities.append({"metadata": metadata, "similarity": sim})
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]

    def _compute_task_similarity(
        self, chars1: Dict[str, Any], chars2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two tasks."""
        score = 0.0
        weight_sum = 0.0

        # Task type (weight 0.3)
        if chars1.get("task_type") == chars2.get("task_type"):
            score += 0.3
        weight_sum += 0.3

        # Dataset size (weight 0.2)
        s1 = chars1.get("dataset_size", 1000)
        s2 = chars2.get("dataset_size", 1000)
        size_sim = min(s1, s2) / max(s1, s2)
        score += 0.2 * size_sim
        weight_sum += 0.2

        # Feature count (weight 0.2)
        f1 = chars1.get("num_features", 10)
        f2 = chars2.get("num_features", 10)
        feat_sim = min(f1, f2) / max(f1, f2)
        score += 0.2 * feat_sim
        weight_sum += 0.2

        # Complexity (weight 0.15)
        if chars1.get("data_complexity") == chars2.get("data_complexity"):
            score += 0.15
        weight_sum += 0.15

        # Number of classes (weight 0.15)
        c1 = chars1.get("num_classes", 1)
        c2 = chars2.get("num_classes", 1)
        class_sim = min(c1, c2) / max(c1, c2)
        score += 0.15 * class_sim
        weight_sum += 0.15

        return score / weight_sum if weight_sum > 0 else 0.0

    def _transfer_weights(
        self, target_arch: nn.Module, source_metadata: TaskMetadata
    ) -> nn.Module:
        """Transfer compatible weights from a similar task."""
        if not source_metadata.final_weights:
            return target_arch

        tgt_state = target_arch.state_dict()
        src_weights = source_metadata.final_weights

        for name, param in tgt_state.items():
            if name in src_weights:
                src_param = src_weights[name]
                if param.shape == src_param.shape:
                    tgt_state[name] = src_param.clone()
                    print(f"  ✓ Transferred {name}")
                else:
                    # Attempt partial transfer for compatible dims
                    min_shape = tuple(min(dim_t, dim_s) for dim_t, dim_s in zip(param.shape, src_param.shape))
                    if len(min_shape) == 2:
                        tgt_state[name][: min_shape[0], : min_shape[1]] = src_param[: min_shape[0], : min_shape[1]]
                        print(f"  ⚡ Partially transferred {name}")

        target_arch.load_state_dict(tgt_state)
        return target_arch

    def _default_initialize(self, architecture: nn.Module) -> nn.Module:
        """Apply default weight initialization."""
        for module in architecture.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        return architecture

    def get_learning_recommendations(self, task_chars: Dict[str, Any]) -> Dict[str, Any]:
        """Get learning recommendations based on meta-knowledge."""
        similar = self._find_similar_tasks(task_chars, top_k=3)
        if not similar:
            return self._default_recommendations()

        recs: Dict[str, List[Tuple[Any, float]]] = {
            "learning_rate": [],
            "batch_size": [],
            "epochs": [],
            "optimizer": [],
            "regularization": [],
        }
        for info in similar:
            cfg = info["metadata"].architecture_config
            weight = info["similarity"]
            if "learning_rate" in cfg:
                recs["learning_rate"].append((cfg["learning_rate"], weight))
            if "batch_size" in cfg:
                recs["batch_size"].append((cfg["batch_size"], weight))
            if "epochs" in cfg:
                recs["epochs"].append((cfg["epochs"], weight))

        final: Dict[str, Any] = {}
        for key, vals in recs.items():
            if not vals:
                continue
            if key in {"learning_rate", "regularization"}:
                weighted_sum = sum(v * w for v, w in vals)
                total_w = sum(w for _, w in vals)
                final[key] = weighted_sum / (total_w + 1e-8)
            else:
                vals.sort(key=lambda x: x[1], reverse=True)
                final[key] = vals[0][0]

        return final

    def _default_recommendations(self) -> Dict[str, Any]:
        """Default learning recommendations."""
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "regularization": 0.01,
        }

    def update_knowledge(
        self, task_chars: Dict[str, Any], architecture_config: Dict[str, Any], performance_feedback: Dict[str, Any]
    ) -> None:
        """Update meta-knowledge with performance feedback."""
        for metadata in self.task_memory:
            sim = self._compute_task_similarity(task_chars, metadata.task_characteristics)
            if sim > 0.9:
                metadata.performance_metrics.update(performance_feedback)
                algo = performance_feedback.get("algorithm_used")
                perf = performance_feedback.get("final_accuracy", 0.0)
                if algo:
                    self.algorithm_performance[algo].append(perf)
                print("📈 Updated knowledge for similar task")
                break

    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get statistics about meta-learning performance."""
        stats: Dict[str, Any] = {
            "total_tasks_learned": len(self.task_database),
            "memory_utilization": len(self.task_memory) / self.config["max_memory_size"],
            "algorithm_performance": {},
        }
        for algo, perfs in self.algorithm_performance.items():
            if perfs:
                stats["algorithm_performance"][algo] = {
                    "mean_performance": np.mean(perfs),
                    "std_performance": np.std(perfs),
                    "num_tasks": len(perfs),
                }
        return stats

    def save_meta_knowledge(self, filepath: str) -> None:
        """Save meta-learning knowledge to disk."""
        knowledge = {
            "task_database": self.task_database,
            "maml_weights": self.maml_learner.meta_weights,
            "prototypes": self.protonet_learner.prototypes,
            "algorithm_performance": dict(self.algorithm_performance),
            "config": self.config,
        }
        with open(filepath, "wb") as f:
            pickle.dump(knowledge, f)
        print(f"💾 Meta-knowledge saved to {filepath}")

    def load_meta_knowledge(self, filepath: str) -> None:
        """Load meta-learning knowledge from disk."""
        try:
            with open(filepath, "rb") as f:
                knowledge = pickle.load(f)
            self.task_database = knowledge.get("task_database", {})
            self.maml_learner.meta_weights = knowledge.get("maml_weights", {})
            self.protonet_learner.prototypes = knowledge.get("prototypes", {})
            self.algorithm_performance = defaultdict(list, knowledge.get("algorithm_performance", {}))
            self.task_memory.clear()
            for tm in self.task_database.values():
                self.task_memory.append(tm)
            print(f"📚 Meta-knowledge loaded from {filepath}")
        except FileNotFoundError:
            print(f"⚠️ Meta-knowledge file {filepath} not found; starting fresh")
        except Exception as e:
            print(f"❌ Error loading meta-knowledge: {e}")

    def clear_memory(self) -> None:
        """Clear all meta-learning memory."""
        self.task_memory.clear()
        self.task_database.clear()
        self.maml_learner.meta_weights.clear()
        self.protonet_learner.prototypes.clear()
        self.algorithm_performance.clear()
        print("🧹 Meta-learning memory cleared")
