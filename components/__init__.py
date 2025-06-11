"""
Dynamic Neural Network Compiler Package
======================================

A comprehensive system for automatically generating optimized PyTorch neural networks
based on task characteristics using meta-learning, model compression, and AutoML.

Main Components:
- TaskAnalyzer: Analyzes input data and task requirements
- ArchitectureGenerator: Generates neural network architectures
- MetaLearner: Applies meta-learning for initialization
- ModelCompressor: Compresses models using pruning and quantization
- AutoMLOptimizer: Optimizes hyperparameters and architecture
- NeuralCompiler: Final compilation and optimization
- DynamicNeuralNetworkCompiler: Main orchestrator class

Usage:
    from dynamic_nn_compiler import DynamicNeuralNetworkCompiler
    
    compiler = DynamicNeuralNetworkCompiler()
    result = compiler.compile_model(data, labels, task_description)
"""

__version__ = "1.0.0"
__author__ = "Dynamic NN Compiler Team"
__email__ = "support@dynamic-nn-compiler.com"

# Core imports
from .task_analyzer import (
    TaskAnalyzer,
    TaskType,
    DataComplexity,
    TaskCharacteristics
)

from .architecture_generator import (
    ArchitectureGenerator,
    NetworkTemplate,
    LayerConfig,
    ArchitectureSearchSpace,
    ResidualBlock,
    AttentionBlock,
    ConvolutionalBlock
)

from .meta_learner import (
    MetaLearner,
    TaskMetadata,
    MetaKnowledgeBase,
    SimilarityMetrics,
    MetaInitializer
)

from .model_compressor import (
    ModelCompressor,
    CompressionConfig,
    PruningStrategy,
    QuantizationConfig,
    KnowledgeDistillation
)

from .automl_optimizer import (
    AutoMLOptimizer,
    OptimizationConfig,
    HyperparameterSpace,
    BayesianOptimizer,
    EvolutionaryOptimizer
)

from .neural_compiler import (
    NeuralCompiler,
    CompilationConfig,
    OptimizationPass,
    GraphOptimizer,
    MemoryOptimizer
)

from .dynamic_nn_compiler import (
    DynamicNeuralNetworkCompiler
)

# Utility imports
from .utils import (
    ModelUtils,
    DataUtils,
    VisualizationUtils,
    MetricsUtils,
    ConfigManager
)

# Exception classes
from .exceptions import (
    CompilerError,
    TaskAnalysisError,
    ArchitectureGenerationError,
    CompressionError,
    OptimizationError
)

# Configuration and constants
from .config import (
    DEFAULT_CONFIG,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_LOSS_FUNCTIONS,
    MODEL_SIZE_LIMITS,
    PERFORMANCE_THRESHOLDS
)

# Export all public classes and functions
__all__ = [
    # Main classes
    'DynamicNeuralNetworkCompiler',
    'TaskAnalyzer',
    'ArchitectureGenerator', 
    'MetaLearner',
    'ModelCompressor',
    'AutoMLOptimizer',
    'NeuralCompiler',
    
    # Data classes and enums
    'TaskType',
    'DataComplexity',
    'TaskCharacteristics',
    'NetworkTemplate',
    'LayerConfig',
    'TaskMetadata',
    'CompressionConfig',
    'OptimizationConfig',
    'CompilationConfig',
    
    # Utility classes
    'ModelUtils',
    'DataUtils',
    'VisualizationUtils',
    'MetricsUtils',
    'ConfigManager',
    
    # Exception classes
    'CompilerError',
    'TaskAnalysisError',
    'ArchitectureGenerationError',
    'CompressionError',
    'OptimizationError',
    
    # Configuration
    'DEFAULT_CONFIG',
    'SUPPORTED_ACTIVATIONS',
    'SUPPORTED_OPTIMIZERS',
    'SUPPORTED_LOSS_FUNCTIONS',
    'MODEL_SIZE_LIMITS',
    'PERFORMANCE_THRESHOLDS',
]

def get_version():
    """Return the version of the package."""
    return __version__

def get_supported_task_types():
    """Return list of supported task types."""
    return [task_type.value for task_type in TaskType]

def get_default_config():
    """Return the default configuration dictionary."""
    return DEFAULT_CONFIG.copy()

def create_compiler(config=None, **kwargs):
    """
    Factory function to create a DynamicNeuralNetworkCompiler instance.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        DynamicNeuralNetworkCompiler instance
    """
    if config is None:
        config = get_default_config()
    config.update(kwargs)
    return DynamicNeuralNetworkCompiler(config)

def quick_compile(data, labels, task_description="", **kwargs):
    """
    Quick compilation function for simple use cases.
    
    Args:
        data: Input training data tensor
        labels: Target labels tensor
        task_description: Optional task description
        **kwargs: Additional configuration parameters
        
    Returns:
        Compiled model result dictionary
    """
    compiler = create_compiler(**kwargs)
    return compiler.compile_model(data, labels, task_description)

# Package-level configuration
import logging

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version compatibility check
import sys
if sys.version_info < (3, 7):
    raise RuntimeError("Dynamic Neural Network Compiler requires Python 3.7 or higher")

# PyTorch version check
try:
    import torch
    if torch.__version__ < "1.8.0":
        import warnings
        warnings.warn(
            "Dynamic Neural Network Compiler is optimized for PyTorch 1.8.0+. "
            f"Current version: {torch.__version__}",
            UserWarning
        )
except ImportError:
    raise ImportError("PyTorch is required but not installed")

# Optional dependency checks
_OPTIONAL_DEPENDENCIES = {
    'scikit-learn': 'sklearn',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'tensorboard': 'tensorboard',
    'wandb': 'wandb'
}

_AVAILABLE_DEPENDENCIES = {}
for name, import_name in _OPTIONAL_DEPENDENCIES.items():
    try:
        __import__(import_name)
        _AVAILABLE_DEPENDENCIES[name] = True
    except ImportError:
        _AVAILABLE_DEPENDENCIES[name] = False

def check_dependencies():
    """Check which optional dependencies are available."""
    return _AVAILABLE_DEPENDENCIES.copy()

def get_info():
    """Get package information."""
    info = {
        'version': __version__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'torch_version': torch.__version__,
        'available_dependencies': check_dependencies(),
        'supported_task_types': get_supported_task_types(),
    }
    return info

# Package initialization message
if __name__ != "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    # Print welcome message only in interactive mode
    try:
        if hasattr(sys, 'ps1'):  # Interactive mode
            print(f"Dynamic Neural Network Compiler v{__version__} initialized")
            missing_deps = [name for name, available in _AVAILABLE_DEPENDENCIES.items() if not available]
            if missing_deps:
                print(f"Optional dependencies not found: {', '.join(missing_deps)}")
    except:
        pass

# Cleanup namespace
del sys, logging, warnings
