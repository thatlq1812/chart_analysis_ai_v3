"""
Training Infrastructure Module.

Provides run management, config resolution, and experiment tracking
for all training scripts (SLM, ResNet, YOLO).

Key Components:
    RunManager: Handles run isolation, config freezing, artifact management
    ExperimentTracker: Abstracts WandB / TensorBoard / JSON logging

Usage:
    from src.training import RunManager, ExperimentTracker

    manager = RunManager(
        config_path="config/training.yaml",
        cli_overrides=["slm_training.training.learning_rate=1e-5"],
        run_prefix="slm_lora",
    )
    tracker = ExperimentTracker(
        backend="wandb",
        project="chart_analysis_ai_v3",
        run_name=manager.run_name,
        config=manager.config,
    )
"""

from src.training.run_manager import RunManager
from src.training.experiment_tracker import ExperimentTracker

__all__ = ["RunManager", "ExperimentTracker"]
