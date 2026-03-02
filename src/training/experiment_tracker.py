"""
Experiment Tracker abstraction layer.

Supports multiple backends for logging training metrics, configs, and artifacts:
    - wandb: Weights & Biases (recommended for academic reporting)
    - tensorboard: TensorBoard (built-in, no account needed)
    - json: Local JSON file logging (always-available fallback)
    - none: No tracking (silent)

The tracker gracefully degrades: if wandb is not installed or login fails,
it automatically falls back to tensorboard, then json.

Example:
    tracker = ExperimentTracker(
        backend="wandb",
        project="chart_analysis_ai_v3",
        run_name="slm_lora_llama-1b_20260302",
        config={"learning_rate": 2e-4, "epochs": 3},
        log_dir=Path("runs/.../logs"),
    )

    for epoch in range(3):
        tracker.log_metrics({"train_loss": 0.5, "val_loss": 0.4}, step=epoch)

    tracker.log_artifact("runs/.../final/", name="model_weights")
    tracker.finish()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking interface with multiple backend support.

    Automatically falls back through backends if preferred one is unavailable:
    wandb -> tensorboard -> json -> none

    Attributes:
        backend: Active backend name
        project: Project name for grouping
        run_name: Unique run identifier
    """

    SUPPORTED_BACKENDS = ("wandb", "tensorboard", "json", "none")

    def __init__(
        self,
        backend: str = "json",
        project: str = "chart_analysis_ai_v3",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize experiment tracker.

        Args:
            backend: Preferred backend ("wandb", "tensorboard", "json", "none")
            project: Project name for experiment grouping
            run_name: Human-readable run identifier
            config: Hyperparameters dict to log at init
            log_dir: Directory for local logs (tensorboard/json)
            tags: Optional tags for filtering runs
        """
        if backend not in self.SUPPORTED_BACKENDS:
            logger.warning(
                f"Unknown backend '{backend}', falling back to 'json' | "
                f"supported={self.SUPPORTED_BACKENDS}"
            )
            backend = "json"

        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._config = config or {}
        self._log_dir = Path(log_dir) if log_dir else None
        self._tags = tags or []
        self._step_metrics: List[Dict[str, Any]] = []
        self._active = False

        # Try to initialize preferred backend, fall back if needed
        self.backend = self._init_backend(backend)
        self._active = True

        logger.info(
            f"ExperimentTracker initialized | backend={self.backend} | "
            f"project={self.project} | run_name={self.run_name}"
        )

    def _init_backend(self, preferred: str) -> str:
        """
        Initialize the preferred backend, falling back if unavailable.

        Args:
            preferred: Preferred backend name

        Returns:
            Name of the actually initialized backend
        """
        fallback_chain = {
            "wandb": ["wandb", "tensorboard", "json"],
            "tensorboard": ["tensorboard", "json"],
            "json": ["json"],
            "none": ["none"],
        }

        for backend in fallback_chain.get(preferred, ["json"]):
            try:
                if backend == "wandb":
                    self._init_wandb()
                    return "wandb"
                elif backend == "tensorboard":
                    self._init_tensorboard()
                    return "tensorboard"
                elif backend == "json":
                    self._init_json()
                    return "json"
                elif backend == "none":
                    return "none"
            except Exception as exc:
                logger.warning(
                    f"Backend '{backend}' initialization failed, trying next | error={exc}"
                )
                continue

        return "none"

    # -------------------------------------------------------------------------
    # Backend Initialization
    # -------------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases tracking."""
        import wandb  # noqa: F811 -- guarded import

        self._wandb = wandb
        self._wandb_run = wandb.init(
            project=self.project,
            name=self.run_name,
            config=self._config,
            tags=self._tags,
            reinit=True,
        )
        logger.info(f"WandB initialized | url={self._wandb_run.get_url()}")

    def _init_tensorboard(self) -> None:
        """Initialize TensorBoard SummaryWriter."""
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = self._log_dir or Path("runs") / self.run_name / "logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # Log config as text
        if self._config:
            config_text = json.dumps(self._config, indent=2, default=str)
            self._tb_writer.add_text("config", f"```\n{config_text}\n```", 0)

        logger.info(f"TensorBoard initialized | log_dir={tb_dir}")

    def _init_json(self) -> None:
        """Initialize JSON file logger."""
        json_dir = self._log_dir or Path("runs") / self.run_name / "logs"
        json_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = json_dir / "metrics.json"

        # Write initial config
        init_data = {
            "project": self.project,
            "run_name": self.run_name,
            "config": self._config,
            "tags": self._tags,
            "started_at": datetime.now().isoformat(),
            "metrics": [],
        }
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(init_data, f, indent=2, default=str)

        logger.info(f"JSON tracker initialized | path={self._json_path}")

    # -------------------------------------------------------------------------
    # Logging Interface
    # -------------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log training metrics for a given step/epoch.

        Args:
            metrics: Dict of metric name -> value (e.g. {"loss": 0.42, "acc": 0.95})
            step: Training step or epoch number
        """
        if not self._active:
            return

        if self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, global_step=step)
        elif self.backend == "json":
            entry = {"step": step, "timestamp": datetime.now().isoformat()}
            entry.update(metrics)
            self._step_metrics.append(entry)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log or update the run configuration.

        Args:
            config: Configuration dict to log
        """
        if not self._active:
            return

        self._config.update(config)

        if self.backend == "wandb":
            self._wandb.config.update(config)
        elif self.backend == "tensorboard":
            config_text = json.dumps(config, indent=2, default=str)
            self._tb_writer.add_text("config_update", f"```\n{config_text}\n```", 0)

    def log_artifact(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
    ) -> None:
        """
        Log an artifact (model weights, evaluation results, etc.).

        Args:
            path: Path to file or directory
            name: Artifact name (defaults to filename)
            artifact_type: Type classification ("model", "dataset", "evaluation")
        """
        if not self._active:
            return

        path = Path(path)
        artifact_name = name or path.name

        if self.backend == "wandb":
            artifact = self._wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
            )
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._wandb_run.log_artifact(artifact)
            logger.info(f"WandB artifact logged | name={artifact_name} | type={artifact_type}")
        elif self.backend == "json":
            entry = {
                "artifact_name": artifact_name,
                "artifact_type": artifact_type,
                "path": str(path),
                "logged_at": datetime.now().isoformat(),
            }
            self._step_metrics.append({"_artifact": entry})

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log final summary metrics (best values, totals).

        Args:
            summary: Summary metrics dict
        """
        if not self._active:
            return

        if self.backend == "wandb":
            for key, value in summary.items():
                self._wandb_run.summary[key] = value
        elif self.backend == "tensorboard":
            # Add as hparams
            metric_dict = {
                k: v for k, v in summary.items() if isinstance(v, (int, float))
            }
            if metric_dict:
                self._tb_writer.add_hparams(
                    hparam_dict=self._config if self._config else {},
                    metric_dict=metric_dict,
                )

    # -------------------------------------------------------------------------
    # HuggingFace Trainer Integration
    # -------------------------------------------------------------------------

    def get_report_to(self) -> List[str]:
        """
        Get the report_to list for HuggingFace TrainingArguments.

        Returns:
            List of tracker names compatible with HF Trainer
            (e.g. ["wandb"], ["tensorboard"], or [])
        """
        if self.backend == "wandb":
            return ["wandb"]
        elif self.backend == "tensorboard":
            return ["tensorboard"]
        return []

    def get_logging_dir(self) -> Optional[str]:
        """
        Get the logging directory for HuggingFace Trainer.

        Returns:
            Path string for TensorBoard logs, or None
        """
        if self.backend == "tensorboard" and self._log_dir:
            return str(self._log_dir)
        return None

    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------

    def finish(self) -> None:
        """Flush all pending data and close the tracker."""
        if not self._active:
            return

        if self.backend == "wandb":
            self._wandb_run.finish()
            logger.info("WandB run finished")
        elif self.backend == "tensorboard":
            self._tb_writer.flush()
            self._tb_writer.close()
            logger.info("TensorBoard writer closed")
        elif self.backend == "json":
            self._flush_json()
            logger.info(f"JSON metrics finalized | path={self._json_path}")

        self._active = False

    def _flush_json(self) -> None:
        """Write accumulated metrics to JSON file."""
        if not hasattr(self, "_json_path"):
            return

        try:
            with open(self._json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {"metrics": []}

        data["metrics"].extend(self._step_metrics)
        data["finished_at"] = datetime.now().isoformat()

        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        self._step_metrics.clear()

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()

    def __repr__(self) -> str:
        return (
            f"ExperimentTracker(backend='{self.backend}', "
            f"project='{self.project}', run_name='{self.run_name}')"
        )
