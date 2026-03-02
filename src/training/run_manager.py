"""
Run Manager for training experiments.

Handles:
    - Run ID generation and directory isolation
    - Hierarchical config resolution (YAML base + CLI overrides)
    - Config freezing (immutable snapshot per run)
    - Run registry (index of all runs)
    - Artifact path management

Example:
    manager = RunManager(
        config_path="config/training.yaml",
        cli_overrides=["slm_training.training.learning_rate=1e-5"],
        run_prefix="slm_lora_llama-1b",
    )
    print(manager.run_dir)        # runs/slm_lora_llama-1b_20260302_153022/
    print(manager.config)         # Resolved OmegaConf DictConfig
    print(manager.checkpoints_dir)# runs/.../checkpoints/

    # After training completes:
    manager.finalize(metrics={"eval_loss": 0.42, "json_valid_rate": 0.96})
"""

import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Project root (2 levels up from src/training/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
REGISTRY_FILE = RUNS_DIR / "run_registry.json"


class RunManager:
    """
    Manages isolated training runs with config resolution and artifact tracking.

    Each run creates a unique directory with frozen config, checkpoints,
    logs, and metadata. A global registry tracks all runs for comparison.

    Attributes:
        run_id: Unique timestamp-based identifier (e.g. '20260302_153022')
        run_name: Human-readable name (e.g. 'slm_lora_llama-1b_20260302_153022')
        run_dir: Absolute path to the run's isolated directory
        config: Resolved OmegaConf DictConfig (frozen after init)
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        cli_overrides: Optional[List[str]] = None,
        run_prefix: str = "run",
        base_dir: Optional[Union[str, Path]] = None,
        extra_configs: Optional[List[Union[str, Path]]] = None,
    ) -> None:
        """
        Initialize a new training run.

        Args:
            config_path: Path to primary YAML config file
            cli_overrides: List of OmegaConf dot-notation overrides
                          (e.g. ["slm_training.training.learning_rate=1e-5"])
            run_prefix: Prefix for run directory name
            base_dir: Override the default runs/ directory
            extra_configs: Additional YAML files to merge (in order)
        """
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{run_prefix}_{self.run_id}"
        self._base_dir = Path(base_dir) if base_dir else RUNS_DIR
        self.run_dir = self._base_dir / self.run_name
        self._started_at = datetime.now()
        self._finalized = False

        # Resolve config hierarchy
        self.config = self._resolve_config(
            config_path=Path(config_path),
            cli_overrides=cli_overrides or [],
            extra_configs=[Path(p) for p in (extra_configs or [])],
        )

        # Create directory structure
        self._create_run_dirs()

        # Freeze config
        self._freeze_config()

        # Save run metadata
        self._save_metadata(
            config_path=str(config_path),
            cli_overrides=cli_overrides or [],
            extra_configs=[str(p) for p in (extra_configs or [])],
        )

        logger.info(
            f"Run initialized | run_id={self.run_id} | "
            f"run_dir={self.run_dir} | config_hash={self.config_hash[:12]}"
        )

    # -------------------------------------------------------------------------
    # Config Resolution
    # -------------------------------------------------------------------------

    def _resolve_config(
        self,
        config_path: Path,
        cli_overrides: List[str],
        extra_configs: List[Path],
    ) -> DictConfig:
        """
        Resolve config from YAML files + CLI overrides.

        Merge order (later overrides earlier):
            1. base.yaml (if exists in same directory)
            2. config_path (primary YAML)
            3. extra_configs (additional YAMLs, in order)
            4. cli_overrides (highest priority)

        Args:
            config_path: Primary YAML config file
            cli_overrides: CLI dot-notation overrides
            extra_configs: Additional YAML files

        Returns:
            Resolved DictConfig with all overrides applied
        """
        configs: List[DictConfig] = []

        # Layer 1: base.yaml (project-wide defaults)
        base_yaml = config_path.parent / "base.yaml"
        if base_yaml.exists() and base_yaml != config_path:
            configs.append(OmegaConf.load(str(base_yaml)))
            logger.info(f"Loaded base config | path={base_yaml}")

        # Layer 2: Primary config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        configs.append(OmegaConf.load(str(config_path)))
        logger.info(f"Loaded primary config | path={config_path}")

        # Layer 3: Extra configs
        for extra_path in extra_configs:
            if extra_path.exists():
                configs.append(OmegaConf.load(str(extra_path)))
                logger.info(f"Loaded extra config | path={extra_path}")
            else:
                logger.warning(f"Extra config not found, skipping | path={extra_path}")

        # Layer 4: CLI overrides
        if cli_overrides:
            cli_conf = OmegaConf.from_dotlist(cli_overrides)
            configs.append(cli_conf)
            logger.info(f"Applied CLI overrides | count={len(cli_overrides)}")

        # Merge all layers
        resolved = OmegaConf.merge(*configs)

        return resolved

    @property
    def config_hash(self) -> str:
        """SHA-256 hash of the resolved config for reproducibility tracking."""
        config_str = OmegaConf.to_yaml(self.config, resolve=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_config_value(self, dotpath: str, default: Any = None) -> Any:
        """
        Get a config value by dot-separated path.

        Args:
            dotpath: Dot-separated key path (e.g. "slm_training.training.learning_rate")
            default: Default value if key not found

        Returns:
            Config value or default
        """
        try:
            return OmegaConf.select(self.config, dotpath, default=default)
        except Exception:
            return default

    # -------------------------------------------------------------------------
    # Directory Management
    # -------------------------------------------------------------------------

    def _create_run_dirs(self) -> None:
        """Create the isolated run directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)

    @property
    def checkpoints_dir(self) -> Path:
        """Directory for model checkpoints."""
        return self.run_dir / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        """Directory for training logs (TensorBoard, CSV)."""
        return self.run_dir / "logs"

    @property
    def artifacts_dir(self) -> Path:
        """Directory for evaluation results, plots, and other artifacts."""
        return self.run_dir / "artifacts"

    @property
    def final_model_dir(self) -> Path:
        """Directory for the final trained model weights."""
        return self.run_dir / "final"

    # -------------------------------------------------------------------------
    # Config Freezing
    # -------------------------------------------------------------------------

    def _freeze_config(self) -> None:
        """Save the resolved config as an immutable YAML snapshot."""
        frozen_path = self.run_dir / "resolved_config.yaml"
        OmegaConf.save(config=self.config, f=str(frozen_path))
        logger.info(f"Config frozen | path={frozen_path}")

    # -------------------------------------------------------------------------
    # Run Metadata
    # -------------------------------------------------------------------------

    def _save_metadata(
        self,
        config_path: str,
        cli_overrides: List[str],
        extra_configs: List[str],
    ) -> None:
        """Save run metadata (timestamps, git info, config sources)."""
        metadata = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": self._started_at.isoformat(),
            "config_sources": {
                "primary": config_path,
                "extra": extra_configs,
                "cli_overrides": cli_overrides,
            },
            "config_hash": self.config_hash,
            "git": self._get_git_info(),
            "status": "running",
        }

        metadata_path = self.run_dir / "run_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _get_git_info(self) -> Dict[str, Optional[str]]:
        """Get current git commit hash and branch for reproducibility."""
        info: Dict[str, Optional[str]] = {"commit": None, "branch": None, "dirty": None}
        try:
            info["commit"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            info["branch"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(PROJECT_ROOT),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            dirty_check = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["dirty"] = str(bool(dirty_check))
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Git info unavailable")
        return info

    # -------------------------------------------------------------------------
    # Run Finalization
    # -------------------------------------------------------------------------

    def finalize(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "completed",
    ) -> None:
        """
        Finalize the run: update metadata, register in global registry.

        Args:
            metrics: Final metrics dict (e.g. {"eval_loss": 0.42})
            status: Final status ("completed", "failed", "interrupted")
        """
        if self._finalized:
            logger.warning(f"Run already finalized | run_id={self.run_id}")
            return

        finished_at = datetime.now()
        duration_seconds = (finished_at - self._started_at).total_seconds()

        # Update metadata
        metadata_path = self.run_dir / "run_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata.update({
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "duration_human": self._format_duration(duration_seconds),
            "status": status,
            "final_metrics": metrics or {},
        })

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Update global registry
        self._update_registry(metadata)

        self._finalized = True
        logger.info(
            f"Run finalized | run_id={self.run_id} | status={status} | "
            f"duration={metadata['duration_human']}"
        )

    def _update_registry(self, metadata: Dict[str, Any]) -> None:
        """Append run summary to the global registry file."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        registry_path = self._base_dir / "run_registry.json"

        registry: List[Dict[str, Any]] = []
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    registry = json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not read registry, creating new | path={registry_path}")

        entry = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "started_at": metadata.get("started_at"),
            "finished_at": metadata.get("finished_at"),
            "duration_human": metadata.get("duration_human"),
            "status": metadata.get("status"),
            "config_hash": metadata.get("config_hash"),
            "final_metrics": metadata.get("final_metrics", {}),
        }

        # Replace existing entry (if re-finalized) or append
        existing_idx = next(
            (i for i, r in enumerate(registry) if r["run_id"] == self.run_id),
            None,
        )
        if existing_idx is not None:
            registry[existing_idx] = entry
        else:
            registry.append(entry)

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(f"Registry updated | total_runs={len(registry)}")

    def save_artifact(self, data: Any, filename: str) -> Path:
        """
        Save an artifact (JSON, text, etc.) to the run's artifacts directory.

        Args:
            data: Data to save. Dict/list -> JSON, str -> text file
            filename: Filename (e.g. "eval_results.json")

        Returns:
            Path to saved artifact
        """
        artifact_path = self.artifacts_dir / filename

        if isinstance(data, (dict, list)):
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif isinstance(data, str):
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(data)
        else:
            raise TypeError(f"Unsupported artifact type: {type(data)}. Use dict, list, or str.")

        logger.info(f"Artifact saved | path={artifact_path}")
        return artifact_path

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def list_runs(
        base_dir: Optional[Union[str, Path]] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all runs from the registry.

        Args:
            base_dir: Override default runs/ directory
            status: Filter by status (e.g. "completed")

        Returns:
            List of run summary dicts
        """
        registry_path = Path(base_dir or RUNS_DIR) / "run_registry.json"
        if not registry_path.exists():
            return []

        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        if status:
            registry = [r for r in registry if r.get("status") == status]

        return registry

    @staticmethod
    def load_run_config(
        run_dir: Union[str, Path],
    ) -> DictConfig:
        """
        Load the frozen config from an existing run.

        Args:
            run_dir: Path to the run directory

        Returns:
            Frozen DictConfig

        Raises:
            FileNotFoundError: If resolved_config.yaml not found
        """
        config_path = Path(run_dir) / "resolved_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Frozen config not found: {config_path}")
        return OmegaConf.load(str(config_path))

    def __repr__(self) -> str:
        return (
            f"RunManager(run_name='{self.run_name}', "
            f"run_dir='{self.run_dir}', "
            f"config_hash='{self.config_hash[:12]}...')"
        )
