"""Training script for tarp segmentation.

Usage:
    python -m segmentation.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import random
import time
import traceback
from datetime import timedelta
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import logging
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from pyvexcelutils.aws.secrets import Secrets
from pyvexcelutils.commons.app_base import (
    APP_BASE_EXIT_ERROR,
    APP_BASE_EXIT_OK,
    AppBase,
)
from pyvexcelutils.mlflow.mlflow_helper import setup_mlflow, MLFlowNamer
from segmentation.data.dataset import TarpDataset, ResolutionBatchSampler
from segmentation.evaluate import evaluate
from segmentation.data.transforms import get_train_transforms, get_val_transforms
from segmentation.losses.losses import get_loss
from segmentation.metrics.detailed_metrics import DetailedSegmentationMetrics
from segmentation.models.unet import build_model
import sys

def resolve_device() -> torch.device:
    """Pick the best available device: CUDA → MPS → CPU."""
    logging.info
    return torch.device("cuda") if torch.cuda.is_available() else "cpu"



APP_NAME = "ml_segmentation_model_training"
APP_EPILOG = "This application trains a model for segmentation"
class TrainSegmentationModel(AppBase):
    def __init__(self):
        description = "This application trains a model for segmentation"

        super().__init__(
            app_name=APP_NAME, app_description=description, app_epilog=APP_EPILOG
        )

        self.add_argument(
            "--config",
            type=str,
            required=True,
            help=f"Path to the YAML config file (e.g. configs/default.yaml)",
        )

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def build_dataloaders(self, cfg: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
        data_cfg  = cfg["data"]
        use_fixed_size = data_cfg.get("image_size") is not None
        task       = data_cfg.get("task", "binary")
        batch_size = cfg["training"]["batch_size"]
        seed       = cfg["experiment"].get("seed", 42)
        metadata_dir = data_cfg.get("metadata_dir")
        num_workers  = data_cfg.get("num_workers", 4)
        # pin_memory speeds up host→GPU transfers; not supported on MPS
        pin_memory = device.type == "cuda"

        class_mapper = data_cfg.get("class_mapper")
        max_samples  = data_cfg.get("max_samples")

        train_ds = TarpDataset(
            split_csv=data_cfg["train_csv"],
            task=task,
            transform=get_train_transforms(cfg),
            metadata_dir=metadata_dir,
            class_mapper=class_mapper,
            max_samples=max_samples,
        )
        val_ds = TarpDataset(
            split_csv=data_cfg["val_csv"],
            task=task,
            transform=get_val_transforms(cfg),
            metadata_dir=metadata_dir,
            class_mapper=class_mapper,
            max_samples=max_samples,
        )

        if use_fixed_size:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            train_sampler = ResolutionBatchSampler(train_ds, batch_size=batch_size, seed=seed)
            val_sampler   = ResolutionBatchSampler(val_ds,   batch_size=batch_size, drop_last=False)
            train_loader  = DataLoader(train_ds, batch_sampler=train_sampler,
                                    num_workers=num_workers, pin_memory=pin_memory)
            val_loader    = DataLoader(val_ds,   batch_sampler=val_sampler,
                                    num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader


    def build_optimizer(self, model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"].get("weight_decay", 1e-5),
        )


    def build_scheduler(self, optimizer, cfg: dict):
        name = cfg["training"].get("scheduler", "cosine")
        epochs = cfg["training"]["epochs"]
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
        return None


    def mlflow_setup(self, cfg: dict):
        logging.info("Setting up the MLflow logger...")
        try:
            mlflow_cfg = cfg.get("experiment", {})
            aws_profile = mlflow_cfg.get("aws_profile")

            username, password, url = Secrets(aws_profile=aws_profile).get_mlflow_user(
                env=self.args.env
            )
            experiment_name = MLFlowNamer.define_experiment_name(
                mlflow_cfg.get("mlflow_project_name"),
                "segmentation",
                mlflow_cfg.get("segmentation_mode", "binary"),
                mlflow_cfg.get("mlflow_subproject_name", "tarps"),
            )
            setup_mlflow(username, password, url, experiment_name)
        except Exception as error:
            logging.error(
                f"Some error arose while setting up MLflow logger: {error}")


    # ── Mixed-precision helpers ───────────────────────────────────────────────────

    @staticmethod
    def _build_detailed_metrics(cfg: dict) -> DetailedSegmentationMetrics:
        """Build a DetailedSegmentationMetrics instance from the YAML config.

        ``num_classes`` for DetailedSegmentationMetrics counts **all** classes
        including background (class 0):
        - binary task  → 2  (background + 1 foreground)
        - multiclass   → cfg[model][num_classes]  (already includes background)
        """
        task              = cfg["data"].get("task", "binary")
        model_num_classes = cfg["model"].get("num_classes", 1)
        dm_num_classes    = 2 if task == "binary" else model_num_classes
        threshold         = cfg["inference"].get("threshold", 0.5)
        # Optional: {class_id: name} mapping from config, e.g. {1: "tarp"}
        class_names_raw   = cfg["data"].get("class_names")
        class_names       = (
            {int(k): v for k, v in class_names_raw.items()}
            if class_names_raw else None
        )
        return DetailedSegmentationMetrics(
            num_classes = dm_num_classes,
            threshold   = threshold,
            class_names = class_names,
            pixel_only  = True,   # instance metrics only on test set
        )


    # ── Mixed-precision helpers ───────────────────────────────────────────────────

    @staticmethod
    def _make_autocast(device: torch.device, enabled: bool):
        """Return an autocast context manager, or a no-op if AMP is disabled.

        * CUDA  → torch.amp.autocast('cuda', float16)   – full AMP
        * MPS   → torch.amp.autocast('mps',  float16)   – autocast only (no scaler)
        * CPU   → nullcontext                            – AMP not useful on CPU
        """
        if not enabled or device.type == "cpu":
            return nullcontext()
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)


    @staticmethod
    def _make_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler | None:
        """Return a GradScaler for CUDA AMP, or None otherwise.

        GradScaler is only meaningful on CUDA. MPS and CPU run without it.
        """
        if enabled and device.type == "cuda":
            return torch.amp.GradScaler("cuda")
        return None


    # ─────────────────────────────────────────────────────────────────────────────


    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        task: str,
        grad_clip: float,
        autocast_ctx,
        scaler: torch.amp.GradScaler | None,
        metrics: DetailedSegmentationMetrics,
    ) -> dict[str, float]:
        model.train()
        metrics.reset()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Train", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            with autocast_ctx:
                logits = model(images)
                loss   = self._compute_loss(criterion, logits, masks, task)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            metrics.update(logits.detach(), masks)

        result = metrics.compute()
        result["loss"] = total_loss / len(loader)
        return result


    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        task: str,
        autocast_ctx,
        metrics: DetailedSegmentationMetrics,
    ) -> dict[str, float]:
        model.eval()
        metrics.reset()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Val", leave=False):
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            with autocast_ctx:
                logits = model(images)
                total_loss += self._compute_loss(criterion, logits, masks, task).item()
            metrics.update(logits, masks)

        result = metrics.compute()
        result["loss"] = total_loss / len(loader)
        return result


    @staticmethod
    def _compute_loss(criterion, logits, masks, task):
        if task == "binary":
            return criterion(logits.squeeze(1), masks.float())
        return criterion(logits, masks)


    @staticmethod
    def _prefix(prefix: str, d: dict) -> dict:
        return {f"{prefix}/{k}": v for k, v in d.items()}


    @staticmethod
    def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = {}
        for k, v in d.items():
            key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(TrainSegmentationModel._flatten(v, key, sep))
            elif v is not None:
                items[key] = v
        return items


    def train(self, cfg: dict):
        self.set_seed(cfg["experiment"].get("seed", 42))
        self.device = resolve_device()

        # ── CUDA-specific backend optimisations ───────────────────────────────────
        if self.device.type == "cuda":
            # TF32 on Ampere (A100/A10/A30/…): faster matmul with negligible precision loss.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True

        # ── Mixed precision ───────────────────────────────────────────────────────
        use_amp      = cfg["training"].get("mixed_precision", False)
        autocast_ctx = self._make_autocast(self.device, use_amp)
        scaler       = self._make_scaler(self.device, use_amp)
        amp_label    = (
            "AMP enabled (autocast + GradScaler)" if scaler is not None
            else f"AMP enabled (autocast only, no scaler on {self.device.type})" if use_amp and self.device.type != "cpu"
            else "AMP disabled"
        )
        print(f"  Device: {self.device}  |  {amp_label}")

        jira_ticket_url = getattr(self, "jira_ticket_url", None)
        self.mlflow_setup(cfg)

        run_name = MLFlowNamer.define_run_name(
            [cfg["training"].get("loss", "dice_bce")],
            "AdamW",
            cfg["training"].get("learning_rate", 1e-4),
            cfg["model"].get("architecture", "unet"),
            cfg["model"].get("encoder", "resnet34"),
        )
        with mlflow.start_run(run_name=run_name, description=jira_ticket_url) as run:
            mlflow.log_params(self._flatten(cfg))

            train_loader, val_loader = self.build_dataloaders(cfg, self.device)
            model = build_model(cfg).to(self.device)

            # ── torch.compile (PyTorch 2.x, CUDA only) ────────────────────────────
            if cfg["training"].get("compile", False):
                if hasattr(torch, "compile") and self.device.type == "cuda":
                    model = torch.compile(model)
                    print("  torch.compile() applied — first batch will be slower (tracing)")
                else:
                    print("  torch.compile skipped (requires PyTorch ≥ 2.0 and CUDA)")
            criterion = get_loss(cfg, self.device)
            optimizer = self.build_optimizer(model, cfg)
            scheduler = self.build_scheduler(optimizer, cfg)

            ckpt_dir = Path("checkpoints") / cfg["experiment"]["name"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            best_iou        = 0.0
            patience        = cfg["training"].get("early_stopping_patience", 15)
            patience_counter = 0
            grad_clip       = cfg["training"].get("grad_clip", 1.0)
            ckpt_path       = ckpt_dir / "best.pt"

            # Build the shared metrics object (reset internally each epoch)
            dm_metrics = self._build_detailed_metrics(cfg)

            for epoch in range(cfg["training"]["epochs"]):
                task = cfg["data"].get("task", "binary")
                train_metrics = self.train_one_epoch(
                    model, train_loader, criterion, optimizer,
                    self.device, task, grad_clip, autocast_ctx, scaler,
                    metrics=dm_metrics,
                )
                val_metrics = self.validate(
                    model, val_loader, criterion, self.device, task, autocast_ctx,
                    metrics=dm_metrics,
                )

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["pixel/iou_micro"])
                elif scheduler is not None:
                    scheduler.step()

                mlflow.log_metrics(
                    {**self._prefix("train", train_metrics), **self._prefix("val", val_metrics)},
                    step=epoch,
                )
                mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

                print(
                    f"Epoch {epoch+1:3d} | "
                    f"train_loss={train_metrics['loss']:.4f}  train_iou={train_metrics['pixel/iou_micro']:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f}  val_iou={val_metrics['pixel/iou_micro']:.4f}"
                )

                if val_metrics["pixel/iou_micro"] > best_iou:
                    best_iou = val_metrics["pixel/iou_micro"]
                    patience_counter = 0
                    torch.save(model, f"{run.info.run_id}.pth")
                    print("Best model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # ──────── Test evaluation with the best checkpoint ────────────
            print("\n=== Running test evaluation on best checkpoint ===")
            torch.cuda.empty_cache()
            best_model = torch.load(
                f"{run.info.run_id}.pth", weights_only=False)
            test_metrics = evaluate(cfg, best_model)
            mlflow.log_metrics(self._prefix("test", test_metrics))


    def run(self) -> bool:
        logging.info(f"Running {self.app_name} app...")
        try:
            # Load config from --config argument
            config_path = self.args.config
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            if self.__initialize_app(cfg):
                init_time = time.time()
                exit_ok = self.train(cfg)
                finish_time = time.time()
                logging.info(
                    f"Total elapsed time {timedelta(seconds=finish_time - init_time)}"
                )
            else:
                logging.info(f"Initializing {self.app_name}... failed")
                exit_ok = False
        except Exception as e:
            logging.critical(f"Uncaught exception: {str(e)}")
            logging.critical(traceback.format_exc())
            exit_ok = False
        finally:
            logging.info(f"{self.app_name} app finished")
        return exit_ok


    def __initialize_app(self, cfg: dict):
        logging.info(f"Initializing {self.app_name}...")
        try:
            exp_cfg = cfg.get("experiment", {})
            self.set_seed(exp_cfg.get("seed", 42))
            jira_ticket = exp_cfg.get("jira_ticket")
            self.task   = cfg["data"].get("task", "binary")
            self.is_binary_segmentation = cfg["model"].get("num_classes", 1) == 1
            if jira_ticket:
                self.jira_ticket_url = MLFlowNamer.define_jira_ticket_url(
                    jira_ticket
                )


        except Exception as e:
            logging.exception(f"Exception raised initializing app", e)
            return False

        return True

    def cleanup(self) -> bool:
        return True


def main():
    app = TrainSegmentationModel()
    if app.start():
        sys.exit(APP_BASE_EXIT_OK)

    sys.exit(APP_BASE_EXIT_ERROR)


if __name__ == "__main__":
    main()