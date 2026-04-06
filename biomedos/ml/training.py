"""Training loop utilities for graph models."""

from __future__ import annotations

from pathlib import Path

import torch
from pydantic import BaseModel

from biomedos.ml.dataset import EdgeSplit, GraphMLDataset
from biomedos.ml.models.base import BaseGNNModel


class TrainingConfig(BaseModel):
    """Configuration for graph model training."""

    epochs: int = 100
    learning_rate: float = 0.001
    patience: int = 10
    weight_decay: float = 1e-4
    min_delta: float = 1e-4
    checkpoint_path: str | None = None


class TrainingResult(BaseModel):
    """Summary of a training run."""

    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    checkpoint_path: str | None = None


class Trainer:
    """Unified training loop with early stopping."""

    def __init__(
        self,
        model: BaseGNNModel,
        *,
        device: str = "cpu",
    ) -> None:
        """Initialize the trainer."""

        self.model = model.to(device)
        self.device = device
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def fit(
        self,
        dataset: GraphMLDataset,
        config: TrainingConfig,
    ) -> TrainingResult:
        """Train a model using the provided configuration."""

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_state: dict[str, torch.Tensor] | None = None
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, config.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            train_loss = self._compute_loss(dataset, dataset.train_split)
            train_loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_split = (
                    dataset.val_split
                    if dataset.val_split.labels.numel() > 0
                    else dataset.train_split
                )
                val_loss = self._compute_loss(dataset, val_split)

            train_losses.append(float(train_loss.item()))
            val_losses.append(float(val_loss.item()))

            if float(val_loss.item()) + config.min_delta < best_val_loss:
                best_val_loss = float(val_loss.item())
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                if config.checkpoint_path is not None:
                    self._save_checkpoint(config.checkpoint_path, best_state)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_loss = best_val_loss if best_val_loss < float("inf") else val_losses[-1]
        return TrainingResult(
            best_epoch=best_epoch,
            best_val_loss=best_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            checkpoint_path=config.checkpoint_path,
        )

    def _compute_loss(self, dataset: GraphMLDataset, split: EdgeSplit) -> torch.Tensor:
        """Compute split loss for a batch-free small-graph setting."""

        features = dataset.global_x.to(self.device)
        edge_index = dataset.edge_index.to(self.device)
        graph_relation_index = dataset.relation_index.to(self.device)
        logits = self.model.score_pairs(
            features,
            edge_index,
            graph_relation_index,
            split.source_index.to(self.device),
            split.target_index.to(self.device),
            split.relation_index.to(self.device),
        )
        labels = split.labels.to(self.device)
        if labels.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return self.loss_fn(logits, labels)

    def _save_checkpoint(self, checkpoint_path: str, state_dict: dict[str, torch.Tensor]) -> None:
        """Persist a model checkpoint to disk."""

        target = Path(checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, target)
