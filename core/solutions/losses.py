import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxEntropyOODLoss(nn.Module):
    """Push OOD pixels toward maximum softmax uncertainty."""
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, ood_mask: torch.Tensor) -> torch.Tensor:
        """Compute negative entropy over pasted OOD pixels."""
        if not ood_mask.any():
            return logits.sum() * 0.0

        ood_logits = logits.permute(0, 2, 3, 1)[ood_mask]

        probs = F.softmax(ood_logits, dim=-1)
        log_probs = F.log_softmax(ood_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        loss_ood = -torch.mean(entropy) / math.log(logits.size(1))
        return loss_ood


class LogitNormOODLoss(nn.Module):
    """Penalize large logit magnitudes on OOD pixels."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, ood_mask: torch.Tensor) -> torch.Tensor:
        """Penalize high logit norms on OOD pixels."""
        if not ood_mask.any():
            return logits.sum() * 0.0
            
        ood_logits = logits.permute(0, 2, 3, 1)[ood_mask]
        
        l2_norms = torch.norm(ood_logits / self.temperature, p=2, dim=-1)
        
        return torch.mean(l2_norms) / math.sqrt(logits.size(1))


class EntropyLogitNormOODLoss(nn.Module):
    """Combine softmax uncertainty with low-energy/logit-magnitude pressure."""
    def __init__(self, entropy_weight: float = 1.0, logit_norm_weight: float = 0.05):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.logit_norm_weight = logit_norm_weight
        self.entropy_loss = MaxEntropyOODLoss()
        self.logit_norm_loss = LogitNormOODLoss()

    def forward(self, logits: torch.Tensor, ood_mask: torch.Tensor) -> torch.Tensor:
        if not ood_mask.any():
            return logits.sum() * 0.0

        return (
            self.entropy_weight * self.entropy_loss(logits, ood_mask)
            + self.logit_norm_weight * self.logit_norm_loss(logits, ood_mask)
        )


class CombinedFineTuningLoss(nn.Module):
    """Combine semantic CE on clean pixels with OOD loss on pasted pixels."""
    def __init__(
        self,
        ood_loss_type: str = "max_entropy",
        ignore_index: int = 255,
        ood_entropy_weight: float = 1.0,
        ood_logit_norm_weight: float = 0.05,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        if ood_loss_type == "max_entropy":
            self.ood_loss_fn = MaxEntropyOODLoss()
        elif ood_loss_type in {"entropy_logit_norm", "combined"}:
            self.ood_loss_fn = EntropyLogitNormOODLoss(
                entropy_weight=ood_entropy_weight,
                logit_norm_weight=ood_logit_norm_weight,
            )
        elif ood_loss_type in {"logit_norm", "energy"}:
            self.ood_loss_fn = LogitNormOODLoss()
        else:
            raise ValueError(f"Unsupported OOD loss type: {ood_loss_type}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, ood_mask: torch.Tensor) -> dict:
        """Return total loss plus CE and OOD components for logging."""
        training_labels = labels.clone()
        training_labels[ood_mask] = self.ce_loss.ignore_index
        
        n_classes = logits.size(1)
        valid_mask = (training_labels >= 0) & (training_labels < n_classes)
        invalid_mask = (~valid_mask) & (training_labels != self.ce_loss.ignore_index)
        if invalid_mask.any():
            warnings.warn(
                f"Found invalid labels; mapping to ignore_index. "
                f"Min: {training_labels[invalid_mask].min().item()}, "
                f"Max: {training_labels[invalid_mask].max().item()}",
                RuntimeWarning,
                stacklevel=2,
            )
            # Prevent device-side asserts in NLLLoss.
            training_labels[invalid_mask] = self.ce_loss.ignore_index
            
        if (training_labels != self.ce_loss.ignore_index).any():
            loss_ce = self.ce_loss(logits, training_labels)
        else:
            loss_ce = logits.sum() * 0.0
        
        loss_ood = self.ood_loss_fn(logits, ood_mask)
        
        return {
            "loss_ce": loss_ce,
            "loss_ood": loss_ood,
            "total_loss": loss_ce + loss_ood
        }
