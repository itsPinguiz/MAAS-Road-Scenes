import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxEntropyOODLoss(nn.Module):
    """
    Ricalibrazione dei Logit - Task 4: Assorbimento Tassonomico.
    Spinge i pixel identificati come anomali (outlier) verso la massima incertezza 
    (distribuzione uniforme). Minimizzare il negativo dell'entropia equivale a massimizzarla.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, ood_mask: torch.Tensor) -> torch.Tensor:
        """
        Calcola la perdita di entropia sui pixel outlier.
        Args:
            logits: Previsioni del modello [B, C, H, W]
            ood_mask: Maschera binaria dove True/1 indica i pixel dell'anomalia incollata [B, H, W]
        """
        # Se non ci sono pixel OOD nel batch, la loss è 0
        if not ood_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Seleziona solo i logits appartenenti ai pixel out-of-distribution
        # ood_logits shape: [N_ood_pixels, C]
        ood_logits = logits.permute(0, 2, 3, 1)[ood_mask]

        # Calcolo delle probabilità via Softmax
        probs = F.softmax(ood_logits, dim=-1)
        
        # Log Softmax per stabilità numerica (evita log(0))
        log_probs = F.log_softmax(ood_logits, dim=-1)
        
        # Entropia = - sum(P * log(P))
        # Vogliamo massimizzare l'entropia, quindi minimizziamo la Negative Entropy (-H)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Minimizziamo la media dell'entropia negata sui pixel outlier
        loss_ood = -torch.mean(entropy) 
        return loss_ood


class LogitNormOODLoss(nn.Module):
    """
    Alternativa Energy/Logit Normaizzazione per la Ricalibrazione dei Logit.
    Penalizza i vettori di logit con magnitudine troppo elevata sui pixel OOD.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, ood_mask: torch.Tensor) -> torch.Tensor:
        if not ood_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        ood_logits = logits.permute(0, 2, 3, 1)[ood_mask]
        
        # Normalizzazione: spinge la norma L2 verso zero, o calcola cross entropy 
        # verso una distribuzione uniforme. In questo caso spingiamo la norma L2 dei logit a 0, 
        # che costringe softmax(logits) verso l'uniforme.
        l2_norms = torch.norm(ood_logits, p=2, dim=-1)
        
        return torch.mean(l2_norms)


class CombinedFineTuningLoss(nn.Module):
    """
    Modulo completo che unisce:
    - CrossEntropyLoss standard per il task in-distribution sui pixel puliti.
    - OODLoss (es. MaxEntropyOODLoss) sui pixel patchati con Outlier Pasting.
    """
    def __init__(self, ood_loss_type: str = "max_entropy", ignore_index: int = 255):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        if ood_loss_type == "max_entropy":
            self.ood_loss_fn = MaxEntropyOODLoss()
        elif ood_loss_type == "logit_norm":
            self.ood_loss_fn = LogitNormOODLoss()
        else:
            raise ValueError(f"OOD loss type non supportato: {ood_loss_type}")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, ood_mask: torch.Tensor) -> dict:
        """
        Ritorna un dizionario con la perdita totale e i singoli componenti per il logging.
        """
        # La Cross Entropy standard viene applicata su tutta l'immagine tranne i pixel
        # in cui abbiamo incollato deliberatamente l'anomalia (li impostiamo a ignore_index)
        
        # Creiamo un tensore etichette sicuro per non sporcare l'originale
        training_labels = labels.clone()
        training_labels[ood_mask] = self.ce_loss.ignore_index
        
        # FIX: Safeguard against CUDA device-side assert (t >= 0 && t < n_classes failed)
        n_classes = logits.size(1)
        valid_mask = (training_labels >= 0) & (training_labels < n_classes)
        invalid_mask = (~valid_mask) & (training_labels != self.ce_loss.ignore_index)
        if invalid_mask.any():
            # Trovato un target fuori range! Stampiamo a terminale e clippiamo
            print(f"[ERROR] Found invalid labels! Min: {training_labels[invalid_mask].min().item()}, Max: {training_labels[invalid_mask].max().item()}")
            # Force invalid pixels to ignore_index to prevent NLLLoss kernel crash
            training_labels[invalid_mask] = self.ce_loss.ignore_index
            
        loss_ce = self.ce_loss(logits, training_labels)
        
        # Calcoliamo la loss OOD esclusivamente sui pixel incollati
        loss_ood = self.ood_loss_fn(logits, ood_mask)
        
        return {
            "loss_ce": loss_ce,
            "loss_ood": loss_ood,
            "total_loss": loss_ce + loss_ood  # Si possono aggiungere pesi tramite cfg
        }
