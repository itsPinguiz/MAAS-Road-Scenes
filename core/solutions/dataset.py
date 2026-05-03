import os
import random
import glob
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from core.solutions.augmentation import PerspectiveOutlierPasting

class OutlierAugmentedDataset(Dataset):
    """
    Dataset Wrapper per l'addestramento OOD.
    Avvolge un dataset standard (es. Cityscapes) e inietta dinamicamente gli estrattori di feature
    (anomalie/oggetti croppati) sfruttando l'Outlier Pasting ad ogni __getitem__,
    eseguendo le operazioni su CPU in multithreading prima del caricamento in GPU.
    """
    def __init__(
        self, 
        base_dataset: Dataset, 
        outliers_dir: str, 
        ignore_index: int = 19,  # deve corrispondere a ignore_index in CombinedFineTuningLoss
        transform: PerspectiveOutlierPasting = None
    ):
        """
        Args:
            base_dataset: Un dataset PyTorch che restituisce (image, semantic_mask)
            outliers_dir: Percorso alla directory con file .png (RGBA) con oggetti cropped
            ignore_index: Il valore che la semantic_mask assumerà sui pixel dell'anomalia incollata
            transform: Istanza di PerspectiveOutlierPasting (creata in automatico se None)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.ignore_index = ignore_index
        self.transform_pipeline = transform if transform is not None else PerspectiveOutlierPasting()
        
        # Scansiona folder degli outlier per immagini valide (es. PNG con canale Alpha)
        self.outlier_paths = glob.glob(os.path.join(outliers_dir, "*.png"))
        if len(self.outlier_paths) == 0:
             print(f"ATTENZIONE: Nessun outlier .png trovato in {outliers_dir}")

        # Utility transforms da PIL a Tensor se base_dataset non lo fa già.
        self.pil_to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_random_outlier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carica un random outlier dal pool e separa i canali RGB(immagine) dalla Maschera (canale Alpha).
        """
        if not self.outlier_paths:
            # Fallback tensor vuoto
            return torch.zeros(3, 100, 100), torch.zeros(100, 100)

        path = random.choice(self.outlier_paths)
        with Image.open(path) as img:
            img = img.convert("RGBA")
            
            # Estraiamo l'RGB
            rgb_img = img.convert("RGB")
            # Convertiamo l'RGB in tensor (Tensor di Float [3, H, W] in range [0, 1])
            rgb_tensor = self.pil_to_tensor(rgb_img)
            
            # Estraiamo il canale Alpha e lo convertiamo in tensor (Mask [H, W])
            alpha_channel = img.getchannel('A')
            # pil_to_tensor restituisce [1, H, W] da [0, 1], noi vogliamo solo [H, W] float
            alpha_tensor = self.pil_to_tensor(alpha_channel).squeeze(0)
            
            return rgb_tensor, alpha_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Intercetta l'item dal dataset base, lo trasforma aggiungendo l'anomalia
        (Outlier Pasting) e sovrascrive l'etichetta.
        """
        # 1. Carica il sample Base Cityscapes
        # Nota: assumiamo che base_dataset restituisca già tensori: 
        # img -> FloatTensor [3, H, W], mask -> LongTensor [H, W]
        data_tuple = self.base_dataset[idx]
        img, base_mask = data_tuple[0], data_tuple[1]
        
        # Gestione fallback per chi ritorna tuple PIL da dataset base o dimensioni diverse
        if isinstance(img, Image.Image):
            img = self.pil_to_tensor(img)
            
        if isinstance(base_mask, Image.Image):
            import numpy as np
            base_mask = torch.from_numpy(np.array(base_mask)).long()
            
        # 2. Ottieni un outlier
        outlier_img, outlier_mask = self._load_random_outlier()
        
        # 3. Esegui la Pipeline di Data Augmentation su CPU 
        # (questo calcolerà prospettive, boundary checks su numpy/CPU memory e paste)
        aug_image, aug_mask, ood_bin_mask = self.transform_pipeline(
            img, 
            base_mask, 
            outlier_img, 
            outlier_mask
        )
        
        # 4. Modifichiamo il target originale: tutti i pixel sovrascritti dall'outlier 
        # devono riflettere la nuova classe anomala (es "254" o ignore_index)
        # per consentire una facile discriminazione post-forward.
        if ood_bin_mask.any():
            aug_mask[ood_bin_mask] = self.ignore_index

        # Restituiamo il tridente: 
        # - L'immagine pronta da dare al modello, 
        # - La label semantica pronta per la CrossEntropy (modificata)
        # - La mapping OOD pronta per la MaxEntropyOODLoss
        return aug_image, aug_mask, ood_bin_mask
