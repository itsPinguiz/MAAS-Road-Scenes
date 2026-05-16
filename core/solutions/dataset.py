import os
import random
import glob
import warnings
from typing import Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from core.solutions.augmentation import PerspectiveOutlierPasting

class OutlierAugmentedDataset(Dataset):
    """Wrap a semantic dataset and paste random OOD cutouts during loading."""
    def __init__(
        self, 
        base_dataset: Dataset, 
        outliers_dir: str, 
        ignore_index: int = 19,
        transform: PerspectiveOutlierPasting = None
    ):
        """
        Args:
            base_dataset: PyTorch dataset returning ``(image, semantic_mask)``.
            outliers_dir: Directory with RGBA PNG cutouts.
            ignore_index: Label assigned to pasted anomaly pixels.
            transform: Optional pasting transform.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.ignore_index = ignore_index
        self.transform_pipeline = transform if transform is not None else PerspectiveOutlierPasting()
        
        self.outlier_paths = sorted(glob.glob(os.path.join(outliers_dir, "*.png")))
        if len(self.outlier_paths) == 0:
            warnings.warn(f"No outlier PNG files found in {outliers_dir}", RuntimeWarning, stacklevel=2)

        self.pil_to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _load_random_outlier(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load one RGBA cutout and split it into RGB tensor and alpha mask."""
        if not self.outlier_paths:
            return torch.zeros(3, 100, 100), torch.zeros(100, 100)

        path = random.choice(self.outlier_paths)
        with Image.open(path) as img:
            img = img.convert("RGBA")
            
            rgb_img = img.convert("RGB")
            rgb_tensor = self.pil_to_tensor(rgb_img)
            
            alpha_channel = img.getchannel('A')
            alpha_tensor = self.pil_to_tensor(alpha_channel).squeeze(0)
            
            return rgb_tensor, alpha_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return image, semantic mask, and binary OOD mask for one sample."""
        data_tuple = self.base_dataset[idx]
        img, base_mask = data_tuple[0], data_tuple[1]
        
        if isinstance(img, Image.Image):
            img = self.pil_to_tensor(img)
        elif torch.is_tensor(img):
            if torch.is_floating_point(img):
                img = img.float()
                if img.numel() > 0 and img.max() > 1.0:
                    img = img / 255.0
            else:
                img = img.float() / 255.0
            
        if isinstance(base_mask, Image.Image):
            import numpy as np
            base_mask = torch.from_numpy(np.array(base_mask)).long()
        elif torch.is_tensor(base_mask):
            base_mask = base_mask.long()

        if not self.outlier_paths:
            return img, base_mask, torch.zeros_like(base_mask, dtype=torch.bool)
            
        outlier_img, outlier_mask = self._load_random_outlier()
        
        aug_image, aug_mask, ood_bin_mask = self.transform_pipeline(
            img, 
            base_mask, 
            outlier_img, 
            outlier_mask
        )
        
        if ood_bin_mask.any():
            aug_mask[ood_bin_mask] = self.ignore_index

        return aug_image, aug_mask, ood_bin_mask
