import random
import torch
import numpy as np
import cv2
from core.utility.config_loader import cfg
from typing import Tuple, Optional

class PerspectiveOutlierPasting:
    """
    Augmentation Pipeline basata su "Outlier Pasting". 
    Risolve tre vulnerabilità critiche:
    1. Esposizione Multi-Scala Universale (Scale Gap)
    2. Integrazione di Profondità e Prospettiva (Vertical Blindness)
    3. Stress-Test sui Confini (Boundary Uncertainty)
    """
    def __init__(self):
        # Carica configurazioni dal config_loader
        self.config = cfg.solutions.augmentation
        self.p_apply = self.config.p_apply
        self.p_small = self.config.p_small
        self.small_range = self.config.small_scale_range
        self.large_range = self.config.large_scale_range
        self.perspective_strength = self.config.perspective_strength
        self.boundary_rate = self.config.boundary_rate
        self.boundary_classes = self.config.boundary_classes
        self.alpha = self.config.alpha

    def get_bimodal_scale(self) -> float:
        """
        Task 1: Esposizione Multi-Scala Universale
        Campiona un fattore di scala target da una distribuzione bimodale estrema.
        """
        if random.random() < self.p_small:
            # Anomalie minuscole (micro-dettagli)
            return random.uniform(self.small_range[0], self.small_range[1])
        else:
            # Anomalie massicce
            return random.uniform(self.large_range[0], self.large_range[1])

    def apply_perspective(self, base_scale: float, y_center: float, img_height: int) -> float:
        """
        Task 2: Integrazione di Profondità e Prospettiva
        Ridimensiona l'anomalia basandosi sull'altezza nell'immagine (coordinata Y).
        Un Y basso (vicino all'orizzonte, tipicamente ~H/2 in Cityscapes) rimpicciolisce l'oggetto.
        Un Y alto (vicino al cofano, H) ingrandisce l'oggetto.
        """
        # Normalizziamo Y da 0 a 1. Assumiamo che l'orizzonte stradale sia a metà immagine (0.5).
        # Per evitare scaling negativi:
        normalized_y = max(0.0, y_center / img_height) 
        
        # Formula di prospettiva lineare: scale_factor decresce in altezza (più è alto l'oggetto fisicamente, più "Y" è piccolo)
        # Se Y è piccolo, l'oggetto è lontano -> penalizziamo la scala.
        # perspective_strength regola quanto la prospettiva distorce. 
        # (Es. strength = 1.0 -> l'oggetto a y=0 è microscopico, a y=H è alla scala massima).
        perspective_factor = (normalized_y ** self.perspective_strength)
        
        # Evita scale zero o invalide mettendo un clip minimo al 10% della scala determinata.
        final_scale = base_scale * max(0.1, perspective_factor)
        return final_scale

    def find_boundary_coordinates(self, gt_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Task 3: Stress-Test sui Confini ("Context-Aware" Pasting)
        Troviamo i confini tra classi critiche (pali, muri, vegetazione) e strada/cielo.
        """
        # Creiamo un array booleano per i pixel che appartengono alle "boundary_classes"
        critical_pixels = np.isin(gt_mask, self.boundary_classes).astype(np.uint8)
        
        if np.sum(critical_pixels) == 0:
            return None # Nessuna classe critica nel frame, ritorno None
        
        # Usiamo operazioni morfologiche per trovare i "bordi" delle classi critiche
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(critical_pixels, kernel, iterations=1)
        eroded = cv2.erode(critical_pixels, kernel, iterations=1)
        
        edges = dilated - eroded
        edge_coords = np.argwhere(edges > 0)
        
        if len(edge_coords) > 0:
            # Scegliamo casualmente un pixel di confine
            idx = random.randint(0, len(edge_coords) - 1)
            y, x = edge_coords[idx]
            return int(x), int(y)
        
        return None

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, outlier_img: torch.Tensor, outlier_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applica l'Outlier Pasting.
        Atteso:
            image: [C, H, W] float Tensor, in range [0, 1] o [-1, 1]
            mask: [H, W] long Tensor (Ground truth task di segmentazione Cityscapes)
            outlier_img: [C, H_o, W_o] float Tensor (Immagine ostacolo ritagliata)
            outlier_mask: [H_o, W_o] uint8/float Tensor (Maschera esatta dell'ostacolo: 1 oggetto, 0 sfondo)
        Ritorna:
            aug_image: L'immagine aumentata [C, H, W]
            aug_mask: La nuova maschera OOD unita (la classe OOD riceverà un label dedicato o verrà sfruttato in fase di loss) [H, W]
            ood_binary_mask: Maschera binaria dei pixel outlier incollati -> 1 se incollato, 0 altrove. [H, W]
        """
        if random.random() > self.p_apply:
            # Usa shape interi per la maschera binaria di offset (tensore di zero)
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)
            
        C, H, W = image.shape
        _, H_o, W_o = outlier_img.shape
        
        # 1. Trova le coordinate (Context-Aware / Rand)
        use_boundary = (random.random() < self.boundary_rate)
        center_coords = None
        
        if use_boundary:
            # Mask proviene dal dataloader worker (CPU), nessuna sinc. GPU richiesta
            gt_numpy = mask.numpy()
            center_coords = self.find_boundary_coordinates(gt_numpy)
            
        if center_coords is None:
            # Fallback a coordinate casuali nella metà inferiore dell'immagine
            # assumendo che la strada sia sotto (y > H/3).
            cx = random.randint(int(W*0.1), int(W*0.9))
            cy = random.randint(int(H*0.3), int(H*0.95))
            center_coords = (cx, cy)
            
        cx, cy = center_coords
        
        # 2. Multi-Scale & Perspective Aware Size
        base_area_ratio = self.get_bimodal_scale()
        # Vogliamo che target_area / (H*W) = base_area_ratio * prospettiva
        # Applico il modificatore prospettico:
        final_area_ratio = self.apply_perspective(base_area_ratio, cy, H)
        
        target_area = final_area_ratio * (H * W)
        
        # Manteniamo l'aspect ratio dell'outlier: W_o / H_o = aspect_ratio
        aspect_ratio = W_o / float(H_o + 1e-6)
        
        # target_H * target_W = target_area  => target_H * (target_H * aspect_ratio) = target_area
        target_H = max(5, int(np.sqrt(target_area / aspect_ratio)))
        target_W = max(5, int(target_H * aspect_ratio))
        
        # Ridimensiona l'outlier (Interpolazione Bilineare per Immagine, Nearest per Maschera)
        outlier_img_res = torch.nn.functional.interpolate(outlier_img.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
        outlier_mask_res = torch.nn.functional.interpolate(outlier_mask.unsqueeze(0).unsqueeze(0).float(), size=(target_H, target_W), mode='nearest').squeeze(0).squeeze(0)
        
        # Punti top-left per il pasting
        x0 = cx - target_W // 2
        y0 = cy - target_H // 2
        
        # Se esce dai bordi, facciamo clip
        x_min, x_max = max(0, x0), min(W, x0 + target_W)
        y_min, y_max = max(0, y0), min(H, y0 + target_H)
        
        # Dimensioni relative del patch utile
        ox_min = x_min - x0
        ox_max = target_W - (x0 + target_W - x_max)
        oy_min = y_min - y0
        oy_max = target_H - (y0 + target_H - y_max)
        
        if (x_max <= x_min) or (y_max <= y_min):
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)
        
        # Estrai patch da fondere
        valid_outlier_img = outlier_img_res[:, oy_min:oy_max, ox_min:ox_max]
        valid_outlier_mask = outlier_mask_res[oy_min:oy_max, ox_min:ox_max]
        
        valid_outlier_mask_bool = (valid_outlier_mask > 0.5)

        # Copie
        aug_image = image.clone()
        aug_mask = mask.clone()
        ood_binary_mask = torch.zeros_like(mask, dtype=torch.bool)
        
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        
        # 1. Color Matching/Jitter (range [0, 1])
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        valid_outlier_img = jitter(valid_outlier_img)
        
        # FIX: Normalize outlier to match base Cityscapes ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=valid_outlier_img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=valid_outlier_img.device).view(3, 1, 1)
        valid_outlier_img = (valid_outlier_img - mean) / std
        
        # 2. Gaussian Noise su patch rimosso: applicheremo noise globale alla fine
        
        # 3. Edge Blurring (Feathering)
        k_size = max(3, int(min(target_H, target_W) * 0.08))
        if k_size % 2 == 0:
            k_size += 1
            
        feathered_mask = TF.gaussian_blur(
            valid_outlier_mask.unsqueeze(0), 
            kernel_size=[k_size, k_size], 
            sigma=[max(1.0, k_size / 3.0), max(1.0, k_size / 3.0)]
        ).squeeze(0)
        
        alpha_mask = feathered_mask * self.alpha
        
        # Blend (Pasting forte controllato da alpha sfocata per feathering)
        # Effettuiamo blending solo sui pixel interni (evitando dark halos)
        for c in range(C):
            patch = aug_image[c, y_min:y_max, x_min:x_max]
            patch[valid_outlier_mask_bool] = (
                (1.0 - alpha_mask[valid_outlier_mask_bool]) * patch[valid_outlier_mask_bool] +
                alpha_mask[valid_outlier_mask_bool] * valid_outlier_img[c, valid_outlier_mask_bool]
            )
            aug_image[c, y_min:y_max, x_min:x_max] = patch
            
        # OOD Mask
        patch_ood = ood_binary_mask[y_min:y_max, x_min:x_max]
        patch_ood[valid_outlier_mask_bool] = True
        ood_binary_mask[y_min:y_max, x_min:x_max] = patch_ood
        
        # 2. Global Gaussian Noise (to mask compression differences without creating a local beacon)
        global_noise = torch.randn_like(aug_image) * 0.05
        aug_image = aug_image + global_noise
        
        # Volendo si mappa 'aug_mask' impostando un indice OOD riservato temporaneamente
        # ma spesso è meglio usare `ood_binary_mask` durante il training per differenziare In-Dist vs Out-Dist.
        
        return aug_image, aug_mask, ood_binary_mask
