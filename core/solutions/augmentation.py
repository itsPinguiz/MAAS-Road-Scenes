import random
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from core.utility.config_loader import cfg
from typing import Tuple, Optional

class PerspectiveOutlierPasting:
    """Paste COCO cutouts into road scenes with scale and boundary awareness."""
    def __init__(self):
        self.config = cfg.solutions.augmentation
        self.p_apply = self.config.p_apply
        self.p_small = self.config.p_small
        self.small_range = self.config.small_scale_range
        self.large_range = self.config.large_scale_range
        self.perspective_strength = self.config.perspective_strength
        self.boundary_rate = self.config.boundary_rate
        self.boundary_classes = self.config.boundary_classes
        self.road_rate = getattr(self.config, 'road_rate', 0.0)
        self.road_class_id = getattr(self.config, 'road_class_id', 0)
        self.alpha = self.config.alpha
        self.edge_feathering = getattr(self.config, 'edge_feathering', False)
        self.noise_std = getattr(self.config, 'noise_std', 0.05)
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def get_bimodal_scale(self) -> float:
        """Sample a target area ratio from the configured bimodal scale ranges."""
        if random.random() < self.p_small:
            return random.uniform(self.small_range[0], self.small_range[1])
        return random.uniform(self.large_range[0], self.large_range[1])

    def apply_perspective(self, base_scale: float, y_center: float, img_height: int) -> float:
        """Shrink far-away paste locations using the vertical image position."""
        normalized_y = max(0.0, y_center / img_height) 
        perspective_factor = (normalized_y ** self.perspective_strength)
        final_scale = base_scale * max(0.1, perspective_factor)
        return final_scale

    def find_boundary_coordinates(self, gt_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Pick a random boundary pixel from configured critical classes."""
        critical_pixels = np.isin(gt_mask, self.boundary_classes).astype(np.uint8)
        
        if not critical_pixels.any():
            return None
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(critical_pixels, kernel, iterations=1)
        eroded = cv2.erode(critical_pixels, kernel, iterations=1)
        
        edges = dilated - eroded
        edge_coords = np.argwhere(edges > 0)
        
        if len(edge_coords) > 0:
            idx = random.randint(0, len(edge_coords) - 1)
            y, x = edge_coords[idx]
            return int(x), int(y)
        
        return None

    def find_class_coordinates(self, gt_mask: np.ndarray, class_id: int) -> Optional[Tuple[int, int]]:
        """Pick a random pixel from one semantic class, preferring lower-image road regions."""
        class_pixels = np.argwhere(gt_mask == class_id)
        if len(class_pixels) == 0:
            return None

        height = gt_mask.shape[0]
        lower_pixels = class_pixels[class_pixels[:, 0] >= int(height * 0.35)]
        candidates = lower_pixels if len(lower_pixels) > 0 else class_pixels
        idx = random.randint(0, len(candidates) - 1)
        y, x = candidates[idx]
        return int(x), int(y)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor, outlier_img: torch.Tensor, outlier_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply outlier pasting.

        Returns the augmented image, semantic mask, and a binary mask of pasted pixels.
        """
        if random.random() > self.p_apply:
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)
        if outlier_mask.numel() == 0 or not (outlier_mask > 0.5).any():
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)
            
        C, H, W = image.shape
        _, H_o, W_o = outlier_img.shape
        
        gt_numpy = mask.detach().cpu().numpy()
        use_boundary = (random.random() < self.boundary_rate)
        center_coords = None
        
        if use_boundary:
            center_coords = self.find_boundary_coordinates(gt_numpy)

        if center_coords is None and random.random() < self.road_rate:
            center_coords = self.find_class_coordinates(gt_numpy, self.road_class_id)
            
        if center_coords is None:
            # Road pixels are usually in the lower half of Cityscapes-like frames.
            cx = random.randint(int(W*0.1), int(W*0.9))
            cy = random.randint(int(H*0.3), int(H*0.95))
            center_coords = (cx, cy)
            
        cx, cy = center_coords
        
        base_area_ratio = self.get_bimodal_scale()
        final_area_ratio = self.apply_perspective(base_area_ratio, cy, H)
        
        target_area = final_area_ratio * (H * W)
        aspect_ratio = W_o / float(H_o + 1e-6)
        target_H = max(5, int(np.sqrt(target_area / aspect_ratio)))
        target_W = max(5, int(target_H * aspect_ratio))
        
        outlier_img_res = F.interpolate(outlier_img.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
        outlier_mask_res = F.interpolate(outlier_mask.unsqueeze(0).unsqueeze(0).float(), size=(target_H, target_W), mode='nearest').squeeze(0).squeeze(0)
        
        x0 = cx - target_W // 2
        y0 = cy - target_H // 2
        
        x_min, x_max = max(0, x0), min(W, x0 + target_W)
        y_min, y_max = max(0, y0), min(H, y0 + target_H)
        
        ox_min = x_min - x0
        ox_max = target_W - (x0 + target_W - x_max)
        oy_min = y_min - y0
        oy_max = target_H - (y0 + target_H - y_max)
        
        if (x_max <= x_min) or (y_max <= y_min):
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)
        
        valid_outlier_img = outlier_img_res[:, oy_min:oy_max, ox_min:ox_max]
        valid_outlier_mask = outlier_mask_res[oy_min:oy_max, ox_min:ox_max]
        
        valid_outlier_mask_bool = (valid_outlier_mask > 0.5)
        if not valid_outlier_mask_bool.any():
            return image, mask, torch.zeros_like(mask, dtype=torch.bool)

        aug_image = image.clone()
        aug_mask = mask.clone()
        ood_binary_mask = torch.zeros_like(mask, dtype=torch.bool)

        valid_outlier_img = self.jitter(valid_outlier_img)

        if self.edge_feathering:
            k_size = max(3, int(min(target_H, target_W) * 0.08))
            if k_size % 2 == 0:
                k_size += 1
            feathered_mask = TF.gaussian_blur(
                valid_outlier_mask.unsqueeze(0),
                kernel_size=[k_size, k_size],
                sigma=[max(1.0, k_size / 3.0), max(1.0, k_size / 3.0)]
            ).squeeze(0)
            alpha_mask = feathered_mask * self.alpha
        else:
            alpha_mask = valid_outlier_mask.float() * self.alpha
        alpha_mask = alpha_mask.clamp(0.0, 1.0)

        # Restrict blending to foreground pixels to avoid dark halos.
        patch = aug_image[:, y_min:y_max, x_min:x_max]
        alpha = alpha_mask.unsqueeze(0)
        mask_for_blend = valid_outlier_mask_bool.unsqueeze(0)
        blended_patch = (1.0 - alpha) * patch + alpha * valid_outlier_img
        aug_image[:, y_min:y_max, x_min:x_max] = torch.where(mask_for_blend, blended_patch, patch)
            
        patch_ood = ood_binary_mask[y_min:y_max, x_min:x_max]
        patch_ood[valid_outlier_mask_bool] = True
        ood_binary_mask[y_min:y_max, x_min:x_max] = patch_ood
        
        # Mask compression artifacts without creating a local noise cue.
        if self.noise_std > 0:
            global_noise = torch.randn_like(aug_image) * self.noise_std
            aug_image = (aug_image + global_noise).clamp(0.0, 1.0)

        return aug_image, aug_mask, ood_binary_mask
