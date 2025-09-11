# Auto-generated single-file for BatchFixedSizePad
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple

# ---- original imports from contributing modules ----
from mmdet.registry import MODELS
from mmengine.structures import PixelData

# ---- BatchFixedSizePad (target) ----
class BatchFixedSizePad(nn.Module):
    """Fixed size padding for batch images.

    Args:
        size (Tuple[int, int]): Fixed padding size. Expected padding
            shape (h, w). Defaults to None.
        img_pad_value (int): The padded pixel value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
    """

    def __init__(self,
                 size: Tuple[int, int],
                 img_pad_value: int = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255) -> None:
        super().__init__()
        self.size = size
        self.pad_mask = pad_mask
        self.pad_seg = pad_seg
        self.img_pad_value = img_pad_value
        self.mask_pad_value = mask_pad_value
        self.seg_pad_value = seg_pad_value

    def forward(
        self,
        inputs: Tensor,
        data_samples: Optional[List[dict]] = None
    ) -> Tuple[Tensor, Optional[List[dict]]]:
        """Pad image, instance masks, segmantic segmentation maps."""
        src_h, src_w = inputs.shape[-2:]
        dst_h, dst_w = self.size

        if src_h >= dst_h and src_w >= dst_w:
            return inputs, data_samples

        inputs = F.pad(
            inputs,
            pad=(0, max(0, dst_w - src_w), 0, max(0, dst_h - src_h)),
            mode='constant',
            value=self.img_pad_value)

        if data_samples is not None:
            # update batch_input_shape
            for data_sample in data_samples:
                data_sample.set_metainfo({
                    'batch_input_shape': (dst_h, dst_w),
                    'pad_shape': (dst_h, dst_w)
                })

            if self.pad_mask:
                for data_sample in data_samples:
                    masks = data_sample.gt_instances.masks
                    data_sample.gt_instances.masks = masks.pad(
                        (dst_h, dst_w), pad_val=self.mask_pad_value)

            if self.pad_seg:
                for data_sample in data_samples:
                    gt_sem_seg = data_sample.gt_sem_seg.sem_seg
                    h, w = gt_sem_seg.shape[-2:]
                    gt_sem_seg = F.pad(
                        gt_sem_seg,
                        pad=(0, max(0, dst_w - w), 0, max(0, dst_h - h)),
                        mode='constant',
                        value=self.seg_pad_value)
                    data_sample.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

        return inputs, data_samples
