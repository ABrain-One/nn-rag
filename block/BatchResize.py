# Auto-generated single-file for BatchResize
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Union
import numpy as np

# ---- original imports from contributing modules ----
from mmdet.registry import MODELS

# ---- mmdet.structures.det_data_sample.DetDataSample ----
class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.
        - ``pred_track_instances``(InstanceData): Instances of tracking
            predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> pred_track_instances = InstanceData(metainfo=img_meta)
         >>> pred_track_instances.bboxes = torch.rand((5, 4))
         >>> pred_track_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(
         ...    pred_track_instances=pred_track_instances)
         >>> assert 'pred_track_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    def proposals(self) -> InstanceData:
        return self._proposals

    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    def proposals(self):
        del self._proposals

    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    def gt_instances(self):
        del self._gt_instances

    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    def pred_instances(self):
        del self._pred_instances

    # directly add ``pred_track_instances`` in ``DetDataSample``
    # so that the ``TrackDataSample`` does not bother to access the
    # instance-level information.
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, '_pred_track_instances', dtype=InstanceData)

    def pred_track_instances(self):
        del self._pred_track_instances

    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    def ignored_instances(self):
        del self._ignored_instances

    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    def gt_sem_seg(self):
        del self._gt_sem_seg

    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    def pred_sem_seg(self):
        del self._pred_sem_seg

# ---- BatchResize (target) ----
class BatchResize(nn.Module):
    """Batch resize during training. This implementation is modified from
    https://github.com/Purkialo/CrowdDet/blob/master/lib/data/CrowdHuman.py.

    It provides the data pre-processing as follows:
    - A batch of all images will pad to a uniform size and stack them into
      a torch.Tensor by `DetDataPreprocessor`.
    - `BatchFixShapeResize` resize all images to the target size.
    - Padding images to make sure the size of image can be divisible by
      ``pad_size_divisor``.

    Args:
        scale (tuple): Images scales for resizing.
        pad_size_divisor (int): Image size divisible factor.
            Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
    """

    def __init__(
        self,
        scale: tuple,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
    ) -> None:
        super().__init__()
        self.min_size = min(scale)
        self.max_size = max(scale)
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(
        self, inputs: Tensor, data_samples: List[DetDataSample]
    ) -> Tuple[Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes."""

        batch_height, batch_width = inputs.shape[-2:]
        target_height, target_width, scale = self.get_target_size(
            batch_height, batch_width)

        inputs = F.interpolate(
            inputs,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False)

        inputs = self.get_padded_tensor(inputs, self.pad_value)

        if data_samples is not None:
            batch_input_shape = tuple(inputs.size()[-2:])
            for data_sample in data_samples:
                img_shape = [
                    int(scale * _) for _ in list(data_sample.img_shape)
                ]
                data_sample.set_metainfo({
                    'img_shape': tuple(img_shape),
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': batch_input_shape,
                    'scale_factor': (scale, scale)
                })

                data_sample.gt_instances.bboxes *= scale
                data_sample.ignored_instances.bboxes *= scale

        return inputs, data_samples

    def get_target_size(self, height: int,
                        width: int) -> Tuple[int, int, float]:
        """Get the target size of a batch of images based on data and scale."""
        im_size_min = np.min([height, width])
        im_size_max = np.max([height, width])
        scale = self.min_size / im_size_min
        if scale * im_size_max > self.max_size:
            scale = self.max_size / im_size_max
        target_height, target_width = int(round(height * scale)), int(
            round(width * scale))
        return target_height, target_width, scale

    def get_padded_tensor(self, tensor: Tensor, pad_value: int) -> Tensor:
        """Pad images according to pad_size_divisor."""
        assert tensor.ndim == 4
        target_height, target_width = tensor.shape[-2], tensor.shape[-1]
        divisor = self.pad_size_divisor
        padded_height = (target_height + divisor - 1) // divisor * divisor
        padded_width = (target_width + divisor - 1) // divisor * divisor
        padded_tensor = torch.ones([
            tensor.shape[0], tensor.shape[1], padded_height, padded_width
        ]) * pad_value
        padded_tensor = padded_tensor.type_as(tensor)
        padded_tensor[:, :, :target_height, :target_width] = tensor
        return padded_tensor
