from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class nnUNetTrainerNoMBC(nnUNetTrainer):
    """
    Disables mirroring and Brightness/Contrast Augmentation
    """

    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_training_transforms(self, *args, **kwargs):
        tr_transforms = super().get_training_transforms(*args, **kwargs)
        self.print_to_log_file(f"Old transforms: {tr_transforms.transforms}")

        def _strip_undesired(comp_transform: ComposeTransforms) -> ComposeTransforms:
            to_keep = []
            for t in comp_transform.transforms:
                if isinstance(t, RandomTransform):
                    if isinstance(t.transform, (MultiplicativeBrightnessTransform, ContrastTransform)):
                        continue
                to_keep.append(t)
            return ComposeTransforms(to_keep)

        new_tr_transforms = _strip_undesired(tr_transforms)
        self.print_to_log_file(f"New transforms: {new_tr_transforms.transforms}")
        return new_tr_transforms
