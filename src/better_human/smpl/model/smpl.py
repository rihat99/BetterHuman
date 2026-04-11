import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp
# ... (other imports)
from ..base import SMPLBase, SMPLOutputs

class SMPL(SMPLBase):
    """
    A concrete implementation of the original SMPL model.
    """
    def __init__(self, model_path: str, gender: str = 'neutral', num_betas: int = 10, **kwargs):
        self.num_betas = num_betas
        self.gender = gender

        super().__init__(**kwargs)

        smpl_data = np.load(model_path, allow_pickle=True)

        self._load_model_specific_data(smpl_data)
        self._load_model_base_data(smpl_data, model_config='smpl.json')
        self._compute_static_variables()

    def _load_model_specific_data(self, model_data: dict):
        """
        Loads the SMPL model data from a .pkl file.
        """

        path = resources.files('better_human.smpl.config').joinpath("smpl_3d_segmentation.npy")
        self.segmentation_3d = np.load(path, allow_pickle=True).item()

        with resources.files('better_human.smpl.config').joinpath("body_densities.json").open('r') as f:
            self.body_densities = json.load(f)

        self.kinematic_order = [
            0, 
            1, 4, 7, 10, 
            2, 5, 8, 11, 
            3, 6, 9, 
            12, 15, 
            13, 16, 18, 20, 22, 
            14, 17, 19, 21, 23,
        ]

    def from_classic(
        self,
        betas: torch.Tensor,
        body_pose: torch.Tensor,
        translation: torch.Tensor,
        global_orient: torch.Tensor,
    ):
        
        _, neutral_joints = self.forward_shape(betas)
        root_offset = neutral_joints[:, 0, :3]  # (B, 3)

        shape = translation.shape[:-1]

        if len(shape) > 1:
            root_offset = root_offset.unsqueeze(1)

        q = torch.zeros((*shape, self.nq), device=self.vertices_template.device)

        q[..., :3] = translation + root_offset
        # q[..., :3] = translation
        q[..., 3:7] = pp.so3(global_orient).Exp().tensor()
        q[..., 7:] = pp.so3(body_pose.reshape(*shape, self.num_joints-1, 3)).Exp().tensor().reshape(*shape, (self.num_joints-1)*4)

        return q
    
    def to_classic(self, betas: torch.Tensor, q: torch.Tensor):
        
        _, neutral_joints = self.forward_shape(betas)
        root_offset = neutral_joints[:, 0, :3]  # (B, 3)

        shape = q.shape[:-1]

        if len(shape) > 1:
            root_offset = root_offset.unsqueeze(1)

        translation = q[..., :3] - root_offset
        global_orient = pp.SO3(q[..., 3:7]).Log().tensor()
        body_pose = pp.SO3(q[..., 7:].reshape(*shape, self.num_joints-1, 4)).Log().tensor().reshape(*shape, (self.num_joints-1)*3)
    
        return {
            'betas': betas,
            'body_pose': body_pose,
            'translation': translation,
            'global_orient': global_orient,
        }