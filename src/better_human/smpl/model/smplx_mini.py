import pickle
import numpy as np
import json
from importlib import resources

import torch
import pypose as pp
# ... (other imports)
from ..base import SMPLBase, SMPLOutputs


class SMPLXMini(SMPLBase):
    """
    A concrete implementation of the SMPL-X model, which extends SMPL
    with hand and face pose parameters.
    """
    def __init__(
            self, 
            model_path: str,
            gender: str = 'neutral',
            num_betas: int = 10,
            **kwargs
    ):
        
        self.num_betas = num_betas
        self.gender = gender

        super().__init__(**kwargs)

        smplx_data = np.load(model_path, allow_pickle=True)
        
        # self._load_model_base_data(smplx_data, model_config='smplx_mini.json')
        self._load_model_specific_data(smplx_data, model_config='smplx_mini.json')
        self._compute_static_variables()

    def _load_model_specific_data(self, model_data: dict, model_config: str = None):

        # load config as class attributes
        with resources.files('better_human.smpl.config').joinpath(model_config).open('r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

        with resources.files('better_human.smpl.config').joinpath(self.vertices_segmentation_type).open('r') as f:
            self.vertices_segmentation = json.load(f)

        # Register model parameters as buffers
        self.register_buffer('vertices_template', torch.tensor(model_data['v_template'], dtype=torch.float32)) # (V, 3)
        
        # Faces are not used in computation but are essential for visualization
        self.register_buffer('faces', torch.tensor(model_data['f'].astype(np.int64), dtype=torch.long)) # (F, 3)

        # Shape blend shapes
        shape_blending = torch.tensor(model_data['shapedirs'][:, :, :self.num_betas], dtype=torch.float32)
        self.register_buffer('shape_blending', shape_blending)   # (V, 3, num_betas)

        # Pose blend shapes
        # Original shape is (V, 3, (J-1)*9). Reshape to (V*3, (J-1)*9)
        pose_blending = torch.tensor(model_data['posedirs'], dtype=torch.float32)

        pose_blending = pose_blending.reshape(self.num_vertices, 3, (55-1), 9)
        pose_blending = pose_blending[:, :, :self.num_joints-1, :].reshape(self.num_vertices, 3, (self.num_joints-1)*9)

        # We need ( (J-1)*9, V*3 ) for matmul, so we transpose
        pose_blending = pose_blending.reshape(-1, (self.num_joints-1)*9).T

        self.register_buffer('pose_blending', pose_blending) # ( (J-1)*9, V*3 )

        # Joint regressor
        self.register_buffer('joint_regressor', torch.tensor(model_data['J_regressor'][:22, :], dtype=torch.float32)) # (J, V)

        # LBS weights
        lbs_weights_old = torch.tensor(model_data['weights'], dtype=torch.float32) # (V, 55)
        lbs_weights = torch.tensor(model_data['weights'][:, :22], dtype=torch.float32) # (V, J)

        lbs_weights[:, 15] += lbs_weights_old[:, 22: 25].sum(dim=1)
        lbs_weights[:, 20] += lbs_weights_old[:, 25: 40].sum(dim=1)
        lbs_weights[:, 21] += lbs_weights_old[:, 40: 55].sum(dim=1)
        
        self.register_buffer('lbs_weights', lbs_weights) # (V, J)

        # Kinematic tree
        self.parent_tree = model_data['kintree_table'][:, :22]

        self.kinematic_order = [
            0, 
            1, 4, 7, 10, 
            2, 5, 8, 11, 
            3, 6, 9, 
            12, 15, 
            13, 16, 18, 20, 
            14, 17, 19, 21,
        ]

        path = resources.files('better_human.smpl.config').joinpath("smplx_mini_3d_segmentation.npy")
        self.segmentation_3d = np.load(path, allow_pickle=True).item()

        with resources.files('better_human.smpl.config').joinpath("body_densities.json").open('r') as f:
            self.body_densities = json.load(f)
    

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
        
