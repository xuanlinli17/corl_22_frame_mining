import numpy as np
from copy import deepcopy
import torch, torch.nn as nn, torch.nn.functional as F

from .mlp import ConvMLP, LinearMLP
from ..builder import BACKBONES
from pyrl.utils.torch import ExtendedModule

from pytorch3d.transforms import quaternion_to_matrix

class SE3Net(ExtendedModule):
    def __init__(self, in_channels=3, mlp_spec=[64, 128, 256], norm_cfg=dict(type="LN1d", eps=1e-6), act_cfg=dict(type="ReLU")):
        super(SE3Net, self).__init__()
        self.conv = ConvMLP(
            [in_channels] + mlp_spec,
            norm_cfg,
            act_cfg=act_cfg,
            inactivated_output=False,
        )  # k -> 64 -> 128 -> 256
        self.mlp = LinearMLP([mlp_spec[-1], 128, 9], norm_cfg=None, act_cfg=act_cfg, inactivated_output=True)

    def forward(self, feature):
        assert feature.ndim == 3, f"Feature shape {feature.shape}!"
        feature = self.mlp(self.conv(feature).max(-1)[0])
        feature[:, 0] += 1.0
        feature[:, 4] += 1.0
        return feature


@BACKBONES.register_module()
class PointNet(ExtendedModule):
    def __init__(
        self,
        feat_dim,
        mlp_spec=[64, 128, 1024],
        global_feat=True,
        norm_cfg=dict(type="LN1d", eps=1e-6),
        act_cfg=dict(type="ReLU"),
        learn_se3=False,
    ):
        super(PointNet, self).__init__()
        self.global_feat = global_feat

        mlp_spec = deepcopy(mlp_spec)
        self.conv = ConvMLP(
            [
                feat_dim,
            ]
            + mlp_spec,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inactivated_output=False,
        )
        self.learn_se3 = learn_se3
        if self.learn_se3:
            self.se3_net = SE3Net(in_channels=feat_dim)

    def forward(self, inputs, concat_state=None, **kwargs):
        xyz = inputs["xyz"] if isinstance(inputs, dict) else inputs
        assert not ("hand_pose" in kwargs.keys() and "obj_pose_info" in kwargs.keys())
        if "hand_pose" in kwargs.keys():
            hand_pose = kwargs.pop("hand_pose")
            # use the first hand to transform point cloud coordinates
            hand_xyz = hand_pose[:, 0, :3]
            hand_quat = hand_pose[:, 0, 3:]
            hand_rot = quaternion_to_matrix(hand_quat)
            xyz = xyz - hand_xyz[:, None, :]
            xyz = torch.einsum('bni,bij->bnj', xyz, hand_rot)   
        if "obj_pose_info" in kwargs.keys():
            # use the object part pose to transform point cloud coordinates
            obj_pose_info = kwargs.pop("obj_pose_info")
            center = obj_pose_info['center'] # [B, 3]
            xyz = xyz - center[:, None, :]
            if 'rot' in obj_pose_info.keys():
                rot = obj_pose_info['rot'] # [B, 3, 3]
                xyz = torch.einsum('bni,bij->bnj', xyz, rot)  

        with torch.no_grad():
            if isinstance(inputs, dict):
                feature = [xyz]
                if "rgb" in inputs:
                    feature.append(inputs["rgb"])
                if "seg" in inputs:
                    feature.append(inputs["seg"])
                if concat_state is not None: # [B, C]
                    feature.append(concat_state[:, None, :].expand(-1, xyz.shape[1], -1))
                feature = torch.cat(feature, dim=-1)
            else:
                feature = xyz

            feature = feature.permute(0, 2, 1).contiguous()

        if self.learn_se3:
            # Learn adaptive SE(3) based on the point cloud input
            se3_out = self.se3_net(feature)
            rot_ax_1, rot_ax_2 = se3_out[:, :3], se3_out[:, 3:6]
            transl = se3_out[:, 6:]
            xyz = xyz - transl[:, None, :]
            # 6D rotation representation
            rot_ax_1 = F.normalize(rot_ax_1, dim=-1)
            rot_ax_3 = torch.cross(rot_ax_1, rot_ax_2)
            rot_ax_3 = F.normalize(rot_ax_3)
            rot_ax_2 = torch.cross(rot_ax_3, rot_ax_1)
            rot = torch.stack([rot_ax_1, rot_ax_2, rot_ax_3], dim=-1)
            xyz = torch.einsum('bni,bij->bnj', xyz, rot)
            if not isinstance(inputs, dict):
                feature = xyz
            else:
                feature = torch.cat([xyz.permute(0, 2, 1), feature[:, 3:, :]], dim=1)

        feature = self.conv(feature)
        if self.global_feat:
            feature = feature.max(-1)[0]
        else:
            gl_feature = feature.max(-1, keepdims=True)[0].repeat(1, 1, feature.shape[-1])
            feature = torch.cat([feature, gl_feature], dim=1)
        return feature

