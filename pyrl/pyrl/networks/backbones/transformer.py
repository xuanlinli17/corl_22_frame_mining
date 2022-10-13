import torch
import torch.nn as nn
from pyrl.utils.torch import ExtendedModuleList
from ..builder import BACKBONES, build_backbone
from ..modules import build_attention_layer

from pyrl.utils.torch import ExtendedModule
from pyrl.utils.data import split_dim, GDict

from pytorch3d.transforms import quaternion_to_matrix

from .mlp import LinearMLP

import numpy as np
import open3d as o3d
import time


class TransformerBlock(ExtendedModule):
    def __init__(self, attention_cfg, mlp_cfg, dropout=None):
        super(TransformerBlock, self).__init__()
        self.attn = build_attention_layer(attention_cfg)
        self.mlp = build_backbone(mlp_cfg)
        assert mlp_cfg.mlp_spec[0] == mlp_cfg.mlp_spec[-1] == attention_cfg.embed_dim

        self.ln1 = nn.LayerNorm(attention_cfg.embed_dim)
        self.ln2 = nn.LayerNorm(attention_cfg.embed_dim)

        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x, mask=None, history=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim], the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length], a mask for disallowing attention to padding tokens
        :param history: [B, H, E] [history length, batch size, embed_dim], the histroy embeddings
        :param ret_history: bool, if we return the emebeding in previous segments
        :param histroy_len: int, the maximum number of history information we store
        :return: [B, N, E] [batch size, length, length] a single tensor containing the output from the Transformer block
        """
        ret_history = x if history is None else torch.cat([history, x], dim=1)
        o = self.attn(x, mask, history)
        x = x + o
        x = self.ln1(x)
        o = self.mlp(x)
        o = self.dropout(o)
        x = x + o
        x = self.ln2(x)
        return x, ret_history.detach()


@BACKBONES.register_module()
class TransformerEncoder(ExtendedModule):
    def __init__(self, block_cfg, pooling_cfg=None, mlp_cfg=None, num_blocks=6, with_task_embedding=False):
        super(TransformerEncoder, self).__init__()

        if with_task_embedding:
            embed_dim = block_cfg["attention_cfg"]["embed_dim"]
            self.task_embedding = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.xavier_normal_(self.task_embedding)
        self.with_task_embedding = with_task_embedding

        self.attn_blocks = ExtendedModuleList([TransformerBlock(**block_cfg) for i in range(num_blocks)])
        self.pooling = None if pooling_cfg is None else build_attention_layer(pooling_cfg, default_args=dict(type="AttentionPooling"))
        self.global_mlp = build_backbone(mlp_cfg) if mlp_cfg is not None else None

    def forward(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] or [B, N, F] A single tensor or a series of tensor containing the output from the Transformer
        """
        assert x.ndim == 3
        B, N, E = x.shape
        assert mask is None or list(mask.shape) == [B, N, N], f"{mask.shape} {[B, N, N]}"
        if mask is None:
            mask = torch.ones(B, N, N, dtype=x.dtype, device=x.device)
        mask = mask.type_as(x)
        if self.with_task_embedding:
            one = torch.ones_like(mask[:, :, 0])
            mask = torch.cat([one.unsqueeze(1), mask], dim=1)  # (B, N+1, N)
            one = torch.ones_like(mask[:, :, 0])
            mask = torch.cat([one.unsqueeze(2), mask], dim=2)  # (B, N+1, N+1)
            x = torch.cat([torch.repeat_interleave(self.task_embedding, x.size(0), dim=0), x], dim=1)

        for i, attn in enumerate(self.attn_blocks):
            x = attn(x, mask)[0]
        if self.pooling is not None:
            x = self.pooling(x)
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x

    def get_atten_score(self, x, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :return: [B, F] or [B, N, F] A single tensor or a series of tensor containing the output from the Transformer
        """
        assert x.ndim == 3
        B, N, E = x.shape
        assert mask is None or mask.shape == [B, N, N], f"{mask.shape} {[B, N, N]}"
        if mask is None:
            mask = torch.ones(B, N, N, dtype=x.dtype, device=x.device)
        mask = mask.type_as(x)

        ret = []
        for attn in self.attn_blocks:
            score = attn.attn.get_atten_score(x)
            x = attn(x, mask)[0]
            ret.append(score)
        return torch.stack(ret, dim=0)


@BACKBONES.register_module()
class TransformerXL(ExtendedModule):
    def __init__(self, block_cfg, latent_proj_cfg=None, mlp_cfg=None, num_blocks=6, history_len=-1, decode=True):
        super(TransformerXL, self).__init__()
        self.latent_proj = build_backbone(latent_proj_cfg)
        assert block_cfg["attention_cfg"].type == "MultiHeadSelfAttentionXL"

        # Create sharded bias in C and D terms in A_{i,j}^{rel}
        latent_dim = block_cfg["attention_cfg"]["latent_dim"]
        num_heads = block_cfg["attention_cfg"]["num_heads"]
        self.u = nn.Parameter(torch.Tensor(num_heads, latent_dim))
        self.v = nn.Parameter(torch.Tensor(num_heads, latent_dim))
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)
        block_cfg["attention_cfg"]["u"] = self.u
        block_cfg["attention_cfg"]["v"] = self.v

        self.attn_blocks = ExtendedModuleList([TransformerBlock(**block_cfg) for i in range(num_blocks)])
        self.num_blocks = num_blocks
        self.history_len = history_len
        self.decode = decode
        self.global_mlp = build_backbone(mlp_cfg)

    def forward(self, x, rnn_states=None, mask=None):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the Transformer, a tensor of shape
        :param mask: [B, N, N] [batch size, len, len] a mask for disallowing attention to padding tokens.
        :param rnn_states: [H * NL, B, E] [history length * num of layer, batch size, embed_dim] # the shape is to match the RNN hidden state.
        :return: [B, F] or [B, N, F] A single tensor or a series of tensor containing the output from the Transformer
        """
        # if rnn_states is not None:
        # print('R shape', rnn_states.shape)
        assert x.ndim == 3
        x = self.latent_proj(x) if self.latent_proj is not None else x
        # print(x.shape)
        B, N, E = x.shape
        assert mask is None or mask.shape == [B, N, N], f"{mask.shape} {[B, N, N]}"
        if rnn_states is None:
            if self.history_len > 0:
                rnn_states = torch.zeros(B, self.history_len, self.num_blocks, E, device=self.device, dtype=self.dtype)
        else:
            assert rnn_states.ndim == 3
            rnn_states = rnn_states.transpose(0, 1).contiguous()
            rnn_states = split_dim(rnn_states, 1, [-1, self.num_blocks])  # [B, H, NL, E]
        # H = rnn_states.shape[1]
        H = 0 if rnn_states is None else rnn_states.shape[1]
        # if rnn_states is not None:
        # rnn_states
        # Convert mask from [B, N, N] to [B, N, N + H]
        if mask is None:
            mask = torch.ones(B, N, N + H, dtype=x.dtype, device=x.device)
        elif H > 0:
            mask = torch.cat([torch.ones(B, N, H, dtype=x.dtype, device=x.device), mask.type_as(x)], dim=-1)
        if self.decode:
            mask[..., H:] = torch.tril(mask[..., H:], diagonal=0)  # [B, N, N]
        new_states = []
        for i, attn in enumerate(self.attn_blocks):
            state_i = rnn_states[:, :, i]  # [B, H, E]
            # print(x.shape, state_i.shape, rnn_states.shape)
            # exit(0)
            x, state_i = attn(x, mask, state_i)
            new_states.append(state_i)
        new_states = torch.stack(new_states, dim=2)  # [B, H, NL, E]
        if self.history_len > 0:
            if new_states.shape[1] < self.history_len:
                pad_shape = list(new_states.shape)
                pad_shape[1] = self.history_len - pad_shape[1]
                new_states = torch.cat([torch.zeros(*pad_shape, dtype=new_states.dtype, device=new_states.device), new_states], axis=1)
            elif new_states.shape[1] > self.history_len:
                new_states = new_states[:, -self.history_len :]
        new_states = new_states.reshape([B, -1, E])
        new_states = new_states.transpose(0, 1)
        if self.global_mlp is not None:
            x = self.global_mlp(x)
        return x, new_states


@BACKBONES.register_module()
class TransformerFrame(ExtendedModule):
    def __init__(self, num_frames, backbone_cfg, transformer_cfg=None, mask_type='full',
                 num_obj_frames=0):
        """
        Decompose the action space into the base component and components for each end effector.
        The feature for action output in each component is a fusion (by attention) of global features
        from point clouds transformed into different frames (the base frame and the ee-frames).

        State and action components in ManiSkill:
        https://github.com/haosulab/ManiSkill/wiki/Detailed-Explanation-of-The-Agent-State-Vector
        https://github.com/haosulab/ManiSkill/wiki/Detailed-Explanation-of-Action

        Note that if point cloud is egocentric, then `base_pos` and `base_orientation` are removed from the state vector. 

        num_frames: in [2 + num_obj_frames, 3 + num_obj_frames]; 
            2 = [base, ee_arm1] for single arm env; 3 = [base, ee_arm1, ee_arm2] for dual arm env
            num_obj_frames = number of object frames used
        num_obj_frames: int

        backbone_cfg: point cloud feature extraction backbone (e.g. PointNet)
            In config file, you can just set num_frames to "nhand + 1" (i.e. base + nhands), which will be replaced by 
            function "get_kwargs_from_shape" in pyrl/networks/utils.py

        mask_type: 'full' or 'identity'; 'full' = attention mask all 1; 'identity' = identity attention mask
            'skip': skip the Transformer block entirely

        """
        super(TransformerFrame, self).__init__()

        self.num_obj_frames = num_obj_frames
        assert num_frames in [2 + num_obj_frames, 3 + num_obj_frames], f"num_frames: {num_frames}"
        self.num_frames = num_frames
        self.backbones = ExtendedModuleList([build_backbone(backbone_cfg) for i in range(self.num_frames)])
        self.mask_type = mask_type
        assert self.mask_type in ['full', 'identity', 'skip']
        if self.mask_type == 'skip':
            self.attn = None
        else:
            self.attn = build_backbone(transformer_cfg) if transformer_cfg is not None else None

        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window() 
        # self.opt = self.vis.get_render_option()
        # self.opt.background_color = np.array([0.5,0.5,0.5])
        # self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        # self.geometry = o3d.geometry.PointCloud()
        # self.geometry.points = o3d.utility.Vector3dVector(np.random.random(size=(10,3)))
        # self.geometry.colors = o3d.utility.Vector3dVector(np.random.random(size=(10,3)))
        # self.vis.add_geometry(self.geometry)

    def forward(self, pcd, hand_pose, robot_state, obj_pose_info=None, return_pre_tf=False, **kwargs):
        # hand_pose: [B, nhands, 7]
        # obj_pose_info = {'center': [B, Ninst, 3], 'rot': [B, Ninst, 3, 3]}
        assert isinstance(pcd, dict) and "xyz" in pcd
        pcds = [pcd.copy() for _ in range(self.num_frames)]
        nhands = hand_pose.shape[1]
        assert nhands + 1 + self.num_obj_frames == self.num_frames, (
            f"nhands + 1: {nhands + 1}, num_obj_frames: {self.num_obj_frames}, num_frames: {self.num_frames}")
        for i in range(nhands):
            cur_hand_xyz = hand_pose[:, i, :3]
            cur_hand_quat = hand_pose[:, i, 3:]
            cur_hand_rot = quaternion_to_matrix(cur_hand_quat)
            cur_xyz = pcd['xyz'] - cur_hand_xyz[:, None, :]
            cur_xyz = torch.einsum('bni,bij->bnj', cur_xyz, cur_hand_rot)
            pcds[i + 1]['xyz'] = cur_xyz
        if obj_pose_info is not None:
            center = obj_pose_info['center']
            if 'rot' in obj_pose_info.keys():
                rot = obj_pose_info['rot']
            else:
                rot = None
            for i in range(center.shape[1]):
                cur_xyz = pcd['xyz'] - center[:, [i], :]
                if rot is not None:
                    cur_xyz = torch.einsum('bni,bij->bnj', cur_xyz, rot[:, i, :, :])
                pcds[1 + nhands + i]['xyz'] = cur_xyz        
        
        # self.geometry.points = o3d.utility.Vector3dVector(
        #     torch.cat([pcds[i]['xyz'][0] + torch.tensor([[0.0, 2.0, 0.0]], device=pcds[i]['xyz'].device)*i for i in range(len(pcds))], axis=0).detach().cpu().numpy()
        # )        
        # self.geometry.colors = o3d.utility.Vector3dVector(
        #     torch.cat([pcds[0]['rgb'][0] for _ in range(len(pcds))], axis=0).detach().cpu().numpy()
        # )
        # self.vis.update_geometry(self.geometry)
        # tt = time.time()
        # while time.time() - tt < 1.0:
        #     self.vis.poll_events()
        #     self.vis.update_renderer()

        feats = [self.backbones[i](pcds[i], concat_state=robot_state) for i in range(self.num_frames)] # [B, Nframe, C]
        feats = torch.stack(feats, dim=1)
        feats_pre_tf = feats
        if self.mask_type != 'skip':
            if self.attn is not None:
                B, Nframe = feats.size(0), feats.size(1)
                if self.mask_type == 'identity':
                    mask = torch.ones([B, Nframe, Nframe], device=feats.device)
                    mask = mask * torch.eye(Nframe, device=feats.device)[None, :, :]
                elif self.mask_type == 'full':
                    mask = None
                feats = self.attn(feats, mask=mask)
        if return_pre_tf:
            return feats_pre_tf, feats
        else:
            return feats




@BACKBONES.register_module()
class TransformerLinkVisual(ExtendedModule):
    def __init__(self, cross_attn_cfg, self_attn_cfg, action_dim, embed_dim, robot_state_dim, num_blocks=1,):
        super(TransformerLinkVisual, self).__init__()

        self.link_embedding = nn.Parameter(torch.empty(action_dim, embed_dim))
        nn.init.xavier_normal_(self.link_embedding)

        self.num_blocks = int(num_blocks)
        self.cross_attn_blocks = ExtendedModuleList()
        self.ln_cross_attn = ExtendedModuleList()
        self.self_attn_blocks = ExtendedModuleList()
        self.ln_self_attn = ExtendedModuleList()
        self.projections = ExtendedModuleList()
        self.ff_blocks = ExtendedModuleList()
        self.ln_ff = ExtendedModuleList()
        for i in range(self.num_blocks):
            self.cross_attn_blocks.append(build_attention_layer(cross_attn_cfg))
            self.ln_cross_attn.append(nn.LayerNorm(embed_dim))
            self.self_attn_blocks.append(build_attention_layer(self_attn_cfg))
            self.ln_self_attn.append(nn.LayerNorm(embed_dim))
            self.projections.append(nn.Linear(embed_dim + robot_state_dim, embed_dim))
            self.ff_blocks.append(
                LinearMLP(mlp_spec=[embed_dim, embed_dim * 3, embed_dim], norm_cfg=None, bias="auto", inactivated_output=False)
            )
            self.ln_ff.append(nn.LayerNorm(embed_dim))

    def forward(self, visual_feature, robot_state):
        """
        :param visual_feature: [B, Nframe, E] ([batch size, Nframe, embed_dim])
        :param robot_state: [B, dim_robot_state]

        :return: [B, dim_action, E']
        """
        assert visual_feature.ndim == 3
        B, Nframe, E = visual_feature.shape
        visual_feature = torch.cat([visual_feature, robot_state[:, None, :].repeat_interleave(Nframe, dim=1)], dim=-1)
        x = self.link_embedding[None, :, :].repeat_interleave(B, dim=0) # [B, action_dim, E]

        if self.num_blocks == 1:
            cur_vf_proj = self.projections[0](visual_feature)
            x = x + self.cross_attn_blocks[0](cur_vf_proj, x)
            x = self.ln_cross_attn[0](x)
            x = x + self.self_attn_blocks[0](x)
            x = self.ln_self_attn[0](x)
            x = x + self.ff_blocks[0](x)
            x = self.ln_ff[0](x)            
        else:
            for i in range(self.num_blocks):
                cur_vf_proj = self.projections[i](visual_feature)
                x = x + self.self_attn_blocks[i](x)
                x = self.ln_self_attn[i](x)
                x = x + self.cross_attn_blocks[i](cur_vf_proj, x)
                x = self.ln_cross_attn[i](x)
                x = x + self.ff_blocks[i](x)
                x = self.ln_ff[i](x)

        return x
        