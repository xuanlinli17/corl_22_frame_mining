"""
End-to-End Training of Deep Visuomotor Policies
    https://arxiv.org/pdf/1504.00702.pdf
Visuomotor as the base class of all visual polices.
"""
from pyrl.utils.meta import get_logger
import torch, torch.nn as nn
from pyrl.utils.torch import ExtendedModule, ExtendedModuleList
from pyrl.utils.data import GDict, DictArray, recover_with_mask, is_seq_of
from ..builder import build_model, BACKBONES
from copy import copy, deepcopy


def get_obj_pose_info_from_obs(obs, idx, use_rot):
    # extract target object-part poses
    assert idx is not None and isinstance(idx, list)
    if "target_box" in obs: # for OpenCabinetDrawer and OpenCabinetDoor, since a cabinet has several links, so we need to get the target link
        obj_pose_info = obs['target_box']
    elif "inst_box" in obs:
        obj_pose_info = obs['inst_box']
    else:
        raise NotImplementedError("For ManiSkill1 envs, please ensure that env_cfg.with_3d_ann=True")
    obj_pose_info = {'center': obj_pose_info[0][:, idx, :], 'rot': obj_pose_info[2][:, idx, :, :]} # center [B, Ninst, 3]; rot [B, Ninst, 3, 3]
    if not use_rot:
        obj_pose_info.pop('rot')
    return obj_pose_info     


@BACKBONES.register_module()
class Visuomotor(ExtendedModule):
    def __init__(self, visual_nn_cfg, mlp_cfg, visual_nn=None, 
                 obj_frame=-1, obj_frame_rot=False):
        """
        Visuomotor policy that takes in a (fused) point cloud represented in a single coordinate frame and outputs actions
            The specific coordinate frame used is specified through the config in pyrl/configs/mfrl/ppo/maniskill/maniskill_pn.py (or maniskill_spresnet.py)
                if env_cfg.nhand_pose is not specified (i.e. env_cfg.nhand_pose == 0):
                    if env_cfg.ego_mode=True -> robot base frame
                    if env_cfg.ego_mode=False -> world frame
                if env_cfg.nhand_pose > 0:
                    use the first end-effector frame to represent the input point cloud
                if env_cfg.with_3d_ann=True,
                    If obj_frame >= 0, this refers to the semantic id of the object frame to be used to transform the input point cloud
                    If obj_frame >= 0, then if obj_frame_rot==True, also use object-centric rotation to transform the input point cloud
        """             
        super(Visuomotor, self).__init__()
        # Feature extractor [Can be shared with other network]
        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn
        # Final mlp that maps the concatenation of visual feature and agent state to 
        self.final_mlp = build_model(mlp_cfg)

        self.saved_feature = None
        self.saved_visual_feature = None

        assert isinstance(obj_frame, int)
        self.obj_frame = obj_frame
        self.obj_frame_rot = obj_frame_rot

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        with_robot_state=True,
        **kwargs,
    ):
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        assert not (feature is not None and visual_feature is not None), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        save_feature = save_feature or (feature is not None or visual_feature is not None)
        obs_keys = obs.keys()
        obs = copy(obs)

        # get agent state vector
        robot_state = None
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)

        # Extract object frame information from observation, if it exists
        if self.obj_frame >= 0:
            obj_pose_info = get_obj_pose_info_from_obs(obs, [self.obj_frame], use_rot=self.obj_frame_rot)
            obj_pose_info['center'] = obj_pose_info['center'].squeeze(1)
            if 'rot' in obj_pose_info.keys():
                obj_pose_info['rot'] = obj_pose_info['rot'].squeeze(1)
        else:
            obj_pose_info = None
        
        # Extract end-effector frame information (we will only use the first end-effector here)
        hand_pose = obs.pop("hand_pose", None)

        # pop unnecessary keys
        for key in obs_keys:
            if "_box" in key or "_seg" in key or key in ["point_to_inst_sem_label", "point_to_target_sem_label"]:
                obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        # if visual feature is not saved, forward the visual backbone
        if feature is None:
            if visual_feature is None:
                assert not (hand_pose is not None and obj_pose_info is not None)
                if hand_pose is None and obj_pose_info is None:
                    feat = self.visual_nn(obs)
                elif hand_pose is not None:
                    feat = self.visual_nn(obs, hand_pose=hand_pose)
                elif obj_pose_info is not None:
                    feat = self.visual_nn(obs, obj_pose_info=obj_pose_info)
            else:
                feat = visual_feature

            if save_feature:
                self.saved_visual_feature = feat.clone()

            if robot_state is not None and with_robot_state:
                assert feat.ndim == robot_state.ndim, "Visual feature and state vector should have the same dimension!"
                feat = torch.cat([feat, robot_state], dim=-1)

            if save_feature:
                self.saved_feature = feat.clone()
        else:
            feat = feature

        # pass the concatenation of visual feature and agent state into final mlp
        if self.final_mlp is not None:
            feat = self.final_mlp(feat)

        return feat




@BACKBONES.register_module()
class VisuomotorTransformerFrame(ExtendedModule):
    def __init__(self, visual_nn_cfg, mlp_cfg, is_value=False, visual_nn=None,
                 mix_action=False,
                 fuse_feature_single_action=False, 
                 use_obj_frames=None, obj_frame_rot=False, 
                 debug=False):
        """
        Visuomotor policy that takes in a (fused) point cloud represented in a single coordinate frame and outputs actions
            The specific coordinate frame used is specified through the config in pyrl/configs/mfrl/ppo/maniskill/maniskill_pn.py (or maniskill_spresnet.py)
                if env_cfg.nhand_pose is not specified (i.e. env_cfg.nhand_pose == 0):
                    if env_cfg.ego_mode=True -> robot base frame
                    if env_cfg.ego_mode=False -> world frame
                if env_cfg.nhand_pose > 0:
                    use the first end-effector frame to represent the input point cloud
                if env_cfg.with_3d_ann=True,
                    If obj_frame >= 0, this refers to the semantic id of the object frame to be used to transform the input point cloud
                    If obj_frame >= 0, then if obj_frame_rot==True, also use object-centric rotation to transform the input point cloud
        """             
        super(VisuomotorTransformerFrame, self).__init__()

        # whether to use object frame information
        self.use_obj_frames = use_obj_frames
        if self.use_obj_frames is not None and visual_nn_cfg is not None:
            assert isinstance(self.use_obj_frames, list) and isinstance(self.use_obj_frames[0], int)
            visual_nn_cfg['num_obj_frames'] = len(self.use_obj_frames)       
        self.obj_frame_rot = obj_frame_rot             

        # Feature extractor [Can be shared with other network]
        self.visual_nn = build_model(visual_nn_cfg) if visual_nn is None else visual_nn
        # Number of input frames
        self.Nframe = self.visual_nn.num_frames
        # Number of input frames minus the number of target-object-part frames
        # In FM-FC and FM-TG, we don't use the object frame to output final action proposal;
        # However, in FM-MA, object frames will output final action proposals, since in FM-MA, each candidate frame predicts the entire action space, 
        # not the action space belonging to a particular (base or end-effector) frame 
        if self.use_obj_frames is not None and not mix_action: 
            self.Nframe_m_obj = self.Nframe - len(self.use_obj_frames) # base and hand frames
        else:
            self.Nframe_m_obj = self.Nframe 

        self.mix_action = mix_action # FrameMiner-MixAction
        self.fuse_feature_single_action = fuse_feature_single_action # whether to use FeatureConcat
        assert (int(isinstance(self.mix_action, dict)) + int(self.fuse_feature_single_action) <= 1), (
            "Only choose at most one option among mix_action and fuse_feature_single_action!"
        )

        self.base_action_dim = 4 # hardcoded
        self.per_hand_action_dim = 9 # hardcoded
        if self.use_obj_frames is None or not self.mix_action:
            self.full_action_dim = self.base_action_dim + (self.Nframe_m_obj - 1) * self.per_hand_action_dim # dim of the entire action space
        else: # if mix_action and use_obj_frames, then Nframe_m_obj == Nframe
            self.full_action_dim = self.base_action_dim + (self.Nframe_m_obj - len(self.use_obj_frames) - 1) * self.per_hand_action_dim
            
        self.is_value = is_value

        if self.mix_action and not self.is_value:
            assert isinstance(self.mix_action, dict), "mix_action has to be a network config!"
            self.mix_action.mlp_spec[0] = self.mix_action.mlp_spec[0] * self.Nframe_m_obj
            self.mix_action.mlp_spec.append(self.Nframe_m_obj * self.full_action_dim)
            self.mix_action_params = build_model(self.mix_action)
        else:
            self.mix_action_params = None

        if not self.is_value and not self.fuse_feature_single_action:
            # For action space decomposition, see https://github.com/haosulab/ManiSkill/wiki/Detailed-Explanation-of-Action
            self.final_mlp = ExtendedModuleList()

            body_mlp_cfg = deepcopy(mlp_cfg) # the final mlp used to output base-related actions
            if self.mix_action:
                body_mlp_cfg.mlp_spec.append(self.full_action_dim)
            else:
                body_mlp_cfg.mlp_spec.append(self.base_action_dim)
            self.final_mlp.append(build_model(body_mlp_cfg))

            for _ in range(self.Nframe_m_obj - 1):
                hand_mlp_cfg = deepcopy(mlp_cfg) # the final mlp used to output end-effector-related actions
                if self.mix_action:
                    hand_mlp_cfg.mlp_spec.append(self.full_action_dim) 
                else:
                    hand_mlp_cfg.mlp_spec.append(self.per_hand_action_dim)
                self.final_mlp.append(build_model(hand_mlp_cfg))
        elif not self.is_value and self.fuse_feature_single_action:
            mlp_cfg.mlp_spec[0] *= self.Nframe_m_obj
            mlp_cfg.mlp_spec.append(self.full_action_dim)
            self.final_mlp = build_model(mlp_cfg)
        else:
            # for value network, we simply ignore mix_action and fuse_feature_single_action
            mlp_cfg.mlp_spec[0] = mlp_cfg.mlp_spec[0] * self.Nframe_m_obj
            self.final_mlp = build_model(mlp_cfg)

        self.saved_feature = None
        self.saved_visual_feature = None

        self.debug = debug

    def forward(
        self,
        obs,
        feature=None,
        visual_feature=None,
        save_feature=False,
        **kwargs,
    ):
        assert isinstance(obs, dict), f"obs is not a dict! {type(obs)}"
        assert not (feature is not None and visual_feature is not None), f"You cannot provide visual_feature and feature at the same time!"
        self.saved_feature = None
        self.saved_visual_feature = None
        save_feature = save_feature or (feature is not None or visual_feature is not None)
        obs_keys = obs.keys()
        obs = copy(obs)

        # Extract end-effector poses
        assert "hand_pose" in obs_keys
        hand_pose = obs.pop("hand_pose") # [B, Nframe_m_obj - 1, 7]
        # Extract object frame information from observation, if it exists
        if self.use_obj_frames is not None:
            obj_pose_info = get_obj_pose_info_from_obs(obs, self.use_obj_frames, use_rot=self.obj_frame_rot)
        else:
            obj_pose_info = None        
        # Extract agent state
        robot_state = None
        for key in ["state", "agent"]:
            if key in obs:
                assert robot_state is None, f"Please provide only one robot state! Obs Keys: {obs_keys}"
                robot_state = obs.pop(key)

        # pop unnecessary keys
        for key in obs_keys:
            if "_box" in key or "_seg" in key or key in ["point_to_inst_sem_label", "point_to_target_sem_label"]:
                obs.pop(key)
        if not ("xyz" in obs or "rgb" in obs or "rgbd" in obs):
            assert len(obs) == 1, f"Observations need to contain only one visual element! Obs Keys: {obs.keys()}!"
            obs = obs[list(obs.keys())[0]]

        # if visual feature is not saved, forward the visual backbone
        if feature is None:
            if visual_feature is None:
                # hand_pose is used for transforming input point cloud into hand-centric point clouds
                # obj_pose_info is used for transforming input point cloud into object-centric point clouds
                feat = self.visual_nn(obs, hand_pose, robot_state=None, obj_pose_info=obj_pose_info)
                if obj_pose_info is not None:
                    feat = feat[:, :self.Nframe_m_obj, :] # obj_pose is only used for attention       
                # feat: [B, Nframe_m_obj, C]
            else:
                feat = visual_feature
            feat_cat = torch.cat([feat, robot_state[:, None, :].repeat_interleave(feat.size(1), dim=1)], dim=-1)
            if save_feature:
                self.saved_visual_feature = feat.clone()
                self.saved_feature = feat_cat.clone()
            feat = feat_cat
        else:
            feat = feature

        # Forward the final mlp, which takes the concatenation of visual feature and agent state as input
        if self.is_value or ((not self.is_value) and self.fuse_feature_single_action):
            feat = feat.view([feat.size(0), -1])
            feat = self.final_mlp(feat)
            return feat
        else:
            if self.mix_action:
                B, Nframe_m_obj, C = feat.size()
                mix_action_params = self.mix_action_params(feat.view(B, -1))
                mix_action_params = mix_action_params.view(B, Nframe_m_obj, -1) # the last dim equals the action dim
                if self.debug:
                    print(torch.softmax(mix_action_params[0], dim=0), flush=True)

            feat = feat.split(1, dim=1)
            feat = [x.squeeze(1) for x in feat]
            for i in range(self.Nframe_m_obj):
                feat[i] = self.final_mlp[i](feat[i])  

            if self.mix_action:
                feat = torch.stack(feat, dim=1)
                feat = feat * torch.softmax(mix_action_params, dim=1)
                feat = feat.sum(dim=1)
            else:
                feat = torch.cat(feat, dim=-1) # action decomposition follows the original ordering of base and hand_pose
            return feat