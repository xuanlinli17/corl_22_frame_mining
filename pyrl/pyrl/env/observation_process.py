import numpy as np
from pyrl.utils.data import sample_and_pad
from pyrl.utils.meta import get_logger


def select_mask(obs, key, mask):
    if key in obs:
        obs[key] = obs[key][mask]


def pcd_uniform_downsample(pcd, num=1200):
    return


def pcd_filter_ground(pcd, eps=1e-3):
    return pcd["xyz"][..., -1] > eps


def pcd_filter_with_mask(obs, mask, env=None):
    # A;; modification happen in this function will be in-place.
    assert isinstance(obs, dict), f"{type(obs)}"
    if "xyz" in obs:
        # Raw point cloud
        for key in ["xyz", "rgb", "seg"]:
            select_mask(obs, key, mask)
    else:
        obs_mode = env.obs_mode
        pcd_filter_with_mask(obs[obs_mode], mask, env)
        for key in ["inst_seg", "target_seg"]:
            select_mask(obs, key, mask)


def pcd_base(obs, env=None, num=1200):
    if not isinstance(obs, dict):
        return obs
    obs_mode = env.obs_mode
    if obs_mode in ["state"]:
        return obs
    if obs_mode in ["pointcloud", "pointcloud_3d_ann"]:

        mask = obs[obs_mode]["xyz"][:, 2] > 1e-3
        for key in ["xyz", "rgb", "seg"]:
            select_mask(obs[obs_mode], key, mask)
        for key in ["inst_seg", "target_seg"]:
            select_mask(obs, key, mask)

        seg = obs[obs_mode]["seg"]

        tot_pts = num
        target_mask_pts = num // 3 * 2
        min_pts = num // 24

        num_pts = np.sum(seg, axis=0)
        base_num = np.minimum(num_pts, min_pts)
        remain_pts = num_pts - base_num
        tgt_pts = base_num + (target_mask_pts - base_num.sum()) * remain_pts // remain_pts.sum()
        back_pts = tot_pts - np.sum(tgt_pts)

        bk_seg = ~seg.any(-1, keepdims=True)
        seg_all = np.concatenate([seg, bk_seg], axis=-1)
        num_all = seg_all.sum(-1)
        tgt_pts = np.concatenate([tgt_pts, np.array([back_pts])], axis=-1)

        chosen_index = []
        for i in range(seg_all.shape[1]):
            if num_all[i] == 0:
                continue
            cur_seg = np.where(seg_all[:, i])[0]
            np.random.shuffle(cur_seg)
            shuffle_indices = cur_seg[: tgt_pts[i]]
            chosen_index.append(shuffle_indices)
        chosen_index = np.concatenate(chosen_index, axis=0)

        if len(chosen_index) < tot_pts:
            n, m = tot_pts // len(chosen_index), tot_pts % len(chosen_index)
            chosen_index = np.concatenate([chosen_index] * n + [chosen_index[:m]], axis=0)
        for key in ["xyz", "rgb", "seg"]:
            select_mask(obs[obs_mode], key, chosen_index)
        for key in ["inst_seg", "target_seg"]:
            select_mask(obs, key, chosen_index)
        return obs
    else:
        get_logger().info(f"Unknown observation mode {obs_mode}")
        exit(0)


def pcd_uniform_downsample(obs, env=None, ground_eps=1e-3, num=1200):
    obs_mode = env.obs_mode

    if ground_eps is not None:
        pcd_filter_with_mask(obs, pcd_filter_ground(obs[obs_mode], eps=ground_eps), env)

    pcd_filter_with_mask(obs, sample_and_pad(obs[obs_mode]["xyz"].shape[0], num), env)
    return obs

