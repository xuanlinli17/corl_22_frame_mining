import numpy as np
from sapien.core import Pose, Articulation, ActorBase

from pyrl.utils.data import is_num
from pyrl.utils.lib3d import apply_pose, check_coplanar, create_obb, create_aabb



def get_colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return int((byteval & (1 << idx)) != 0)

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = [r, g, b]
    cmap = cmap / 255 if normalized else cmap
    return cmap


PALETTE_C = get_colormap(256, False)


def get_actors_by_ids(actors, ids):
    assert isinstance(actors, (list, tuple))
    # Actors can be joint and link
    if is_num(ids):
        ids = [ids]
        sign = True
    else:
        sign = False
    ret = [None for _ in ids]
    for actor in actors:
        if actor.get_id() in ids:
            ret[ids.index(actor.get_id())] = actor
    return ret[0] if sign else ret



def actor_to_bbox(
    actor,
    base_pose=None,
    merge_articulation=True,
    box_type="aabb",
    to_vector=False,
    actor_exclude=[],
    part_exclude=[],
    plaette=PALETTE_C,
):
    if actor.get_name() in actor_exclude:
        return None
    if isinstance(actor, Articulation):
        # Compute obb for the whole articulation
        if merge_articulation:
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
            maxs = -mins
            inv_pose = actor.get_pose().inv() if box_type == "obb" else Pose()
            for link in actor.get_links():
                aabb = actor_to_bbox(link, inv_pose, merge_articulation, "aabb", True, actor_exclude, part_exclude, plaette)
                if aabb is None:
                    continue
                # print(type(aabb), aabb)
                center, extent = aabb
                mins = np.minimum(center - extent / 2, mins)
                maxs = np.maximum(center + extent / 2, maxs)

            link_id = link.get_id()
            transform = actor.get_pose().to_transformation_matrix()
            # print(actor.get_pose(), actor)
            center = (mins + maxs) / 2
            extent = maxs - mins
            center = transform[:3, :3] @ center + transform[:3, 3]
            center_edge = np.concatenate([center, extent], axis=-1)
            # bbox = bbox.rotate(transform[:3, :3])
            # bbox = bbox.translate(transform[:3, 3])
            if to_vector:
                ret = [center, extent, transform[:3, :3]]
            else:
                ret = create_obb(center_edge, R=transform[:3, :3], color=plaette[link_id] / 255.0)
        else:
            ret = {}
            for link in actor.get_links():
                ret[link.get_name()] = actor_to_bbox(link, base_pose, merge_articulation, box_type, to_vector, actor_exclude, part_exclude, plaette)
    else:
        assert isinstance(actor, ActorBase)
        mins = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        maxs = -mins
        is_empty = True

        # ret = []

        for visual_body in actor.get_visual_bodies():
            if visual_body.get_name() in part_exclude:
                continue
            if visual_body.type == "mesh":
                scale = visual_body.scale
            elif visual_body.type == "box":
                scale = visual_body.half_lengths
            elif visual_body.type == "sphere":
                scale = visual_body.radius
            for i in visual_body.get_render_shapes():
                if i.mesh.indices.reshape(-1, 3).shape[0] < 4 or check_coplanar(i.mesh.vertices * scale):
                    continue
                pose = visual_body.local_pose
                if box_type == "aabb":
                    pose = base_pose * actor.get_pose() * pose
                else:
                    assert base_pose is None
                vertices = apply_pose(pose, i.mesh.vertices * scale)
                mins = np.minimum(mins, vertices.min(0))
                maxs = np.maximum(maxs, vertices.max(0))
                # mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))

                # ret.append(mesh)
                is_empty = False

        if is_empty:
            return None
        if box_type == "aabb":
            center = (mins + maxs) / 2
            extent = maxs - mins
            center_edge = np.concatenate([center, extent], axis=-1)
            if to_vector:
                ret = [center, extent]
            else:
                ret = create_aabb(center_edge, color=plaette[actor.get_id()] / 255.0)
        else:
            transform = actor.get_pose().to_transformation_matrix()
            # print(actor.get_pose(), actor)
            center = (mins + maxs) / 2
            extent = maxs - mins
            center = transform[:3, :3] @ center + transform[:3, 3]
            center_edge = np.concatenate([center, extent], axis=-1)
            # bbox = bbox.rotate(transform[:3, :3])
            # bbox = bbox.translate(transform[:3, 3])
            if to_vector:
                ret = [center, extent, transform[:3, :3]]
            else:
                ret = create_obb(center_edge, R=transform[:3, :3], color=plaette[actor.get_id()] / 255.0)
    return ret
