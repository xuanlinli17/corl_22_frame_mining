import numpy as np, trimesh, sklearn
from sapien.core import Pose
from .o3d_utils import is_o3d, to_o3d, np2mesh
from pyrl.utils.data import normalize, to_gc, to_nc



def angle(x1, x2):
    if isinstance(x1, np.ndarray):
        x1 = normalize(x1)
        x2 = normalize(x2)
        return np.arccos(np.dot(x1, x2))



def apply_pose(pose, x):
    if x is None:
        return x
    if isinstance(pose, np.ndarray):
        pose = Pose.from_transformation_matrix(pose)
    assert isinstance(pose, Pose)
    if isinstance(x, Pose):
        return pose * x
    elif isinstance(x, np.ndarray):
        return to_nc(to_gc(x, dim=3) @ pose.to_transformation_matrix().T, dim=3)
    else:
        print(x, type(x))
        raise NotImplementedError("")


def check_coplanar(vertices):
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(vertices)
    return pca.singular_values_[-1] < 1e-3

