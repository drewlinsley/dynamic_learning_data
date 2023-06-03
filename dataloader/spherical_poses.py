import numpy as np
import torch

def trans_vec(x,y,z): return np.array([x, y, z], dtype=np.float32)

def affine_transform(rot_mat, trans_vec):
    affine_mat = np.eye(4)
    affine_mat[:-1, :-1] = rot_mat
    affine_mat[:-1, -1] = trans_vec
    return affine_mat

def inv_affine_transform(rot_mat, trans_vec):
    inv_rot_mat = rot_mat.transpose()
    inv_trans_vec = -(inv_rot_mat @ trans_vec)
    inv_affine_mat = affine_transform(inv_rot_mat, inv_trans_vec)
    return inv_affine_mat

def invert_axes():
    mat = np.eye(4)
    mat[0, 0] = -1
    mat[1, 1] = -1
    return mat

# Pytorch3d convention
def pitch_mat(gamma):
    return np.array([
    [1,0,0],
    [0,np.cos(gamma),-np.sin(gamma)],
    [0,np.sin(gamma),np.cos(gamma)]], dtype=np.float32)

def yaw_mat(beta):
    return np.array([
    [np.cos(beta),0,np.sin(beta)],
    [0,1,0],
    [-np.sin(beta),0,np.cos(beta)],], dtype=np.float32)

def roll_mat(alpha): 
    return np.array([
    [np.cos(alpha), -np.sin(alpha), 0],
    [np.sin(alpha), np.cos(alpha), 0],
    [0,0,1]], dtype=np.float32)

def cartesian_coords_from_spherical(azim_deg, elev_deg, radius):
    # Pytorch3d convention
    # z+ in, y+ up, x+ left
    # pytorch3d measures elev from xz plane instead of from y
    azim = azim_deg * (np.pi / 180.)
    elev = elev_deg * (np.pi / 180.)
    x = radius * np.cos(elev) * np.sin(azim)
    y = radius * np.sin(elev)
    z = radius * np.cos(elev) * np.cos(azim)
    return trans_vec(x, y, z)

def look_at_rotation(camera_position, at=np.array([0., 0., 0.])):
    up = np.array([0., 1., 0.])
    z_axis = at - camera_position
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rot = np.vstack([x_axis, y_axis, z_axis]).transpose()
    return rot

def camera_to_world_transform(azim, elev, radius, torch_output=False):
    # Camera-to-world transform refers to the action when applied to points in camera frame
    # This means the transform itself refers to camera placement in world frame
    # What we do here is construct the point in pytorch3d convention
    # then transform to OpenCV convention.
    # The frame convention here is
    # z+ in, y+ up, x+ left
    camera_position = cartesian_coords_from_spherical(azim, elev, radius)
    rot = look_at_rotation(camera_position)
    c2w = affine_transform(rot, camera_position)
    # The frame convention output follows OpenCV, which is
    # z+ in, y+ down, x+ right
    # Apply a similarity transform to change basis to OpenCV
    P = invert_axes()
    c2w = P.transpose() @ c2w @ P
    
    return torch.from_numpy(c2w) if torch_output else c2w


def spherical_poses(image_to_camera_transform):
    # Example trajectory definition
    # Ideal for rendering teddybear scene 101_11758_21048
    azims = np.linspace(15., 360.-15., 50+1)
    elevs = np.ones_like(azims)*40.
    camera_trajectory = zip(azims, elevs)
    dist = 10.
    return np.stack(
        [
            camera_to_world_transform(az, el, dist) @ image_to_camera_transform
            for az, el in camera_trajectory
        ], 0
    )

def scale_matrix_by_coeffs(matrix, coeffs):
    tile_shape = [len(coeffs)] + [1]*len(matrix.shape)
    matrices = np.tile(matrix, tile_shape)
    res = np.multiply(matrices.T, coeffs).T
    return res

def spherical_trajectories(extrinsics):
    # Extrinsics describes camera pose in world frame with OpenCV convention:
    # z+ in, y+ down, x+ right
    print(f"extrinsics shape: {extrinsics.shape}")
    start_pose, end_pose = extrinsics[0, :, :], extrinsics[-1, :, :]
    start_R, start_t = start_pose[:3, :3], start_pose[:3, 3]
    end_R, end_t = end_pose[:3, :3], end_pose[:3, 3]
    print(f"start_t: {start_t}\nend_t: {end_t}")

    start_t_r = np.linalg.norm(start_t)
    start_t_hat = start_t / start_t_r
    end_t_r = np.linalg.norm(end_t)
    end_t_hat = end_t / end_t_r

    angle_rad = np.arccos(np.dot(start_t_hat, end_t_hat))

    rot_axis = np.cross(start_t_hat, end_t_hat)
    rot_axis /= np.linalg.norm(rot_axis)
    skew = np.array(
        [
            [0.0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0.0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0.0],
        ]
    )

    num_steps = 50
    angles = np.linspace(0., angle_rad, num_steps+1)
    radii = np.linspace(start_t_r, end_t_r, num_steps+1)
    rots =  np.eye(3) + \
            scale_matrix_by_coeffs(skew, angles) + \
            scale_matrix_by_coeffs((skew @ skew), (1. - np.cos(angles)))
    print(f"angle shape: {angles.shape}")
    print(f"rots shape: {rots.shape}")

    positions_hat = rots @ start_t_hat
    positions_hat /= np.linalg.norm(positions_hat, axis=-1, keepdims=True)
    positions = np.multiply(positions_hat.T, radii).T

    mats = []
    for i in range(len(angles)):
        t = positions[i, :]
        R = look_at_rotation(t)
        E = np.eye(4)[np.newaxis, ...]
        E[0, :3, :3] = R
        E[0, :3, 3] = t
        mats.append(E)

    mats = np.vstack(mats)
    print(f"mats shape: {mats.shape}")
    print(f"*************************************")
    return mats
