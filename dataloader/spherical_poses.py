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
    azims = np.linspace(180.-45., 180.+45., 40+1)
    elevs = np.sin(np.linspace(0., 2*np.pi, 40+1))*30.
    camera_trajectory = zip(azims, elevs)
    dist = 15.
    return np.stack(
        [
            camera_to_world_transform(az, el, dist) @ image_to_camera_transform
            for az, el in camera_trajectory
        ], 0
    )
