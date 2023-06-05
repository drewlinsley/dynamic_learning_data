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

def look_at_rotation(camera_position, at=np.array([0., 0., 0.]), up=np.array([0., 1., 0.])):
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

def project_point_to_plane(point, plane_normal, plane_point=np.array([0., 0., 0.])):
    point -= plane_point
    plane_normal_hat = plane_normal / np.linalg.norm(plane_normal)
    point_parallel = np.dot(point, plane_normal_hat) * plane_normal_hat
    point_proj = point - point_parallel
    point_proj += plane_point
    return point_proj

def get_spherical_trajectory(start_pos, end_pos, winding=None, num_steps=50):
    start_t, end_t = start_pos, end_pos
    start_t_r = np.linalg.norm(start_t)
    start_t_hat = start_t / start_t_r
    end_t_r = np.linalg.norm(end_t)
    end_t_hat = end_t / end_t_r

    angle_rad = np.arccos(np.dot(start_t_hat, end_t_hat))

    rot_axis = np.cross(start_t_hat, end_t_hat)
    rot_axis /= np.linalg.norm(rot_axis)

    #TODO: investigate why we don't make use of winding sign
    # Does original cross always give correct orientation?
    if winding is not None:
        winds = np.abs(winding) // (2*np.pi)
        angle_rad += winds * (2*np.pi)

    print(f"angle: {angle_rad*180./np.pi}, rot_axis: {rot_axis}")

    skew = np.array(
        [
            [0.0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0.0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0.0],
        ]
    )

    angles = np.linspace(0., angle_rad, num_steps+1)
    radii = np.linspace(start_t_r, end_t_r, num_steps+1)
    rots =  np.eye(3) + \
            scale_matrix_by_coeffs(skew, np.sin(angles)) + \
            scale_matrix_by_coeffs((skew @ skew), (1. - np.cos(angles)))

    positions_hat = rots @ start_t_hat
    positions_hat /= np.linalg.norm(positions_hat, axis=-1, keepdims=True)
    positions = positions_hat
    positions = np.multiply(positions_hat.T, radii).T
    return positions

def get_linear_trajectory(start_pos, end_pos, num_steps=50):
    start_t, end_t = start_pos, end_pos
    diff = end_t - start_t
    diff_mag = np.linalg.norm(diff)
    diff_hat = diff / diff_mag

    dists = np.linspace(0., diff_mag, num_steps+1)
    positions = scale_matrix_by_coeffs(diff_hat, dists)
    positions += start_t
    return positions

def estimate_winding(positions, up=np.array([0., -1., 0.])):
    # Assumes positions are centered

    unit_pos = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    # Drop last position since roll wraps around
    cross_pos = unit_pos[:-1, :]
    cross_pos_shift = np.roll(unit_pos, shift=-1, axis=0)[:-1, :]
    cross_prods = np.cross(cross_pos, cross_pos_shift)

    sines = np.linalg.norm(cross_prods, axis=1)
    orientations = np.sign(cross_prods @ up)
    sines *= orientations
    theta = np.sum(np.arcsin(sines))

    return theta

def spherical_trajectories(extrinsics):
    # Extrinsics describes camera pose in world frame with OpenCV convention:
    # z+ in, y+ down, x+ right
    all_pos = np.copy(extrinsics[:, :3, 3].reshape(-1, 3))
    mean_pos = np.mean(all_pos, axis=0, keepdims=True)
    all_pos -= mean_pos

    mid_idx = all_pos.shape[0] // 2
    start_t = all_pos[0, :]
    mid_t = all_pos[mid_idx, :]
    end_t = all_pos[-1, :]

    U, D, VT = np.linalg.svd(all_pos)
    V = VT.T
    '''
    print(f"all_pos shape: {all_pos.shape}")
    print(f"shapes: U: {U.shape}, D: {D.shape}, VT: {VT.shape}")
    print(f"D: {D}")
    print(f"0/1: {D[0]/D[1]}, 1/2: {D[1]/D[2]}")
    print(f"V: {V}")
    '''
    ratio_1 = D[0] / D[1]
    ratio_2 = D[1] / D[2]
    # If third dim contributes significantly less info than second dim, assume planar
    # Otherwise, assume linear or spherical and fit linear trajectory
    if 2.*ratio_1 < ratio_2:
        plane_normal = V[:, 2]
        #print(f"plane_normal: {plane_normal}")
        # TODO: pass in plane normal and project points to plane
        signed_theta = estimate_winding(all_pos)
        positions = get_spherical_trajectory(start_t, end_t, winding=signed_theta)
    else:
        positions = get_linear_trajectory(start_t, end_t)
    positions += mean_pos


    mats = []
    for i in range(positions.shape[0]):
        t = positions[i, :]
        #TODO: Figure out why up=[0, -1, 0] isn't correct
        R = look_at_rotation(t)
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = t
        mats.append(E[np.newaxis, ...])

    mats = np.vstack(mats)
    print(f"mats shape: {mats.shape}")
    print(f"*************************************")
    return mats
