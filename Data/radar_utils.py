from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
from nuscenes.nuscenes import NuScenes




def project_cloud_to_image(nusc,sample):
    """
    Project radar points to camera image plane.

    Args:
        sample: Sample from nuScenes dataset

    Returns:
        uv: np.ndarray of shape (2, N) - projected 2D coordinates
    """
    radar_channel = 'RADAR_FRONT'
    camera_channel = 'CAM_FRONT'
    radar_data = nusc.get('sample_data', sample['data'][radar_channel])
    camera_data = nusc.get('sample_data', sample['data'][camera_channel])

    # --- Load radar point cloud ---
    radar_pc = RadarPointCloud.from_file(nusc.get_sample_data_path(radar_data['token']))

    # --- Get transformations ---
    radar_calib = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    cam_calib = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])

    # Create the extrinsic matrix: with transformation matrices, rightmost applies first so we do Radar--> vehicle --> camera
    T_radar_to_ego = transform_matrix(radar_calib['translation'], Quaternion(radar_calib['rotation']), inverse=False)
    T_ego_to_cam = transform_matrix(cam_calib['translation'], Quaternion(cam_calib['rotation']), inverse=True)
    T_radar_to_cam = T_ego_to_cam @ T_radar_to_ego

    radar_pc_hom = np.vstack((radar_pc.points[:3, :], np.ones(radar_pc.points.shape[1])))  # Create [x,y,z,1]T
    pc_cam = T_radar_to_cam @ radar_pc_hom  # Apply transformation

    # Project to image plane
    K = np.array(cam_calib['camera_intrinsic'])
    uv = K @ (pc_cam[:3, :])

    # Normalize and filter points
    z = pc_cam[2, :]
    u = uv[0, :] / z
    v = uv[1, :] / z

    # Filter points that are in front of camera and within image bounds
    mask = (z > 0) & (u >= 0) & (v >= 0) & (u < 1600) & (v < 900)
    u_filtered = u[mask]
    v_filtered = v[mask]
    z=z[mask]
    valid_idx = np.where(mask)[0]


    # Return as single array of shape (2, N) where N is number of valid points
    uv = np.array([u_filtered, v_filtered])

    return uv,z,valid_idx

def radar_points_to_map(nusc, sample, uv, depth, valid_idx, radar_channel='RADAR_FRONT', H=900, W=1600,
                        max_depth=50.0, max_speed=10.0):
    radar_map = np.zeros((3, H, W), dtype=np.float32)

    radar_data = nusc.get('sample_data', sample['data'][radar_channel])
    radar_path = nusc.get_sample_data_path(radar_data['token'])
    radar_pc = RadarPointCloud.from_file(radar_path)

    # Only use valid points
    vx_comp = radar_pc.points[8, valid_idx]
    vy_comp = radar_pc.points[9, valid_idx]

    for i in range(uv.shape[1]):
        u, v = int(uv[0, i]), int(uv[1, i])
        radar_map[0, v, u] = np.clip(depth[i]/max_depth, 0, 1)
        radar_map[1, v, u] = np.clip((vx_comp[i]/max_speed + 1)/2, 0, 1)
        radar_map[2, v, u] = np.clip((vy_comp[i]/max_speed + 1)/2, 0, 1)

    return radar_map



def radar_to_bev(sample, nusc, radar_channel='RADAR_FRONT', I_h=256, I_w=128, s=0.5):
    """
    Convert radar point cloud to a BEV representation.

    Args:
        sample: nuScenes sample dictionary
        nusc: nuScenes dataset object
        radar_channel: which radar sensor to use
        I_h: BEV grid height (rows)
        I_w: BEV grid width (columns)
        s: cell size in meters

    Returns:
        grid: np.ndarray of shape (I_h, I_w, 3) containing
              [RCS, vx, vy] in each cell
    """

    # Load radar point cloud
    radar_data = nusc.get('sample_data', sample['data'][radar_channel])
    radar_pc = RadarPointCloud.from_file(nusc.get_sample_data_path(radar_data['token']))

    # Radar to ego transformation
    radar_calib = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
    T_radar_to_ego = transform_matrix(radar_calib['translation'], Quaternion(radar_calib['rotation']), inverse=False)

    # Positions in ego frame
    radar_pc_hom = np.vstack((radar_pc.points[:3, :], np.ones(radar_pc.points.shape[1])))
    radar_ego = T_radar_to_ego @ radar_pc_hom
    x, y = radar_ego[0, :], radar_ego[1, :]


    # BEV grid indices
    i = (I_h / 2 - x / s).astype(int)
    j = (I_w / 2 + y / s).astype(int)



    # Filter points outside the grid
    mask = (i >= 0) & (i < I_h) & (j >= 0) & (j < I_w)
    i, j = i[mask], j[mask]

    # Extract radar features (keep in radar frame)
    vx_comp = radar_pc.points[8, :][mask]
    vy_comp = radar_pc.points[9, :][mask]
    rcs = radar_pc.points[5, :][mask]


   # Normalize each channel to [0,1] based on known ranges
    rcs_min, rcs_max = -4, 45
    vx_min, vx_max = -10, 10
    vy_min, vy_max = -10, 10

    rcs = np.clip((rcs - rcs_min) / (rcs_max - rcs_min), 0, 1)
    vx_comp = np.clip((vx_comp - vx_min) / (vx_max - vx_min), 0, 1)
    vy_comp = np.clip((vy_comp - vy_min) / (vy_max - vy_min), 0, 1)
    # Initialize BEV grid
    grid = np.zeros((I_h, I_w, 3), dtype=np.float32)

    # Map points to grid
    grid[i, j, 0] = rcs
    grid[i, j, 1] = vx_comp
    grid[i, j, 2] = vy_comp

    return grid

def radar_to_polar_map(sample, nusc, radar_channel='RADAR_FRONT',chan='RADAR_FRONT',
                                          ref_chan='RADAR_FRONT',
                                          I_h=256, I_w=256, max_range=80,
                                          azimuth_fov=np.deg2rad(120),
                                          nsweeps=5):
    """
    Aggregate multiple radar sweeps and convert to a range–azimuth map
    using RadarPointCloud only.
    """
    # --- Aggregate multiple sweeps in ego frame ---
    radar_pc, times = RadarPointCloud.from_file_multisweep(nusc, sample, ref_chan=ref_chan,chan=chan, nsweeps=nsweeps)

    x, y = radar_pc.points[0, :], radar_pc.points[1, :]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Filter points within FOV and range
    mask = (r < max_range) & (np.abs(theta) < azimuth_fov / 2)
    r, theta = r[mask], theta[mask]

    # Quantize to grid
    r_bin = np.clip((r / max_range * I_h).astype(int), 0, I_h - 1)
    t_bin = np.clip(((theta + azimuth_fov / 2) / azimuth_fov * I_w).astype(int), 0, I_w - 1)

    rcs = radar_pc.points[5, mask]
    vx_comp = radar_pc.points[8, mask]
    vy_comp = radar_pc.points[9, mask]

    radar_map = np.zeros((I_h, I_w, 3), dtype=np.float32)

    # Populate map, max for multiple points per bin
    for i, j, rcs_val, vx_val, vy_val in zip(r_bin, t_bin, rcs, vx_comp, vy_comp):
        radar_map[i, j, 0] = max(radar_map[i, j, 0], np.clip((rcs_val + 4) / 49, 0, 1))
        radar_map[i, j, 1] = max(radar_map[i, j, 1], np.clip((vx_val + 10) / 20, 0, 1))
        radar_map[i, j, 2] = max(radar_map[i, j, 2], np.clip((vy_val + 10) / 20, 0, 1))

    return radar_map
