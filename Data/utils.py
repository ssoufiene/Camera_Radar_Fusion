import os
import subprocess

def load_nuscenes(dataroot="/content/data/sets/nuscenes", version="v1.0-mini", verbose=True):
    """
    Downloads and loads the nuScenes mini dataset if not already available.
    Returns a NuScenes object.
    """
    # Create folder if it doesn’t exist
    os.makedirs(dataroot, exist_ok=True)

    # Check if already downloaded
    archive_path = os.path.join(dataroot, "v1.0-mini.tgz")
    if not os.path.exists(os.path.join(dataroot, "v1.0-mini")):
        print("Downloading nuScenes mini dataset...")
        subprocess.run([
            "wget", "-q", "https://www.nuscenes.org/data/v1.0-mini.tgz", "-O", archive_path
        ])
        subprocess.run(["tar", "-xf", archive_path, "-C", dataroot])
        subprocess.run(["pip", "install", "nuscenes-devkit", "--quiet"])

    print("Loading nuScenes...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
    return nusc



def get_2d_boxes(sample):
    from nuscenes.utils.geometry_utils import view_points
    

    """
    Project 3D bounding boxes to 2D camera image.

    Args:

        sample: Sample from nuScenes dataset

    Returns:
        boxes_2d: List of 2D boxes [x1, y1, x2, y2, label]
        image: PIL Image object
    """
    cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

    # Get 3D boxes, camera intrinsic, and image path
    img_path, boxes, cam_intrinsic = nusc.get_sample_data(cam_data['token'], box_vis_level=1)
    cam_intrinsic = np.array(cam_intrinsic)

    image = Image.open(img_path)
    w, h = image.size
    boxes_2d = []

    for box in boxes:
        corners = view_points(box.corners(), cam_intrinsic, normalize=True)
        min_x, min_y = corners[0].min(), corners[1].min()
        max_x, max_y = corners[0].max(), corners[1].max()

        # Clip to image bounds - ensure valid box dimensions
        min_x = max(0, min_x)
        max_x = min(w, max_x)
        min_y = max(0, min_y)
        max_y = min(h, max_y)

        # Only add boxes that have positive dimensions and are within bounds
        if min_x < max_x and min_y < max_y:
            boxes_2d.append([min_x, min_y, max_x, max_y, box.name])

    return boxes_2d, image
