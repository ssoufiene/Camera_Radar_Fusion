import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image 
from Camera_Radar_Fusion.Data.radar_utils import project_cloud_to_image,radar_points_to_map
from Camera_Radar_Fusion.Data.utils import get_2d_boxes



def custom_collate(batch):
    """Collate function that returns all modalities"""
    images = [item['image'] for item in batch]
    radars = [item['radar'] for item in batch]
    targets = [{'boxes': item['boxes'], 'labels': item['labels']} for item in batch]
    return images, radars, targets


class Projection_Dataset(Dataset):
    def __init__(self, nusc, samples, transform=None, image_size=(900, 1600)):
        self.nusc = nusc
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

        # Build class mapping from nuScenes categories
        self.class_names = [cat['name'] for cat in nusc.category]
        self.CLASS_TO_IDX = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]
        sample = self.nusc.get('sample', sample_token)
        out = {}

        # --- Image ---
        image = self._load_image(sample)
        if self.transform:
            image = self.transform(image)
        out['image'] = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # --- Radar ---
        uv,z,id= project_cloud_to_image(self.nusc,sample)
        radar_map = radar_points_to_map(self.nusc,sample,uv,depth=z,valid_idx=id)
        out['radar'] = torch.from_numpy(radar_map).float()

        # --- Boxes & Labels ---
        boxes_2d, _ = get_2d_boxes(sample)
        labels = []

        if len(boxes_2d) > 0:
            boxes_coords = []
            for box_data in boxes_2d:
                x1, y1, x2, y2, full_name = box_data
                boxes_coords.append([x1, y1, x2, y2])
                label_idx = self.CLASS_TO_IDX.get(full_name, 0)
                labels.append(label_idx)

            boxes = torch.from_numpy(np.array(boxes_coords, dtype=np.float32)).float()
            labels = torch.from_numpy(np.array(labels, dtype=np.int64)).long()
        else:
            boxes = torch.zeros((0, 4)).float()
            labels = torch.zeros((0,), dtype=torch.long)

        out['boxes'] = boxes
        out['labels'] = labels

        return out

    def _load_image(self, sample):
        camera_channel = 'CAM_FRONT'
        camera_data = self.nusc.get('sample_data', sample['data'][camera_channel])
        image_path = self.nusc.get_sample_data_path(camera_data['token'])
        return Image.open(image_path).convert('RGB')

