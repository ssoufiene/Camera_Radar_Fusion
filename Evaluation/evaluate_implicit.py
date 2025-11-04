import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

def evaluate_implicit(model, dataloader, device):
    from sklearn.metrics import f1_score
    import torch.nn.functional as F

    model.eval()
    f1_scores = []

    with torch.no_grad():
        for images, radars, targets in dataloader:
            images = torch.stack(images).to(device)
            radars = torch.stack(radars).to(device)

            cls_logits, _ = model(radars, images)
            # Convert logits to predicted class per pixel
            y_pred = cls_logits.argmax(dim=1)  # shape: (B, H, W)

            for b_idx, target in enumerate(targets):
                # Create per-pixel target map at the same resolution as cls_logits
                H, W = y_pred.shape[1:]
                cls_target_map = torch.zeros((H, W), dtype=torch.int64, device=device)

                boxes = target['boxes']
                labels = target['labels']
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    # map box coordinates to prediction resolution
                    x1 = int(x1 / model.W_feat * W)
                    x2 = int(x2 / model.W_feat * W)
                    y1 = int(y1 / model.H_feat * H)
                    y2 = int(y2 / model.H_feat * H)
                    cls_target_map[y1:y2, x1:x2] = label

                y_true_flat = cls_target_map.cpu().numpy().flatten()
                y_pred_flat = y_pred[b_idx].cpu().numpy().flatten()
                f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
                f1_scores.append(f1)

    model.train()
    return sum(f1_scores) / len(f1_scores)
