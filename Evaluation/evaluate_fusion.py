import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1, y1 = np.maximum(box1[:2], box2[:2])
    x2, y2 = np.minimum(box1[2:], box2[2:])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (
        (box1[2] - box1[0]) * (box1[3] - box1[1])
        + (box2[2] - box2[0]) * (box2[3] - box2[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def evaluate_fusion(model, eval_loader, device, class_names, iou_threshold=0.5, num_viz=5):
    """
    Evaluate an early-fusion radar+camera model.

    Args:
        model: EarlyFusionFasterRCNN
        eval_loader: DataLoader returning (images, radars, targets)
        device: torch device
        class_names: list of class names
        iou_threshold: IoU threshold for TP
        num_viz: number of samples to visualize
    """
    model.eval()
    preds_by_class, gts_by_class = defaultdict(list), defaultdict(list)
    all_data = []

    with torch.no_grad():
        for img_idx, (images, radars, targets) in enumerate(eval_loader):
            # Move to device
            images = [img.to(device) for img in images]
            radars = [rad.to(device) for rad in radars]

            # Forward pass (early fusion requires both images and radars)
            outputs = model(images, radars)

            # Loop over batch
            for batch_idx, (output, target, image) in enumerate(zip(outputs, targets, images)):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                img_array = image.permute(1, 2, 0).cpu().numpy()

                # Collect ground truth and predictions per class
                for b, l in zip(gt_boxes, gt_labels):
                    gts_by_class[int(l)].append((b, img_idx * len(images) + batch_idx))
                for b, l, s in zip(pred_boxes, pred_labels, pred_scores):
                    preds_by_class[int(l)].append((float(s), b, img_idx * len(images) + batch_idx))

                # Store for visualization
                all_data.append({
                    'image': img_array,
                    'gt_boxes': gt_boxes,
                    'gt_labels': gt_labels,
                    'pred_boxes': pred_boxes,
                    'pred_labels': pred_labels,
                    'pred_scores': pred_scores,
                    'img_id': img_idx * len(images) + batch_idx
                })

    # Compute TP/FP for PR curve
    y_true_all, y_score_all = [], []

    for cls, gts in gts_by_class.items():
        preds = sorted(preds_by_class.get(cls, []), key=lambda x: x[0], reverse=True)
        matched = defaultdict(set)

        for score, pbox, img_id in preds:
            matches = [
                (i, compute_iou(pbox, gt_box))
                for i, (gt_box, gt_img) in enumerate(gts)
                if gt_img == img_id and i not in matched[img_id]
            ]

            if matches:
                best_i, best_iou = max(matches, key=lambda x: x[1])
                if best_iou >= iou_threshold:
                    y_true_all.append(1)
                    matched[img_id].add(best_i)
                else:
                    y_true_all.append(0)
            else:
                y_true_all.append(0)
            y_score_all.append(score)

    # Compute PR curve & AP
    precision, recall, _ = precision_recall_curve(y_true_all, y_score_all)
    ap = average_precision_score(y_true_all, y_score_all)

    # Plot PR curve
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="b", lw=2, label=f"mAP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Overall Precision–Recall Curve")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    print(f"Overall mean Average Precision (mAP): {ap:.4f}")

    # Visualize predictions vs ground truth
    visualize_predictions(all_data, class_names, num_viz=num_viz)

    return ap


def visualize_predictions(all_data, class_names, num_viz=5):
    """Visualize predicted boxes vs ground truth."""

    num_samples = min(num_viz, len(all_data))
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 5 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        data = all_data[idx]
        img = data['image']

        # Ground truth
        ax_gt = axes[idx, 0]
        ax_gt.imshow(img)
        for box in data['gt_boxes']:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax_gt.add_patch(rect)
        ax_gt.axis('off')
        ax_gt.set_title(f'Ground Truth (Sample {idx + 1}) - {len(data["gt_boxes"])} boxes', fontsize=12, fontweight='bold')

        # Predictions
        ax_pred = axes[idx, 1]
        ax_pred.imshow(img)
        for box, score in zip(data['pred_boxes'], data['pred_scores']):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax_pred.add_patch(rect)
        ax_pred.axis('off')
        ax_pred.set_title(f'Predictions (Sample {idx + 1}) - {len(data["pred_boxes"])} boxes', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
