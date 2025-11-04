import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class MiddleFusionBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Camera backbone (pretrained)
        cam_resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.cam_backbone = nn.Sequential(*list(cam_resnet.children())[:-2])

        # Radar backbone (same structure, random init)
        rad_resnet = torchvision.models.resnet50(weights=None)
        self.rad_backbone = nn.Sequential(*list(rad_resnet.children())[:-2])

        # Fusion conv (after concatenation)
        self.fusion_conv = nn.Conv2d(2048 + 2048, 1024, kernel_size=1)

        self.out_channels = 1024

        # Placeholder for radar data
        self._radar_input = None

    def set_radar(self, radars):
        """Store radar batch before forward()"""
        self._radar_input = radars

    def forward(self, images):
        """
        images: tensor [B,3,H,W]
        uses self._radar_input set externally
        """
        if self._radar_input is None:
            raise ValueError("Radar input not set. Call set_radar(radars) before forward().")

        # Extract features
        img_feats = self.cam_backbone(images)
        rad_feats = self.rad_backbone(self._radar_input)
        if img_feats.shape[-2:] != rad_feats.shape[-2:]:
            rad_feats = F.interpolate(rad_feats, size=img_feats.shape[-2:],
                              mode='bilinear', align_corners=False)

        # Fuse
        fused = torch.cat([img_feats, rad_feats], dim=1)
        fused = self.fusion_conv(fused)

        return {"0": fused}



class MiddleFusionFasterRCNN(nn.Module):
    def __init__(self, num_classes, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = MiddleFusionBackbone()

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

        self.model = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.to(self.device)
        print(f"✅ MiddleFusionFasterRCNN initialized on {self.device}")

    def forward(self, images, radars, targets=None):
        # Move data to device
        images = [img.to(self.device) for img in images]
        radars = [rad.to(self.device) for rad in radars]
        if targets is not None:
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Stack into batches
        images_batch = torch.stack(images)
        radars_batch = torch.stack(radars)

        self.backbone.set_radar(radars_batch)

        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)


def train_middle_fusion(model, train_loader, num_epochs=10, lr=1e-4, device='cuda', save_path=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    print(f"Training Middle Fusion model on {device} for {num_epochs} epochs")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (images, radars, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            radars = [rad.to(device) for rad in radars]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, radars, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            # Progress log
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {losses.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} — Avg Loss: {avg_loss:.4f}")
        print("-" * 60)

    if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved checkpoint → {save_path}")

    print("✅ Training complete.")
    return model
