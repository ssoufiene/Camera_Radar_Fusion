import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Radar_Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)


class Image_Encoder(nn.Module):
    def __init__(self, out_channels=32, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.proj = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.features(x)
        return self.proj(out)


class Cross_Attention_Fusion(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True)
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)

    def forward(self, query, key_value):
        B, Cq, H, W = query.shape
        query_flat = query.flatten(2).transpose(1, 2)
        key_flat = key_value.flatten(2).transpose(1, 2)

        q = self.q_proj(query_flat)
        k = self.k_proj(key_flat)
        v = self.v_proj(key_flat)

        attn_out, _ = self.attn(q, k, v)
        fused = attn_out.transpose(1, 2).view(B, Cq, H, W)
        return fused


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class RetinaNetHead(nn.Module):
    """RetinaNet detection head with classification and bbox branches"""
    def __init__(self, in_channels=32, num_classes=23, num_layers=4):
        super().__init__()
        self.num_classes = num_classes

        # Classification branch
        self.cls_tower = nn.Sequential()
        for i in range(num_layers):
            self.cls_tower.add_module(f'conv{i}', nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            self.cls_tower.add_module(f'relu{i}', nn.ReLU(inplace=True))

        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

        # Bounding box regression branch
        self.bbox_tower = nn.Sequential()
        for i in range(num_layers):
            self.bbox_tower.add_module(f'conv{i}', nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            self.bbox_tower.add_module(f'relu{i}', nn.ReLU(inplace=True))

        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

    def forward(self, features):
        cls_feat = self.cls_tower(features)
        cls_logits = self.cls_logits(cls_feat)

        bbox_feat = self.bbox_tower(features)
        bbox_preds = self.bbox_pred(bbox_feat)

        return cls_logits, bbox_preds


class Implicit_Fusion(nn.Module):
    def __init__(self, num_classes, radar_in=3, d_model=32, feat_resolution=(900, 1600), device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.d_model = d_model
        self.H_feat, self.W_feat = feat_resolution

        self.radar_encoder = Radar_Encoder(in_channels=radar_in, out_channels=d_model)
        self.image_encoder = Image_Encoder(out_channels=d_model, pretrained=True)
        self.cross_attn = Cross_Attention_Fusion(dim_q=d_model, dim_kv=d_model, num_heads=4)
        self.upsample = nn.Upsample(size=(self.H_feat, self.W_feat), mode='bilinear', align_corners=False)

        # Feature extraction layer
        self.feat_extractor = nn.Sequential(
            nn.Conv2d(d_model, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # RetinaNet detection head
        self.detection_head = RetinaNetHead(in_channels=32, num_classes=num_classes, num_layers=4)

        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

        self.to(self.device)

    def forward(self, radar, image, targets=None):
        radar_feat = self.radar_encoder(radar)
        image_feat = self.image_encoder(image)

        # Cross-attention fusion
        fused_feat = self.cross_attn(image_feat, radar_feat)

        # Upsample to final resolution
        fused_feat_upsampled = self.upsample(fused_feat)

        # Detection head
        cls_logits, bbox_preds = self.detection_head(fused_feat_upsampled)

        if self.training:
            loss_dict = self.compute_losses(cls_logits, bbox_preds, targets)
            return loss_dict
        else:
            return cls_logits, bbox_preds

    def compute_losses(self, cls_logits, bbox_preds, targets):
        B, _, H, W = cls_logits.shape

        # Create target maps from bounding boxes
        cls_targets = torch.zeros(B, H, W, dtype=torch.long, device=cls_logits.device)
        bbox_targets = torch.zeros(B, 4, H, W, dtype=torch.float32, device=bbox_preds.device)

        for b_idx, target in enumerate(targets):
            boxes = target['boxes']
            labels = target['labels']

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                x1_norm = max(0, int(x1 * W / self.W_feat))
                y1_norm = max(0, int(y1 * H / self.H_feat))
                x2_norm = min(W - 1, int(x2 * W / self.W_feat))
                y2_norm = min(H - 1, int(y2 * H / self.H_feat))

                if x2_norm > x1_norm and y2_norm > y1_norm:
                    cls_targets[b_idx, y1_norm:y2_norm, x1_norm:x2_norm] = label
                    # Normalize bbox coordinates to [0, 1]
                    bbox_targets[b_idx, 0, y1_norm:y2_norm, x1_norm:x2_norm] = x1 / self.W_feat
                    bbox_targets[b_idx, 1, y1_norm:y2_norm, x1_norm:x2_norm] = y1 / self.H_feat
                    bbox_targets[b_idx, 2, y1_norm:y2_norm, x1_norm:x2_norm] = x2 / self.W_feat
                    bbox_targets[b_idx, 3, y1_norm:y2_norm, x1_norm:x2_norm] = y2 / self.H_feat

        # Compute losses
        loss_cls = self.focal_loss(cls_logits, cls_targets)
        loss_bbox = self.smooth_l1_loss(bbox_preds, bbox_targets)

        return {'classification': loss_cls, 'bbox_regression': loss_bbox}





def train_model(model, dataloader, num_epochs=5, lr=1e-4):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (images, radars, targets) in enumerate(dataloader):
            radars = torch.stack(radars).to(device)
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(radars, images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

       

    return model