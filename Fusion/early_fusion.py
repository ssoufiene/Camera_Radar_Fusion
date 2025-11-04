
def fusion_concatenation(images, radars):
    """Concatenate image and radar: (3, H, W) + (3, H, W) = (6, H, W)"""
    return [torch.cat([img, rad], dim=0) for img, rad in zip(images, radars)]


def fusion_addition(images, radars):
    """Add image and radar channels elementwise"""
    fused = [img + rad for img, rad in zip(images, radars)]
    return [f.clamp(0, 1) for f in fused]


def fusion_weighted(images, radars, weight_image=0.7, weight_radar=0.3):
    """Weighted combination: w_img * img + w_rad * rad"""
    fused = [weight_image * img + weight_radar * rad for img, rad in zip(images, radars)]
    return [f.clamp(0, 1) for f in fused]


FUSION_STRATEGIES = {
    'concatenation': fusion_concatenation,
    'addition': fusion_addition,
    'weighted': fusion_weighted
}



class EarlyFusionFasterRCNN(nn.Module):
    """Faster R-CNN with early (input-level) radar–camera fusion"""

    def __init__(self, num_classes, fusion_strategy='concatenation', device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.device = device

        # Load pretrained Faster R-CNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Modify first conv layer if concatenation (6 channels)
        if fusion_strategy == 'concatenation':
            self._modify_backbone_for_concatenation()
            self.model.transform.image_mean = [0.0] * 6
            self.model.transform.image_std = [1.0] * 6

        # Adjust the box predictor for our num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def _modify_backbone_for_concatenation(self):
        """Modify first conv layer to accept 6 input channels."""
        old_conv = self.model.backbone.body.conv1  # ResNet backbone first conv
        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize first 3 channels from pretrained conv1
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            # Use mean of RGB weights for radar channels
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            new_conv.weight[:, 3:] = mean_weight.repeat(1, 3, 1, 1)

        # Replace conv layer
        self.model.backbone.body.conv1 = new_conv

    def forward(self, images, radars, targets=None):
        """
        Args:
            images: list of image tensors (3, H, W)
            radars: list of radar tensors (3, H, W)
            targets: list of target dicts (optional, for training)
        Returns:
            dict (losses) if training, or list (detections) if inference
        """
        fused_inputs = FUSION_STRATEGIES[self.fusion_strategy](images, radars)
        return self.model(fused_inputs, targets)



def train_early_fusion(model, train_loader, val_loader=None, num_epochs=5,
                       lr=1e-4, device='cuda', save_path=None):
    """
    Train early fusion model

    Args:
        model: EarlyFusionFasterRCNN model
        train_loader: Training dataloader
        val_loader: Validation dataloader (optional)
        num_epochs: Number of epochs
        lr: Learning rate
        device: torch device
        save_path: Path to save best model
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"Starting training with fusion strategy: {model.fusion_strategy}")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Training loop
        for batch_idx, (images, radars, targets) in enumerate(train_loader):
            # Move to device
            images = [img.to(device) for img in images]
            radars = [rad.to(device) for rad in radars]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, radars, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {losses.item():.4f}")

        # Calculate average training loss
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")
        print("-" * 60)


    torch.save(model_image.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path}")
    print("Training complete!")
    return model





