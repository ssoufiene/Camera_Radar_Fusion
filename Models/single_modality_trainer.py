import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class SingleModalityTrainer:
    def __init__(self, num_classes, modality='image', device=None):
        """
        modality: 'image' or 'radar'
        """
        if modality not in ['image', 'radar']:
            raise ValueError("modality must be 'image' or 'radar'")

        self.modality = modality
        self.num_classes = num_classes
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Setup model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)

        print(f"Initialized SingleModalityTrainer with modality: {modality}")

    def train(self, train_loader, num_epochs=5, lr=1e-4):
        """Train model with selected modality"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print(f"Starting training with {self.modality} modality for {num_epochs} epochs")
        print("=" * 60)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (images, radars, targets) in enumerate(train_loader):
                # Select modality
                if self.modality == 'image':
                    inputs = [img.to(self.device) for img in images]
                elif self.modality == 'radar':
                    inputs = [radar.to(self.device) for radar in radars]

                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = self.model(inputs, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                num_batches += 1

                # Print every 10 batches
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {losses.item():.4f}")

            # Epoch average loss
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
            print("-" * 60)

        print("Training complete!")
        return self.model
