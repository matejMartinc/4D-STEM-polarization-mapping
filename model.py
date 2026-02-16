import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from collections import Counter
from torchvision.models import resnet50, vgg16


@staticmethod
def vector_to_class_idx(vector):
    dy, dx = vector[:, 1], vector[:, 0]
    angle_rad = torch.atan2(dy, dx)
    angle_deg = torch.rad2deg(angle_rad)
    shifted_angle_deg = (angle_deg + 22.5 + 360) % 360
    bins = torch.arange(45, 361, 45, device=vector.device)
    return torch.bucketize(shifted_angle_deg, bins)


@staticmethod
def class_indices_to_vectors(class_indices):
    angles_deg = class_indices.float() * 45.0
    angles_rad = torch.deg2rad(angles_deg)
    dx = torch.cos(angles_rad)
    dy = torch.sin(angles_rad)
    return torch.stack([dx, dy], dim=1)



class ResNet50Encoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=False):
        super().__init__()
        self.resnet = resnet50(pretrained=pretrained)

        # Modify input conv layer to accept 1-channel input instead of RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original classification head (fc layer)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Output: (B, 2048, 1, 1)

        # Add a projection layer to reduce to output_dim
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),  # Helps with stability
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.proj(x)
        return x

class VGG16Encoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=False):
        super().__init__()
        vgg = vgg16(pretrained=pretrained)

        # Modify first conv layer to accept 1-channel input
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # Extract features up to the last pooling layer
        self.features = vgg.features  # Output: (B, 512, 7, 7) for 224x224 input

        # Custom projection head
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (B, 512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x




class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, regression_training=False, contrastive_loss=True, seed=42, backbone=False, encoder_name='custom', output_folder='results'):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.contrastive_loss = contrastive_loss
        self.seed = seed
        self.backbone = backbone
        self.encoder_name = encoder_name
        self.regression_training = regression_training
        self.output_folder = output_folder

        if not self.backbone:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.n_sizes = self._get_conv_output(input_shape)
        else:
            if 'vgg' in self.encoder_name:
                self.encoder = VGG16Encoder(output_dim=256, pretrained=True)
            elif 'res_net' in self.encoder_name:
                self.encoder = ResNet50Encoder(output_dim=256, pretrained=True)
            self.n_sizes = 256


        self.supervised_decoder = nn.Sequential(
            nn.Linear(self.n_sizes, 512),
            nn.ReLU(),
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        if self.regression_training:
            # Head for predicting a 2D direction vector (x,y)
            self.regression_head = nn.Sequential(
                nn.Linear(512, 2),
                nn.Tanh()  # Squeezes output to [-1, 1], good for direction vectors
            )
            self.criterion = nn.MSELoss()
        elif self.contrastive_loss:
            # Standard contrastive head
            self.contrastive_head = nn.Linear(512, 128)
        else:
            # Standard classification head
            self.classification_head = nn.Linear(512, num_classes)
            self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        return output_feat.view(1, -1).size(1)

    def _forward_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        features = self.supervised_decoder(x)
        if self.regression_training:
            # Output the 2D direction vector
            return self.regression_head(features)
        elif self.contrastive_loss:
            return self.contrastive_head(features)
        else:
            return self.classification_head(features)

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        if self.regression_training:
            target_vectors = class_indices_to_vectors(y)
            loss = self.criterion(logits, target_vectors)
            pred_class_indices = vector_to_class_idx(logits)
            acc = self.accuracy(pred_class_indices, y)
            self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        elif self.contrastive_loss:
            loss = self.prototypical_loss_circular(logits, y)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            self.log('train_acc', acc, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        paths, x, y = batch
        logits = self(x)
        if self.regression_training:
            target_vectors = class_indices_to_vectors(y)
            loss = self.criterion(logits, target_vectors)
            pred_class_indices = vector_to_class_idx(logits)
            acc = self.accuracy(pred_class_indices, y)
            for path, p, t in zip(paths, pred_class_indices, y):
                self.validation_step_outputs.append((path, p.item(), t.item()))
            self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        elif self.contrastive_loss:
            loss = self.prototypical_loss_circular(logits, y)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            for path, p, t in zip(paths, preds, y):
                self.validation_step_outputs.append((path, p.item(), t.item()))
            self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        paths, x, y = batch
        logits = self(x)
        if self.regression_training:
            target_vectors = class_indices_to_vectors(y)
            pred_class_indices = vector_to_class_idx(logits)
            loss = self.criterion(logits, target_vectors)
            acc = self.accuracy(pred_class_indices, y)
            for path, p, t in zip(paths, pred_class_indices, y):
                self.test_step_outputs.append((path, p.item(), t.item()))
            self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        elif self.contrastive_loss:
            loss = self.prototypical_loss_circular(logits, y)
        else:
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            for path, p, t in zip(paths, preds, y):
                self.test_step_outputs.append((path, p.item(), t.item()))
            self.log('test_acc', acc, prog_bar=True)
        self.log('test_loss', loss, prog_bar=True)
        return loss


    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            from sklearn.metrics import accuracy_score
            all_preds, all_true = zip(*[(p, t) for _, p, t in self.validation_step_outputs])
            acc = accuracy_score(all_true, all_preds)
            self.log("Val_Accuracy_all", acc)
            self.validation_step_outputs = []

    def on_test_epoch_end(self):
        if self.test_step_outputs:
            from collections import Counter
            from sklearn.metrics import accuracy_score
            all_preds, all_true = zip(*[(p, t) for _, p, t in self.test_step_outputs])
            print("Test counter", Counter(all_preds))
            acc = accuracy_score(all_true, all_preds)
            self.log("Test_Accuracy_Final", acc)
            df = pd.DataFrame(self.test_step_outputs, columns=['path', 'pred', 'true'])
            df.to_csv(self.output_folder + "/" + self.encoder_name + '_' + str(self.seed) + "_test_results.tsv", index=False, sep='\t')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

    def prototypical_loss_circular(self, x, y, num_classes=8, sigma=1.25):

        device = x.device
        prototypes = []
        for cls in range(num_classes):
            class_mask = (y == cls)
            if class_mask.sum() > 0:
                prototype = x[class_mask].mean(dim=0)
            else:
                # Fallback if a class is missing in the batch to avoid NaN
                prototype = torch.zeros(x.shape[1], device=device)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)  # (C, D)
        dists = torch.cdist(x, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        all_classes = torch.arange(num_classes, device=device).float()
        y_unsqueezed = y.unsqueeze(1).float()
        abs_diff = torch.abs(all_classes - y_unsqueezed)
        circular_dist = torch.min(abs_diff, num_classes - abs_diff)
        weights = torch.exp(-(circular_dist ** 2) / (2 * sigma ** 2))

        # Normalize weights so they sum to 1 (create a valid probability distribution)
        target_probs = weights / weights.sum(dim=1, keepdim=True)

        # Compute Loss using KL Divergence
        loss = F.kl_div(log_p_y, target_probs, reduction='batchmean')
        return loss



