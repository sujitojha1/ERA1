import torch
import config
from model import YOLOv3
from loss import YoloLoss
from pytorch_lightning import LightningModule

loss_fn = YoloLoss()

class LitYOLOv3(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = YOLOv3(num_classess=config.NUM_CLASSES)
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsequeeze(1).repeat(1,3,2)
        )

    def forward(self, imgs):
        detections = self(imgs)
        return detections

    def training_step(self, batch, batch_id):





