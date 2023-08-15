import torch
import config
from model import YOLOv3
from loss import YoloLoss
from pytorch_lightning import LightningModule

loss_fn = YoloLoss()

class LitYOLOv3(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsequeeze(1).repeat(1,3,2)
        )
        self.save_hyperparameters()

    def forward(self, imgs):
        detections = self.model(imgs)
        return detections

    def training_step(self, batch, batch_id):
        x,y = batch
        y0, y1, y2 = (y[0], y[1], y[2])

        out = self(x)
        loss = (
            loss_fn(out[0], y0, self.scaled_anchors[0])
            + loss_fn(out[1], y1, self.scaled_anchors[1])
            + loss_fn(out[2], y2, self.scaled_anchors[2])
        )

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.LEARNING_RATE,
            # momentum=0.9,
            # weight_decay=5e-4,
        )
        return {"optimizer": optimizer}





