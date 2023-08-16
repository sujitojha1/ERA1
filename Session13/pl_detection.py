import torch
import config
from model import YOLOv3
from loss import YoloLoss
from pytorch_lightning import LightningModule
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

loss_fn = YoloLoss()

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
)

class LitYOLOv3(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        self.save_hyperparameters()
        self.lr = config.LEARNING_RATE

    def forward(self, imgs):
        detections = self.model(imgs)
        return detections

    def criterion(out, y):
        y0, y1, y2 = (y[0], y[1], y[2])
        scaled_anchors = scaled_anchors.to(self.device)
        loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )
        return loss

    def training_step(self, batch, batch_id):
        x,y = batch
        out = self(x)

        loss = self.criterion(out,y)

        self.log("training loss", loss)
        return loss

    def on_train_end(self) -> None:
        scaled_anchors = scaled_anchors.to(self.device)

        plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, scaled_anchors)
        #print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(self.ap50_95, self.ap50))

        check_class_accuracy(self.model, self.train_dataloader(), threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.test_dataloader(),
            self.model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        print(f"MAP: {mapval.item()}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            # momentum=0.9,
            # weight_decay=5e-4,
        )
        return {"optimizer": optimizer}
    
    def setup(self, stage=None):
        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
            train_csv_path=config.DATASET + "/train25exp.csv", test_csv_path=config.DATASET + "/test25exp.csv"
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.train_eval_loader

    def test_dataloader(self):
        return self.test_loader





