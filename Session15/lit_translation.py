from lightning import LightningModule

class TransformerPredictor(LightningModule):
    def __init__(
            self,

            ):
        
        super().__init__()
        self.save_hyperparameter()
        self.model = 