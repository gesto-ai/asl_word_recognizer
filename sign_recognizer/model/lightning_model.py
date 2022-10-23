import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from sign_recognizer.model.inception3d import InceptionI3d


class InceptionI3dLightning(pl.LightningModule):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """


    def __init__(self, lr, weight_decay, inception_weights, num_classes=400):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
        """
        super().__init__()
        self.model = InceptionI3d()
        self.model.load_state_dict(torch.load(inception_weights))
        self.model.replace_logits(num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()



    def forward(self, x):
        logits = self.model(x)
        return logits


    @staticmethod
    def __localization_loss(logits, labels):
        # compute localization loss
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss


    @staticmethod
    def __classification_loss(logits, labels):
        pred = torch.max(logits, dim=2)[0]
        true = torch.max(labels, dim=2)[0]

        # compute classification loss (with max-pooling along time B x C x T)
        loss = F.binary_cross_entropy_with_logits(pred, true)
        
        return loss

    def loss(self, logits, labels):
        loc_loss = self.__localization_loss(logits, labels)
        cls_loss = self.__classification_loss(logits, labels)
        loss = (0.5 * loc_loss + 0.5 * cls_loss)

        return {"loc_loss": loc_loss, "cls_loss": cls_loss, "total_loss": loss}


    def training_step(self, batch, batch_idx) -> dict:
        inputs, labels, _ = batch

        t = inputs.size(2)

        per_frame_logits = self.forward(inputs)
        
        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')

        # get loss
        loss_dict = self.loss(per_frame_logits, labels)
        self.log_dict(loss_dict, on_step = True, logger = False)

        pred = torch.argmax(per_frame_logits, dim=2)[0]
        true = torch.argmax(labels, dim=2)[0]
        batch_value = self.train_acc(pred, true)
        self.log('train_acc_step', batch_value, prog_bar=True)

        return loss_dict['total_loss']


    def validation_step(self, batch, batch_idx) -> dict:
        inputs, labels, _ = batch

        factor = inputs.size(2)
        per_frame_logits = self.forward(inputs)
        
        # upsample to input size
        per_frame_logits = F.interpolate(
            per_frame_logits, factor, mode='linear')

        # get loss
        loss_dict = self.loss(per_frame_logits, labels)
        loss_dict = {f'val_{k}':v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, on_step = True, logger=False)

        pred = torch.argmax(per_frame_logits, dim=2)[0]
        true = torch.argmax(labels, dim=2)[0]
        self.valid_acc.update(pred, true)

        return loss_dict['val_total_loss']


    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(f'Epoch {self.current_epoch}: ', end='')
        print(f'loc_loss={metrics["loc_loss"]}, ', end='')
        print(f'cls_loss={metrics["cls_loss"]}, ', end='')
        print(f'total_loss={metrics["total_loss"]}')

        self.log('train_acc_epoch', self.train_acc.compute())
        print(f'train acc={self.train_acc.compute()}')

        self.train_acc.reset()

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(f'Epoch {self.current_epoch}: ', end='')
        print(f'val_loc_loss={metrics["val_loc_loss"]}, ', end='')
        print(f'val_cls_loss={metrics["val_cls_loss"]}, ', end='')
        print(f'val_total_loss={metrics["val_total_loss"]}')

        self.log('valid_acc_epoch', self.valid_acc.compute())
        print(f'Validation acc={self.valid_acc.compute()}')

        self.valid_acc.reset()


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_total_loss"}
