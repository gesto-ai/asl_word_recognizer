
import os

import pytorch_lightning as pl
import torch
from torchvision import transforms

from config import Config
from datasets import videotransforms
from datasets.nslt_dataset import NSLT as Dataset
from model.lightning_model import InceptionI3dLightning

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


pl.seed_everything()

def run(cfg, root, train_split, save_model, i3d_weights, weights=None):

    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=False)

    lr = cfg.INIT_LR
    weight_decay = cfg.ADAM_WEIGHT_DECAY
    model = InceptionI3dLightning(lr, weight_decay, i3d_weights, num_classes=dataset.num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        model.load_state_dict(torch.load(weights))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_model,
        save_top_k=1,
        monitor="val_total_loss",
        mode='min',
        filename="model-{epoch:02d}-{val_loss:.2f}")

    trainer = pl.Trainer(
        deterministic=True,
        max_steps=cfg.MAX_STEPS,
        max_epochs=400,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=1)
    trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=val_dataloader)

    # save model to .pt format
    checkpoint = torch.load(f"{save_model}/model-epoch=01-val_loss=0.00.ckpt")
    model.load_state_dict(checkpoint['state_dict'])
    torch.save(model.model.state_dict(), f"{save_model}/test_model.pt")


if __name__ == '__main__':
    cfg = Config()

    run(
        cfg=cfg,
        root=cfg.DATA_ROOT_PATH,
        save_model=cfg.SAVE_MODEL_PATH,
        train_split=cfg.TRAIN_SPLIT_FILE,
        i3d_weights=cfg.ID3_PRETRAINED_WEIGHTS_PATH,
    )
