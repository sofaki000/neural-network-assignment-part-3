import pytorch_lightning
from torch import optim
import torch.nn.functional as F
from VisionTransformer.cifar10Utilities import get_cifar_loaders
import pytorch_lightning as pl
import os
from VisionTransformer.tutorial.model import VisionTransformerModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
train_loader, val_loader, test_loader = get_cifar_loaders()
CHECKPOINT_PATH= ".."


class VisionTransformer(pytorch_lightning.LightningModule):
    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformerModel(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

def train_model(**kwargs):
        print("training model...")
        max_epochs = 100
        trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "../ViT"),
                             accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                             devices=1,
                             max_epochs=max_epochs,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                        LearningRateMonitor("epoch")])
        trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(CHECKPOINT_PATH, "ViT.ckpt")
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            model = VisionTransformer.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
        else:
            pl.seed_everything(42) # To be reproducable
            model = VisionTransformer(**kwargs)
            trainer.fit(model, train_loader, val_loader)
            model = VisionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

        # Test best model on validation and test set
        val_result = trainer.test(model, val_loader, verbose=False)
        test_result = trainer.test(model, test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
        return model, result

if __name__ == '__main__':

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        model, results = train_model(model_kwargs={  'embed_dim': 256,  'hidden_dim': 512, 'num_heads': 8, 'num_layers': 6, 'patch_size': 4,
                                        'num_channels': 3, 'num_patches': 64, 'num_classes': 10, 'dropout': 0.2 },  lr=3e-4)


        print("ViT results", results)