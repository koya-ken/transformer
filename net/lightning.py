import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from .models import TransAm
from data.sin_dataset import SinDataset

class TransformerTrainer(pl.LightningModule):
    def __init__(self,seq_len=250,feature_size=1,output_seq_len=250, latant_dim=1, num_layers=3, dropout=0):
        super().__init__()
        self.model = TransAm(feature_size=feature_size,input_seq_len=seq_len,output_seq_len=output_seq_len, output_feature_size=latant_dim, num_layers=num_layers, dropout=dropout)
        self.input_len = seq_len
        self.latent_dim = latant_dim
        self.output_len = output_seq_len
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X1,X2,Y1,Y2 = batch
        # B,S,E => S,B,E
        # Batch Sequence Embedding
        X2 = X2.permute(1,0,2)
        output = self.model(X2)
        loss = F.mse_loss(output, Y2)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(SinDataset(self.input_len,self.output_len,augumentation=False),shuffle=True, batch_size=32)
