import pytorch_lightning as pl
from net.lightning import TransformerTrainer
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import torch

model = TransformerTrainer.load_from_checkpoint('lightning_logs/version_72/checkpoints/epoch=999.ckpt')

loader = model.train_dataloader()
for batch in loader:
    X1,X2,Y1,Y2 = batch
    for X_,Y_ in zip(X2,Y2):
        X = X_.unsqueeze(0)
        Y = Y_
        if Y.max() > 0.3:
            continue
        inp = X.clone().permute(1,0,2)
        output = model(inp)
        output = output.squeeze().detach().cpu().numpy()
        X = X[0]
        plt.plot(torch.arange(len(X)),X)
        plt.plot(torch.arange(len(Y))+len(X),Y)
        plt.plot(torch.arange(len(output))+len(X),output)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()