import pytorch_lightning as pl
from net.lightning import TransformerTrainer
from argparse import ArgumentParser

def main(args):
    model = TransformerTrainer()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)