import torch
import torch.nn as nn
import torch.optim as optim

import loader
import models
from config.config import GPU_NUMBER
from config.hparams import LSTM_DROPOUT, RANDOM_SEED
from config.storage import LOG_DIR, MODEL_DIR
from logger import Logger
from trainer import Trainer
from utils import helper


def train(
    n_epochs,
    model,
    device,
    model_name=None,
    stdout=True,
    log_path=None,
    log_steps=1,
    random_state=RANDOM_SEED,
):
    train_loader, val_loader = loader.load_data(random_seed=random_state)

    # model
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()

    logger = Logger(stdout=stdout, fname=log_path)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        logger=logger,
        log_steps=log_steps,
        model_name=model_name,
    )

    train_losses, val_losses = trainer.fit(train_loader, val_loader, n_epochs)

    return trainer


def main(args):
    if args.model == "T-ED":
        model = models.T_ED(lstm_dropout=args.lstm_dropout)
    elif args.model == "ST-ED":
        model = models.ST_ED(lstm_dropout=args.lstm_dropout)
    elif args.model == "TE-ED":
        model = models.TE_ED(lstm_dropout=args.lstm_dropout)
    elif args.model == "STE-ED":
        model = models.STE_ED(
            lstm_dropout=args.lstm_dropout, include_search=False
        )
    elif args.model == "STE-ED-S":
        model = models.STE_ED(
            lstm_dropout=args.lstm_dropout, include_search=True
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stdout = not args.background
    if args.model_name:
        if args.log_path is None:
            args.log_path = f"{LOG_DIR}/{args.model_name}.log"

        args.model_name = f"{MODEL_DIR}/{args.model_name}"

    print(
        "=" * 30,
        f"Training {args.model} for {args.epoch} epochs on {device} (seed: {args.random_state})",
        "=" * 30,
    )

    trainer = train(
        args.epoch,
        model,
        device,
        model_name=args.model_name,
        stdout=stdout,
        log_path=args.log_path,
        log_steps=args.log_steps,
        random_state=args.random_state,
    )

    print("=" * 40, f"best loss: {trainer.best_loss:.3f}", "=" * 40)
    if args.log_path:
        print(f"training log have been saved to {args.log_path}")
    if args.model_name:
        print(
            f"model parameters have been saved to {args.model_name}_best.pth"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="training model",
        choices=["T-ED", "ST-ED", "TE-ED", "STE-ED", "STE-ED-S"],
    )
    parser.add_argument(
        "-e", "--epoch", help="number of epochs", type=int, default=100
    )
    parser.add_argument(
        "-d",
        "--device",
        help="device on which training and inference are run",
        default="cuda",
        choices=["cpu", "cuda", *[f"cuda:{i}" for i in range(GPU_NUMBER)]],
    )
    parser.add_argument(
        "--model_name", help="file name of saved model's parameter"
    )
    parser.add_argument("--log_path", help="file name of model training log")
    parser.add_argument(
        "--log_steps",
        help="step size between epochs to output training logs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--background",
        help="whether to output log to file only, or to stdout as well",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--random_state",
        help="seed of random number generator",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--lstm_dropout",
        help="dropout ratio between LSTM",
        type=float,
        default=LSTM_DROPOUT,
    )

    args = parser.parse_args()

    helper.fix_seed(args.random_state)
    main(args)
