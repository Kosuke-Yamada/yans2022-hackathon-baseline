# -*-coding:utf-8-*-

import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from _my_lightning_modules import ReviewDataModule, ReviewRegressionNet


def main(args):
    dm = ReviewDataModule(args)
    net = ReviewRegressionNet(args)
    trainer = pl.Trainer(gpus=[args.gpu], logger=False)

    if args.ckpt_file is None:
        pred = trainer.predict(net, dm)
    else:
        pred = trainer.predict(net, dm, ckpt_path=args.ckpt_file)

    df = pd.read_json(args.input_file, orient="records", lines=True)
    df.loc[:, "pred"] = sum([list(p.numpy().flatten()) for p in pred], [])
    df.loc[:, "pred_helpful_votes"] = df["pred"].apply(lambda x: np.exp(x) - 1)

    output_file = args.output_dir + args.input_file.split("/")[-1]
    df.to_json(output_file, orient="records", force_ascii=False, lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str)

    parser.add_argument(
        "--model_name", type=str, default="cl-tohoku/bert-base-japanese"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
