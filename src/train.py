# -*-coding:utf-8-*-

import argparse

import pytorch_lightning as pl

from _my_lightning_modules import ReviewDataModule, ReviewRegressionNet


def main(args):
    dm = ReviewDataModule(args)
    net = ReviewRegressionNet(args)

    output_model_dir = args.output_model_dir + args.experiment_name + "/"

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_rmse", patience=3, mode="min"),
            pl.callbacks.ModelCheckpoint(
                dirpath=output_model_dir,
                filename=args.run_name,
                verbose=True,
                monitor="val_rmse",
                mode="min",
                save_top_k=1,
            ),
        ],
        logger=[
            pl.loggers.csv_logs.CSVLogger(
                save_dir=args.output_csv_dir,
                name=args.experiment_name,
                version=args.run_name,
            ),
            pl.loggers.mlflow.MLFlowLogger(
                tracking_uri=args.output_mlruns_dir,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
            ),
        ],
    )
    trainer.fit(net, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_model_dir", type=str, required=True)
    parser.add_argument("--output_csv_dir", type=str, required=True)
    parser.add_argument("--output_mlruns_dir", type=str, required=True)

    parser.add_argument("--experiment_name", type=str, default="predict_helpful_votes")
    parser.add_argument(
        "--run_name", type=str, default="cl-thooku_bert-base-japanese_lr1e-5"
    )

    parser.add_argument(
        "--model_name", type=str, default="cl-tohoku/bert-base-japanese"
    )

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    main(args)
