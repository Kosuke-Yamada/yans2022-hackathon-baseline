# -*-coding:utf-8-*-

import argparse
import random

import pandas as pd


def decide_sets(df, n_train, n_val, random_state):
    df_product = df.groupby("product_idx").count()
    product_idx_list = sorted(set(df_product.index))

    random.seed(random_state)
    random.shuffle(product_idx_list)

    val_list = product_idx_list[:n_val]
    train_list = (
        product_idx_list[n_val:]
        if n_train is None
        else product_idx_list[n_val : n_val + n_train]
    )

    sets_mapping = {}
    sets_mapping.update({i: "training-train" for i in train_list})
    sets_mapping.update({i: "training-val" for i in val_list})
    df["sets"] = df["product_idx"].map(sets_mapping)
    df["sets"] = df["sets"].fillna("disuse")
    return df


def main(args):
    df = pd.read_json(args.input_file, orient="records", lines=True)
    df_sets = decide_sets(df, args.n_train, args.n_val, args.random_state)

    df_sets.to_json(
        args.output_dir + "training.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    df_tr = df_sets[df_sets["sets"].str.contains("-train")]
    df_tr.to_json(
        args.output_dir + "training-train.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )

    df_val = df_sets[df_sets["sets"].str.contains("-val")]
    df_val.to_json(
        args.output_dir + "training-val.jsonl",
        orient="records",
        force_ascii=False,
        lines=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()
    main(args)
