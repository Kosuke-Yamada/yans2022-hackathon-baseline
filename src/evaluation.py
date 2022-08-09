# -*-coding:utf-8-*-

import argparse
import json

import pandas as pd
from sklearn.metrics import ndcg_score


def convert_to_submit_format(df, score_column, mode="pred"):
    output_list = []
    for product_idx in sorted(set(df["product_idx"])):
        df_product = df[df["product_idx"] == product_idx]
        scores = [
            {"review_idx": i, mode + "_score": s}
            for i, s in zip(df_product["review_idx"], df_product[score_column])
        ]
        output_list.append({"product_idx": product_idx, mode + "_list": scores})
    return pd.DataFrame(output_list)

def calc_ndcg(df_true, df_pred, k=5):
    df = pd.merge(df_true, df_pred, on="product_idx")
    sum_ndcg = 0
    for df_dict in df.to_dict("records"):
        df_eval = pd.merge(
            pd.DataFrame(df_dict["pred_list"]),
            pd.DataFrame(df_dict["true_list"]),
            on="review_idx",
        )
        ndcg = ndcg_score([df_eval["true_score"]], [df_eval["pred_score"]], k=k)
        sum_ndcg += ndcg
    return {"ndcg@5": sum_ndcg / len(df)}


def main(args):
    df = pd.read_json(args.input_file, orient="records", lines=True)

    df_pred = convert_to_submit_format(df, "pred_helpful_votes", "pred")
    output_pred_file = args.output_dir + "submit_" + args.input_file.split("/")[-1]
    df_pred.to_json(output_pred_file, orient="records", force_ascii=False, lines=True)

    if "helpful_votes" in df.columns:
        df_true = convert_to_submit_format(df, "helpful_votes", "true")
        ndcg_score = calc_ndcg(df_true, df_pred, 5)

        output_eval_file = (
            args.output_dir
            + "eval_"
            + args.input_file.split("/")[-1].replace(".jsonl", ".json")
        )
        with open(output_eval_file, "w") as f:
            json.dump(ndcg_score, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
