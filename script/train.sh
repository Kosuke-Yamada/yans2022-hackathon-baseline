mkdir -p ./data/train/model/
mkdir -p ./data/train/csv/
mkdir -p ./data/train/mlruns/

python ./src/train.py \
    --input_file ./data/preprocessing_shared/training.jsonl \
    --output_model_dir ./data/train/model/ \
    --output_csv_dir ./data/train/csv/ \
    --output_mlruns_dir ./data/train/mlruns/ \
    --experiment_name predict_helpful_votes \
    --run_name cl-tohoku_bert-base-japanese_lr1e-5 \
    --model_name cl-tohoku/bert-base-japanese \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_epochs 1 \
    --gpus 0
