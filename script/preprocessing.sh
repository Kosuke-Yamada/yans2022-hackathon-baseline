mkdir -p ./data/preprocessing_shared/

python ./src/preprocessing.py \
    --input_file ./data/dataset_shared_initial/training.jsonl \
    --output_dir ./data/preprocessing_shared/ \
    --n_train 10 \
    --n_val 10 \
    --random_state 0
