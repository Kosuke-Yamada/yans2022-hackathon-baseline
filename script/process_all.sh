#! データ解凍
tar Jxfv ./data/dataset_shared_initial.tar.xz -C ./data/
# tar Jxfv ./data/dataset_shared.tar.xz -C ./data/

#! 前処理 (小規模な学習セットと開発セット)
mkdir -p ./data/preprocessing_shared/

python ./src/preprocessing.py \
    --input_file ./data/dataset_shared_initial/training.jsonl \
    --output_dir ./data/preprocessing_shared/ \
    --n_train 10 \
    --n_val 10 \
    --random_state 0

#! 学習 (エポック数1)
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

#! 推論 (開発セットとリーダーボード)
INPUT_FILE=./data/preprocessing_shared/training-val.jsonl

EXPERIMENT_NAME=predict_helpful_votes
RUN_NAME=cl-tohoku_bert-base-japanese_lr1e-5

OUTPUT_DIR=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/
mkdir -p ${OUTPUT_DIR}

CKPT_FILE=./data/train/model/${EXPERIMENT_NAME}/${RUN_NAME}.ckpt

python ./src/predict.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt_file ${CKPT_FILE} \
    --model_name cl-tohoku/bert-base-japanese \
    --batch_size 16 \
    --gpu 0

INPUT_FILE=./data/dataset_shared_initial/leader_board.jsonl

python ./src/predict.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --ckpt_file ${CKPT_FILE} \
    --model_name cl-tohoku/bert-base-japanese \
    --batch_size 32 \
    --gpu 0

#! 評価 (開発セットとリーダーボード)
EXPERIMENT_NAME=predict_helpful_votes
RUN_NAME=cl-tohoku_bert-base-japanese_lr1e-5

INPUT_FILE=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/training-val.jsonl

OUTPUT_DIR=./data/evaluation/${EXPERIMENT_NAME}/${RUN_NAME}/
mkdir -p ${EVAL_DIR}

python ./src/evaluation.py \
    --input_file ${INPUT_FILE} \
    --eval_dir ${OUTPUT_DIR}

INPUT_FILE=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/leader_board.jsonl

python ./src/evaluation.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUPTUT_DIR}
