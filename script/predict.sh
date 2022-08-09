INPUT_FILE=./data/preprocessing_shared/training-val.jsonl
# INPUT_FILE=./data/dataset_shared_initial/leader_board.jsonl
# INPUT_FILE=./data/dataset_shared/final_result.jsonl

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
