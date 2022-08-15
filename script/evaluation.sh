EXPERIMENT_NAME=predict_helpful_votes
RUN_NAME=cl-tohoku_bert-base-japanese_lr1e-5

INPUT_FILE=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/training-val.jsonl
# INPUT_FILE=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/leader_board.jsonl
# INPUT_FILE=./data/predict/${EXPERIMENT_NAME}/${RUN_NAME}/final_result.jsonl

OUTPUT_DIR=./data/evaluation/${EXPERIMENT_NAME}/${RUN_NAME}/
mkdir -p ${OUTPUT_DIR}

python ./src/evaluation.py \
    --input_file ${INPUT_FILE} \
    --output_dir ${OUTPUT_DIR}
