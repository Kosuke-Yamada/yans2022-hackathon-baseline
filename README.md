# yans2022-hackathon-baseline

## 概要
- [NLP若手の会 (YANS) 第17回シンポジウム ハッカソン (2022)](https://yans.anlp.jp/entry/yans2022hackathon) におけるベースラインのソースコードです。
- データセットとして、[Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html)を利用し、商品レビュー情報を基に、商品ごとでレビューを役に立つ投票の多い順にランキングするタスク (商品レビューの役に立つ投票数ランキングタスク) に取り組みます。
- [日本語東北大BERT](https://huggingface.co/cl-tohoku/bert-base-japanese)を利用して、各レビューの役に立つ投票数を回帰予測し、その予測された投票数に基づきランキングを行なっています。

## 商品レビューの役に立つ投票数ランキングタスクの説明
- 商品インデックス (`product_idx`) ごとに、役に立つ投票数 (`helpful_votes`) を多い順にレビューインデックス (`review_idx`) をランキングするタスクです。
- 評価指標は、k=5のNormalized Discounted Cumulative Gain (NDCG@5) です。[scikit-learnのndcg_score関数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)を使用しています。
- リーダーボードには、`leader_board.jsonl`から商品ごとのランキングを下記の提出フォーマットにしたものを提出してください。
- 最終評価には、後日配布される`final_result.jsonl`にてリーダーボードと同様のフォーマットにしたものを提出してください。
- リーダーボード、および最終評価では、一部の商品の結果のみ反映されます。

### データセット概要
- 配布するデータセットは`./data/dataset_shared_initial.tar.xz`です。`training.jsonl`と`leader_board.jsonl`が含まれています。
- 後日、`final_result.jsonl`が追加されている`./data/dataset_shared.tar.xz`を配布します。

| データセット | 商品数 | レビュー数 | 平均レビュー数 |
|---|---|---|---|
| training.jsonl | 5323 | 148247 | 27.8 |
| leader_board.jsonl | 500 | 14597 | 29.2 |
| final_result.jsonl | 500 | 12314 | 24.6 |

### データのサンプル
- 学習データのサンプル (`training.jsonl`)
```
{'marketplace': 'JP',
 'product_title': 'イノセンス スタンダード版 [DVD]',
 'product_category': 'Video DVD',
 'star_rating': 5,
 'helpful_votes': 5,
 'vine': 'N',
 'verified_purchase': 'N',
 'review_headline': '見る人を選ぶ傑作',
 'review_body': '深く考える事に抵抗がある人は<br />見ないで下さい。<br />そんな人たちのこの映画に対する批判は、<br />うんざりです。<br />「わからなかった」ではなく<br />「本当にわかろうとする事が出来なかった」のでしょう。<br />わからない人にはわからない、<br />考えられる人、考えるのが好きな人にとっては、<br />一生考えていられる大きなテーマ達、たまりません。<br />まさにこの作品はは偉大な問題群でした。<br />美しい映像はそれを美しく表現してくれましたが、<br />断じて単独の主役ではありません。<br />この作品で私は一生楽しめる。<br />そんな作品です。',
 'review_date': '2004-09-18',
 'review_idx': 0,
 'product_idx': 32179,
 'customer_idx': 7903,
 'sets': 'training'}
```
- リーダーボードデータ (`leader_board.jsonl`) 、および最終評価データ (`final_result.jsonl`) には、役に立つ投票数 (`helpful_votes`) は含まれていません。
- レビューインデックス (`review_idx`) はレビューごとに一意の値が割り振られています。

### 提出フォーマット
- ファイル形式は1行ごとのJSONファイルである`.jsonl`としてください。ファイル名は、リーダーボードセットでは`submit_leader_board.jsonl`、最終評価セットでは`submit_final_result.jsonl`として下さい。
- 辞書のキーを`product_idx`と`pred_list`にし、キー`product_idx`に商品インデックス (`product_idx`) の値を、キー`pred_list`には、レビューインデックス (`review_idx`) と予測されたスコア (`pred_score`) の辞書のリストとしてください。
- `pred_list`がランキング順になっていなくても問題ありません。ただ、`pred_score`は値の大きい方がランキング上位であるようにしてください。

- 提出フォーマットのサンプル (`submit_leader_board.jsonl`)
```
{"product_idx":22,"pred_list":[{"review_idx":155312,"pred_score":0.3339654299},{"review_idx":177288,"pred_score":0.583666117},{"review_idx":3631,"pred_score":0.0624441049},{"review_idx":113220,"pred_score":0.6171164332},{"review_idx":150184,"pred_score":0.9106289978},{"review_idx":115705,"pred_score":1.7892944117},{"review_idx":149366,"pred_score":0.6061067085},{"review_idx":86796,"pred_score":0.468372023},{"review_idx":140831,"pred_score":-0.0919138368},{"review_idx":4329,"pred_score":0.5790296835},{"review_idx":24602,"pred_score":0.2002850354}]}
{"product_idx":64,"pred_list":[{"review_idx":44102,"pred_score":1.8199084533},{"review_idx":71874,"pred_score":0.6788325688},{"review_idx":100851,"pred_score":0.4426570388},{"review_idx":188581,"pred_score":0.9793278513},{"review_idx":126652,"pred_score":1.0696557773},{"review_idx":62176,"pred_score":0.6578231444},{"review_idx":69768,"pred_score":1.6456227195},{"review_idx":13327,"pred_score":0.7172244864},{"review_idx":186064,"pred_score":1.2599495407},{"review_idx":157079,"pred_score":0.1114082466},{"review_idx":253671,"pred_score":0.3962000566},{"review_idx":243439,"pred_score":1.5728116733},{"review_idx":82305,"pred_score":0.5438685364},{"review_idx":202352,"pred_score":0.7125824421},{"review_idx":216472,"pred_score":0.4005214799},{"review_idx":3311,"pred_score":1.0686458198},{"review_idx":193877,"pred_score":1.4004426256},{"review_idx":52543,"pred_score":0.6763690467},{"review_idx":60410,"pred_score":1.6896807335}]}
{"product_idx":200,"pred_list":[{"review_idx":257671,"pred_score":0.2672373926},{"review_idx":235635,"pred_score":1.1028408836},{"review_idx":182631,"pred_score":0.2906559999},{"review_idx":223680,"pred_score":0.3793692582},{"review_idx":152310,"pred_score":1.7258097401},{"review_idx":104869,"pred_score":0.8575741157},{"review_idx":147491,"pred_score":0.6676823582},{"review_idx":160561,"pred_score":1.201371467},{"review_idx":183898,"pred_score":0.3852049438},{"review_idx":54647,"pred_score":0.2401906035},{"review_idx":38966,"pred_score":0.38715112},{"review_idx":176161,"pred_score":0.5829516696},{"review_idx":96375,"pred_score":0.7827028407}]}
```

## インストール
### 1. リポジトリをクローン
```sh
git clone https://github.com/Kosuke-Yamada/yans2022-hackathon-baseline.git
```

### 2. 必要なライブラリの読み込み
a. `environment.yml`からconda環境を作成
```sh
conda env create --file environment.yml
conda activate yans2022-hackathon
```

b. `requirements.txt`からライブラリの読み込み
```sh
python -m pip install -r requirements.txt
```

## 利用方法
- ソースコードは`./src/`に入っており、対応する全ての実行用スクリプトは`./script/`に入っています。
- Jupyter Notebook版が`./notebook/`に入っています。
- データセットや処理されたデータなどは、デフォルトでは`./data/`に格納されます。

### 0. データセットの獲得
- 今回のハッカソンで使用するデータが`./data/dataset_shared_initial.tar.xz`にあります。それを解凍してください。
- 圧縮ファイルには、`training.jsonl`と`leader_board.jsonl`が入っています。
- ハッカソン終盤に`final_result.jsonl`を含む`./data/dataset_shared.tar.xz`を配布します。

#### 実行コード
```
tar Jxfv ./data/dataset_shared_initial.tar.xz -C ./data/
# tar Jxfv ./data/dataset_shared.tar.xz -C ./data/
```

### 1. 前処理 (`preprocessing.py`, `preprocessing.ipynb`)
- `training.jsonl`のデータを学習セットと開発セットに分割しています。

#### 実行コード
```sh
sh ./scripts/preprocessing.sh
```

#### preprocessing.shの例
- `--n_train`と`--n_val`は，学習セットと開発セットとして使用する商品数であり、そのレビューデータを含む。
- `--n_train`を指定しないとき、`--n_val`で指定した商品数のレビューデータ以外の全てを学習セットとする。
```sh
mkdir -p ./data/preprocessing_shared/

python ./src/preprocessing.py \
    --input_file ./data/dataset_shared/training.jsonl \
    --output_dir ./data/preprocessing_shared/ \
    --n_train 10 \
    --n_val 10 \
    --random_state 0
```

### 2. 学習 (`train.py`, `train.ipynb`)
- 日本語東北大BERTを用いて、レビュー本文 (`review_body`) を入力とし、対数変換された役に立つ投票数を予測するように学習する。

#### 実行コード
```sh
sh ./scripts/train.sh
```

#### train.shの例
```sh
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
```

### 3. 推論 (`predict.py`, `predict.ipynb`)
- 学習されたBERTを用いて、レビュー本文を入力とし、対数変換された役に立つ投票数を予測する。

#### 実行コード
```sh
sh ./scripts/predict.sh
```

#### predict.shの例
```sh
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
```

### 4. 評価 (`evaluation.py`, `evaluation.ipynb`)
- 予測された役に立つ投票数を提出フォーマットに変換する。
- また、役に立つ投票数を含むデータの場合、評価スコアであるndcg@5を算出する。

#### 実行コード
```sh
sh ./scripts/evaluation.sh
```

#### evaluation.shの例
```sh
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
```

### 0~4の処理を動かす
- 一通り全て動かしたい場合、以下のコードを実行してください。

#### 実行コード
```sh
sh ./scripts/process_all.sh
```