# NER タスクを解くために BERT を fine-tuning した

## NER (Named Entity Recognition)：固有表現抽出

使用モデル：bert-base-uncased\
データセット：wnut_17\
ディレクトリ構成\

```
.
├── output
├── saved_model/
│   ├── config.json
│   └── ...
├── trained_model/
│   ├── checkpoint
│   └── ...
├── function.py
├── test.py
└── train.py
```

### train.py

モデルを学習するためのファイル

### test.py

モデルをテストデータで評価するためのファイル

F1 で評価している

### function.py

使用する関数があるファイル
