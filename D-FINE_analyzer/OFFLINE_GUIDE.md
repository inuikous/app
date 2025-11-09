# D-FINEオフライン使用ガイド

D-FINEモデルをインターネット接続なしで使用するための手順です。

## 前提条件

- Python 3.8以上
- 必要なパッケージがインストール済み（requirements.txt）
- 最初の1回だけインターネット接続が必要

## オフライン使用の準備（初回のみ）

### ステップ1: モデルのダウンロード

**インターネット接続がある環境で実行：**

```cmd
cd D-FINE_analyzer
python download_model.py
```

このスクリプトは以下を実行します：
- HuggingFace Hubから事前学習済みモデルをダウンロード（約1GB）
- `pretrained_models/dfine-xlarge-coco/` に保存

**出力例：**
```
======================================================================
D-FINEモデルのダウンロード
======================================================================
モデル: ustc-community/dfine-xlarge-coco
保存先: ./pretrained_models/dfine-xlarge-coco

1. Image Processorをダウンロード中...
   ✓ 完了

2. モデルをダウンロード中（約1GB、数分かかります）...
   ✓ 完了

======================================================================
ダウンロード完了！
======================================================================
保存場所: C:\exe\PatternMatching\app\D-FINE_analyzer\pretrained_models\dfine-xlarge-coco

以降はオフラインでも使用できます。
train.py と inference.py は自動的にローカルモデルを使用します。
======================================================================
```

### ステップ2: オフライン環境への移行

ダウンロードが完了したら、以下のディレクトリをオフライン環境にコピー：

```
D-FINE_analyzer/
├── pretrained_models/
│   └── dfine-xlarge-coco/  ← これをコピー
│       ├── config.json
│       ├── model.safetensors
│       ├── preprocessor_config.json
│       └── ...
```

## オフライン環境での使用

### 学習

```cmd
cd D-FINE_analyzer
python train.py
```

**動作確認：**
```
使用デバイス: cuda
ローカルモデルを使用: ./pretrained_models/dfine-xlarge-coco
モデルをロード中...
```

「ローカルモデルを使用」と表示されればオフライン動作成功です。

### 推論

```cmd
python inference.py --input ../test_dataset --ground-truth ../test_dataset/labels.csv
```

**動作確認：**
```
使用デバイス: cuda
ローカルモデルを使用: ./pretrained_models/dfine-xlarge-coco
チェックポイント読み込み: checkpoints/best_model.pth
```

## トラブルシューティング

### エラー: "Connection error"

**原因：** ローカルモデルが見つからず、オンラインからダウンロードしようとしている

**解決策：**
1. `pretrained_models/dfine-xlarge-coco/` ディレクトリが存在するか確認
2. ディレクトリ内に以下のファイルがあるか確認：
   - `config.json`
   - `model.safetensors` または `pytorch_model.bin`
   - `preprocessor_config.json`
3. ファイルが不足している場合は、`download_model.py` を再実行

### 確認方法

```cmd
dir pretrained_models\dfine-xlarge-coco
```

**期待される出力：**
```
config.json
model.safetensors
preprocessor_config.json
...
```

### モデルの再ダウンロード

```cmd
# ディレクトリを削除
rmdir /s /q pretrained_models

# 再ダウンロード
python download_model.py
```

## ディレクトリ構造

```
D-FINE_analyzer/
├── download_model.py          # モデルダウンロードスクリプト
├── train.py                   # 学習スクリプト（オフライン対応）
├── inference.py               # 推論スクリプト（オフライン対応）
├── pretrained_models/         # ローカルモデル保存先（.gitignoreに含む）
│   └── dfine-xlarge-coco/
│       ├── config.json
│       ├── model.safetensors
│       └── preprocessor_config.json
├── checkpoints/               # 学習済みチェックポイント
│   ├── best_model.pth
│   └── last_model.pth
└── coco_dataset/              # COCO形式データセット
    ├── train.json
    └── val.json
```

## 注意事項

1. **初回ダウンロードには約5-10分かかります**（回線速度による）
2. **約1GBのディスク容量が必要**です
3. **`pretrained_models/` は .gitignore に含まれています**
   - Gitリポジトリには含まれません
   - 各環境で個別にダウンロードが必要
4. **学習済みチェックポイント（checkpoints/）とは別物**です
   - `pretrained_models/`: HuggingFaceの事前学習モデル
   - `checkpoints/`: このプロジェクトで学習したモデル

## オフライン動作の仕組み

### 修正内容

**train.py と inference.py の変更：**

```python
# ローカルモデルを優先的に使用（オフライン対応）
local_model_path = "./pretrained_models/dfine-xlarge-coco"
if os.path.exists(local_model_path):
    print(f"ローカルモデルを使用: {local_model_path}")
    model_name_or_path = local_model_path
else:
    print(f"オンラインからモデル読み込み: {model_name}")
    print(f"  （オフライン使用には download_model.py を実行してください）")
    model_name_or_path = model_name

self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
self.model = AutoModelForObjectDetection.from_pretrained(
    model_name_or_path,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
```

この修正により：
- ローカルにモデルがあれば自動的に使用
- なければオンラインからダウンロード（従来通り）
- ユーザーは何も意識せずオフライン動作が可能

## まとめ

1. **初回準備（オンライン環境）：**
   ```cmd
   python download_model.py
   ```

2. **以降の使用（オフライン可）：**
   ```cmd
   python train.py
   python inference.py --input ../test_dataset
   ```

3. **動作確認：**
   - 「ローカルモデルを使用」と表示されればOK
   - インターネット接続不要
