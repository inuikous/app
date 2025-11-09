# D-FINE Analyzer

D-FINEモデルを使用した高精度物体検出システム

## 概要

HuggingFaceのD-FINEモデルを使用して、画像内のハート、アルファベット文字（A-F）を検出します。

## ディレクトリ構成

```
D-FINE_analyzer/
├── convert_to_coco.py      # データセットをCOCO形式に変換
├── train.py                # D-FINEモデルの学習
├── inference.py            # 推論・解析実行
├── utils.py                # ユーティリティ関数
├── requirements.txt        # 依存パッケージ
├── configs/                # 設定ファイル
│   └── train_config.yaml
├── outputs/                # 解析結果
│   ├── images/            # 解析済み画像
│   ├── plots/             # 精度グラフ
│   └── analysis_results.csv
├── coco_dataset/          # COCO形式データセット
│   ├── annotations/
│   └── images/
└── checkpoints/           # 学習済みモデル
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
cd D-FINE_analyzer
pip install -r requirements.txt
```

### 2. モデルのダウンロード（初回のみ、オフライン使用の場合）

```bash
python download_model.py
```

このステップで事前学習済みモデルをローカルに保存します（約1GB）。
以降はインターネット接続なしで学習・推論が可能になります。

詳細は [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) を参照してください。

### 3. データセット変換（COCO形式）

```bash
python convert_to_coco.py
```

### 4. モデル学習

```bash
python train.py
```

### 5. 推論・解析

```bash
python inference.py --input ../dataset
```

## 使用方法

### 基本的な解析

```bash
python inference.py --input ../dataset
```

### カスタムモデルを使用

```bash
python inference.py --input ../dataset --checkpoint checkpoints/best_model.pth
```

### 精度評価

```bash
python inference.py --input ../dataset --ground-truth ../dataset/labels.csv --plot
```

## クラス定義

- 0: heart
- 1: A
- 2: B
- 3: C
- 4: D
- 5: E
- 6: F

## 出力形式

analyzer/と同じ形式で出力されます：
- `outputs/images/` - 解析済み画像
- `outputs/plots/` - 精度グラフ
- `outputs/analysis_results.csv` - 解析結果CSV

## オフライン使用

インターネット接続なしで使用する場合：

1. **初回準備（オンライン環境で実行）：**
   ```bash
   python download_model.py
   ```

2. **以降の使用（オフライン可）：**
   ```bash
   python train.py
   python inference.py --input ../dataset
   ```

詳細な手順とトラブルシューティングは [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) を参照してください。

## 関連ドキュメント

- [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) - オフライン使用の詳細ガイド
- [QUICKSTART.md](QUICKSTART.md) - クイックスタートガイド
- [../README.md](../README.md) - プロジェクト全体のREADME
