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

### 2. データセット変換（COCO形式）

```bash
python convert_to_coco.py
```

### 3. モデル学習

```bash
python train.py
```

### 4. 推論・解析

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
